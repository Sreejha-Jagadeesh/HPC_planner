"""
memory_model.py
---------------
Estimates GPU VRAM usage for training and inference across all memory components.

Components modelled:
  - Model weights
  - Gradients (always FP32 in mixed-precision)
  - Optimizer states (framework-specific)
  - Activations (architecture-specific, gradient-checkpointing aware)
  - Framework/CUDA overhead

Reference:
  Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion
  Parameter Models", SC'20. https://arxiv.org/abs/1910.02054
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Precision(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    FP8  = "fp8"


class Optimizer(str, Enum):
    ADAMW    = "adamw"
    SGD      = "sgd"
    LAMB     = "lamb"
    ADAFACTOR = "adafactor"


class ModelArch(str, Enum):
    TRANSFORMER = "transformer"
    CNN         = "cnn"
    RNN         = "rnn"
    GNN         = "gnn"
    DIFFUSION   = "diffusion"


class ParallelStrategy(str, Enum):
    DDP    = "ddp"
    FSDP   = "fsdp"
    MODEL  = "model_parallel"
    HYBRID = "hybrid"


# Bytes per parameter for each precision
BYTES_PER_PARAM: dict[Precision, float] = {
    Precision.FP32: 4.0,
    Precision.FP16: 2.0,
    Precision.BF16: 2.0,
    Precision.INT8: 1.0,
    Precision.FP8:  0.5,
}

# Optimizer state bytes-per-param multipliers (on top of FP32 master weights)
# AdamW: 2 momentum tensors (FP32) + master weights (FP32) = 12 bytes/param
# SGD: master weights only = 4 bytes/param
# LAMB: same as AdamW = 12 bytes/param
# Adafactor: factored second moments ≈ 1.5× model
OPTIMIZER_BYTES: dict[Optimizer, float] = {
    Optimizer.ADAMW:    12.0,
    Optimizer.SGD:       4.0,
    Optimizer.LAMB:     12.0,
    Optimizer.ADAFACTOR: 6.0,
}

# Sharding efficiency per strategy (fraction of optimizer/grad per GPU after sharding)
# DDP: full replica on each GPU
# FSDP: sharded across world_size GPUs → 1/world_size (modelled as efficiency)
SHARDING_EFFICIENCY: dict[ParallelStrategy, float] = {
    ParallelStrategy.DDP:    1.0,
    ParallelStrategy.FSDP:   0.95,  # slight overhead vs ideal 1/N due to buffers
    ParallelStrategy.MODEL:  0.90,
    ParallelStrategy.HYBRID: 0.85,
}


@dataclass
class ModelSpec:
    """Describes the model to be trained or inferred."""
    params_millions: float          # Total parameter count in millions
    architecture: ModelArch = ModelArch.TRANSFORMER
    precision: Precision = Precision.FP16
    optimizer: Optimizer = Optimizer.ADAMW
    strategy: ParallelStrategy = ParallelStrategy.DDP
    batch_size: int = 32            # Global batch size
    seq_len: int = 2048             # Sequence length (transformers) / input size
    grad_checkpointing: bool = False


@dataclass
class MemoryBreakdown:
    """Per-GPU VRAM usage split by component (GB)."""
    weights_gb: float
    gradients_gb: float
    optimizer_gb: float
    activations_gb: float
    overhead_gb: float

    @property
    def total_gb(self) -> float:
        return (self.weights_gb + self.gradients_gb +
                self.optimizer_gb + self.activations_gb + self.overhead_gb)

    def as_dict(self) -> dict:
        return {
            "weights":    round(self.weights_gb, 3),
            "gradients":  round(self.gradients_gb, 3),
            "optimizer":  round(self.optimizer_gb, 3),
            "activations":round(self.activations_gb, 3),
            "overhead":   round(self.overhead_gb, 3),
            "total":      round(self.total_gb, 3),
        }

    def __str__(self) -> str:
        lines = [
            f"  Weights       : {self.weights_gb:7.2f} GB",
            f"  Gradients     : {self.gradients_gb:7.2f} GB",
            f"  Optimizer     : {self.optimizer_gb:7.2f} GB",
            f"  Activations   : {self.activations_gb:7.2f} GB",
            f"  Overhead      : {self.overhead_gb:7.2f} GB",
            f"  ─────────────────────────",
            f"  Total         : {self.total_gb:7.2f} GB",
        ]
        return "\n".join(lines)


def estimate_training_memory(
    spec: ModelSpec,
    world_size: int = 1,
    local_batch_override: Optional[int] = None,
) -> MemoryBreakdown:
    """
    Estimate per-GPU training VRAM for the given model and world size.

    Parameters
    ----------
    spec : ModelSpec
        Model and training configuration.
    world_size : int
        Total number of GPUs (nodes × gpus_per_node).
    local_batch_override : int, optional
        If provided, use this as the per-GPU batch size instead of
        spec.batch_size / world_size.
    """
    params = spec.params_millions * 1e6
    prec_bytes = BYTES_PER_PARAM[spec.precision]
    opt_bytes  = OPTIMIZER_BYTES[spec.optimizer]
    sharding   = SHARDING_EFFICIENCY[spec.strategy]

    local_batch = local_batch_override or max(1, spec.batch_size // world_size)

    # ── Weights ──────────────────────────────────────────────────────────────
    weights_gb = (params * prec_bytes) / 1e9

    # FSDP/model-parallel shard weights across GPUs
    if spec.strategy in (ParallelStrategy.FSDP, ParallelStrategy.MODEL,
                         ParallelStrategy.HYBRID):
        weights_gb = weights_gb / world_size * (1 + (1 - sharding))

    # ── Gradients ─────────────────────────────────────────────────────────────
    # Gradients kept in FP32 regardless of forward-pass precision
    grads_gb = (params * 4.0) / 1e9
    if spec.strategy == ParallelStrategy.FSDP:
        grads_gb /= world_size

    # ── Optimizer states ──────────────────────────────────────────────────────
    opt_gb = (params * opt_bytes) / 1e9
    if spec.strategy in (ParallelStrategy.FSDP, ParallelStrategy.HYBRID):
        opt_gb /= world_size

    # ── Activations ───────────────────────────────────────────────────────────
    act_gb = _estimate_activations(spec, local_batch)
    if spec.grad_checkpointing:
        act_gb *= 0.33  # sqrt(L) recomputation reduces activations ~3×

    # ── Overhead ──────────────────────────────────────────────────────────────
    # CUDA context + NCCL buffers + framework reserved
    overhead_gb = 2.5

    return MemoryBreakdown(
        weights_gb=weights_gb,
        gradients_gb=grads_gb,
        optimizer_gb=opt_gb,
        activations_gb=act_gb,
        overhead_gb=overhead_gb,
    )


def estimate_inference_memory(
    spec: ModelSpec,
    world_size: int = 1,
) -> MemoryBreakdown:
    """
    Estimate per-GPU VRAM for inference (no optimizer states, smaller activations).
    """
    params = spec.params_millions * 1e6
    prec_bytes = BYTES_PER_PARAM[spec.precision]

    weights_gb  = (params * prec_bytes) / 1e9 / max(world_size, 1)
    grads_gb    = 0.0
    opt_gb      = 0.0
    local_batch = max(1, spec.batch_size // world_size)
    act_gb      = _estimate_activations(spec, local_batch) * 0.25
    overhead_gb = 1.0

    return MemoryBreakdown(
        weights_gb=weights_gb,
        gradients_gb=grads_gb,
        optimizer_gb=opt_gb,
        activations_gb=act_gb,
        overhead_gb=overhead_gb,
    )


def _estimate_activations(spec: ModelSpec, local_batch: int) -> float:
    """Architecture-specific activation memory estimate in GB."""
    seq = spec.seq_len
    prec_bytes = BYTES_PER_PARAM[spec.precision]

    if spec.architecture == ModelArch.TRANSFORMER:
        # Activation memory ≈ 12 × seq_len × batch × hidden_dim × num_layers
        # hidden_dim ≈ sqrt(params / (12 × num_layers))  [rough approximation]
        # For simplicity: empirical ≈ batch × seq × sqrt(params_M) × 2 bytes / 1e9
        # capped at 40 GB to avoid runaway estimates for tiny seq/large models
        import math
        act_gb = (local_batch * seq * 12.0 * math.sqrt(spec.params_millions)
                  * prec_bytes) / 1e9
        return min(act_gb, 40.0)

    elif spec.architecture == ModelArch.CNN:
        # Feature maps for ResNet-scale models
        h = w = 224
        act_gb = (local_batch * h * w * 3 * 4.0 * 4.0) / 1e9
        return max(act_gb, 0.05)

    elif spec.architecture == ModelArch.RNN:
        act_gb = (local_batch * seq * 512 * prec_bytes) / 1e9
        return max(act_gb, 0.01)

    elif spec.architecture == ModelArch.GNN:
        # Graph activations scale with node count; use seq_len as proxy
        act_gb = (local_batch * seq * 256 * prec_bytes) / 1e9
        return max(act_gb, 0.01)

    elif spec.architecture == ModelArch.DIFFUSION:
        # U-Net activations (latent diffusion)
        act_gb = (local_batch * 64 * 64 * 4 * 10 * prec_bytes) / 1e9
        return max(act_gb, 0.1)

    return 1.0  # fallback


def min_gpus_required(mem: MemoryBreakdown, gpu_vram_gb: float,
                      safety_margin: float = 0.90) -> int:
    """Minimum GPUs needed so that per-GPU usage fits within VRAM."""
    import math
    usable = gpu_vram_gb * safety_margin
    return max(1, math.ceil(mem.total_gb / usable))
