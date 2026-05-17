"""
tests/test_memory_model.py
--------------------------
Unit tests for VRAM estimation.

Run with:  python -m pytest tests/ -v
"""

import math
import sys
from pathlib import Path

# Allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).parent.parent))

from hpc_planner.planner.memory_model import (
    ModelSpec, ModelArch, Precision, Optimizer, ParallelStrategy,
    estimate_training_memory, estimate_inference_memory, min_gpus_required,
    BYTES_PER_PARAM, OPTIMIZER_BYTES,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def small_transformer() -> ModelSpec:
    return ModelSpec(
        params_millions=125,      # GPT-2 small
        architecture=ModelArch.TRANSFORMER,
        precision=Precision.FP16,
        optimizer=Optimizer.ADAMW,
        strategy=ParallelStrategy.DDP,
        batch_size=32,
        seq_len=1024,
        grad_checkpointing=False,
    )


def large_transformer() -> ModelSpec:
    return ModelSpec(
        params_millions=7000,     # LLaMA-2 7B
        architecture=ModelArch.TRANSFORMER,
        precision=Precision.FP16,
        optimizer=Optimizer.ADAMW,
        strategy=ParallelStrategy.DDP,
        batch_size=256,
        seq_len=2048,
        grad_checkpointing=False,
    )


# ── Weight memory tests ───────────────────────────────────────────────────────

def test_weight_memory_fp16():
    spec = small_transformer()
    mem = estimate_training_memory(spec, world_size=1)
    expected = (125e6 * 2.0) / 1e9   # 0.25 GB
    assert abs(mem.weights_gb - expected) < 0.05, (
        f"Expected ~{expected:.2f} GB, got {mem.weights_gb:.2f} GB"
    )


def test_weight_memory_fp32():
    spec = small_transformer()
    spec.precision = Precision.FP32
    mem = estimate_training_memory(spec, world_size=1)
    expected = (125e6 * 4.0) / 1e9   # 0.5 GB
    assert abs(mem.weights_gb - expected) < 0.05


def test_weight_memory_scales_with_params():
    spec_small = small_transformer()
    spec_large = large_transformer()
    mem_small = estimate_training_memory(spec_small, world_size=1)
    mem_large = estimate_training_memory(spec_large, world_size=1)
    ratio = spec_large.params_millions / spec_small.params_millions
    actual_ratio = mem_large.weights_gb / mem_small.weights_gb
    assert abs(actual_ratio - ratio) < 1.0, (
        f"Weight memory should scale linearly with params. "
        f"Expected ratio {ratio:.0f}, got {actual_ratio:.1f}"
    )


# ── Optimizer memory tests ────────────────────────────────────────────────────

def test_adamw_optimizer_memory():
    """AdamW should cost 12 bytes/param total (master weights + 2 moments)."""
    spec = small_transformer()
    mem = estimate_training_memory(spec, world_size=1)
    expected = (125e6 * 12.0) / 1e9   # 1.5 GB
    assert abs(mem.optimizer_gb - expected) < 0.1, (
        f"AdamW optimizer expected {expected:.2f} GB, got {mem.optimizer_gb:.2f} GB"
    )


def test_sgd_optimizer_cheaper_than_adamw():
    spec_adam = small_transformer()
    spec_sgd  = small_transformer()
    spec_sgd.optimizer = Optimizer.SGD
    mem_adam = estimate_training_memory(spec_adam, world_size=1)
    mem_sgd  = estimate_training_memory(spec_sgd, world_size=1)
    assert mem_sgd.optimizer_gb < mem_adam.optimizer_gb, (
        "SGD should use less optimizer memory than AdamW"
    )


# ── Gradient checkpointing tests ──────────────────────────────────────────────

def test_grad_checkpointing_reduces_activations():
    spec_no_ckpt = large_transformer()
    spec_ckpt    = large_transformer()
    spec_ckpt.grad_checkpointing = True

    mem_no = estimate_training_memory(spec_no_ckpt, world_size=8)
    mem_ck = estimate_training_memory(spec_ckpt,    world_size=8)

    assert mem_ck.activations_gb < mem_no.activations_gb, (
        "Gradient checkpointing should reduce activation memory"
    )
    ratio = mem_ck.activations_gb / mem_no.activations_gb
    assert ratio < 0.5, (
        f"Expected >50% activation reduction with checkpointing, got {ratio:.2f}"
    )


# ── FSDP sharding tests ───────────────────────────────────────────────────────

def test_fsdp_reduces_per_gpu_memory():
    spec_ddp  = large_transformer()
    spec_fsdp = large_transformer()
    spec_fsdp.strategy = ParallelStrategy.FSDP

    world_size = 8
    mem_ddp  = estimate_training_memory(spec_ddp,  world_size=world_size)
    mem_fsdp = estimate_training_memory(spec_fsdp, world_size=world_size)

    assert mem_fsdp.total_gb < mem_ddp.total_gb, (
        "FSDP should reduce per-GPU memory compared to DDP"
    )


# ── Inference memory tests ────────────────────────────────────────────────────

def test_inference_memory_less_than_training():
    spec = large_transformer()
    mem_train = estimate_training_memory(spec, world_size=1)
    mem_infer = estimate_inference_memory(spec, world_size=1)
    assert mem_infer.total_gb < mem_train.total_gb
    assert mem_infer.gradients_gb == 0.0
    assert mem_infer.optimizer_gb == 0.0


# ── min_gpus_required tests ───────────────────────────────────────────────────

def test_min_gpus_required_basic():
    mem = estimate_training_memory(large_transformer(), world_size=1)
    # 7B model with AdamW on FP16 should require more than 1 A100-80GB
    n = min_gpus_required(mem, gpu_vram_gb=80.0)
    assert n >= 2, f"Expected ≥2 GPUs for 7B AdamW, got {n}"


def test_min_gpus_required_small_model():
    mem = estimate_training_memory(small_transformer(), world_size=1)
    n = min_gpus_required(mem, gpu_vram_gb=80.0)
    assert n == 1, f"GPT-2 small should fit on 1 A100-80GB, got {n}"


def test_total_memory_is_sum_of_components():
    mem = estimate_training_memory(large_transformer(), world_size=4)
    computed = (mem.weights_gb + mem.gradients_gb + mem.optimizer_gb +
                mem.activations_gb + mem.overhead_gb)
    assert abs(mem.total_gb - computed) < 1e-6


# ── Run if executed directly ──────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_weight_memory_fp16,
        test_weight_memory_fp32,
        test_weight_memory_scales_with_params,
        test_adamw_optimizer_memory,
        test_sgd_optimizer_cheaper_than_adamw,
        test_grad_checkpointing_reduces_activations,
        test_fsdp_reduces_per_gpu_memory,
        test_inference_memory_less_than_training,
        test_min_gpus_required_basic,
        test_min_gpus_required_small_model,
        test_total_memory_is_sum_of_components,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  ✓  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  ✗  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{passed+failed} tests passed")
    if failed:
        sys.exit(1)
