"""
slurm_emitter.py
----------------
Derives optimal SLURM resource parameters and emits a job script.

Design decision: parameters are derived analytically from the memory model
and communication model — not by heuristics or lookup tables. Each parameter
has a documented rationale.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

from .memory_model import (
    ModelSpec, MemoryBreakdown, ParallelStrategy, Precision,
    estimate_training_memory, min_gpus_required,
)
from .comm_model import Interconnect, estimate_comm_efficiency, INTERCONNECT_SPECS


@dataclass
class ClusterSpec:
    """Describes the target HPC cluster."""
    gpu_vram_gb: float = 80.0
    gpu_name: str = "A100 80GB"
    gpus_per_node: int = 8
    cpu_cores_per_node: int = 128
    ram_per_node_gb: float = 512.0
    interconnect: Interconnect = Interconnect.INFINIBAND400
    max_nodes: int = 32
    has_nvlink: bool = True


@dataclass
class SlurmParams:
    """Derived SLURM parameters with explanations."""
    nodes: int
    ntasks_per_node: int
    gpus_per_node: int
    cpus_per_task: int
    mem_gb: int
    ntasks_total: int
    local_batch_size: int
    world_size: int

    # Flags
    use_fsdp: bool = False
    use_gradient_checkpointing: bool = False

    # Warnings
    warnings: list[str] = field(default_factory=list)

    @property
    def slurm_directives(self) -> list[str]:
        return [
            f"#SBATCH --nodes={self.nodes}",
            f"#SBATCH --ntasks-per-node={self.ntasks_per_node}",
            f"#SBATCH --gres=gpu:{self.gpus_per_node}",
            f"#SBATCH --cpus-per-task={self.cpus_per_task}",
            f"#SBATCH --mem={self.mem_gb}G",
            f"#SBATCH --ntasks={self.ntasks_total}",
        ]


def derive_slurm_params(
    model: ModelSpec,
    cluster: ClusterSpec,
    job_time: str = "24:00:00",
    partition: str = "gpu",
) -> tuple[SlurmParams, str]:
    """
    Derive SLURM parameters and emit a complete job script.

    Returns
    -------
    params : SlurmParams
        All derived parameters with explanations.
    script : str
        Complete SLURM batch script, ready to submit.
    """
    warnings: list[str] = []

    # ── Step 1: Minimum GPUs from memory budget ──────────────────────────────
    # Start with single-GPU estimate, then iterate
    probe_mem = estimate_training_memory(model, world_size=1)
    min_gpus  = min_gpus_required(probe_mem, cluster.gpu_vram_gb)

    # Round up to full nodes
    min_gpus = math.ceil(min_gpus / cluster.gpus_per_node) * cluster.gpus_per_node
    max_gpus = cluster.max_nodes * cluster.gpus_per_node
    rec_gpus = min(min_gpus, max_gpus)

    # ── Step 2: Scale up for local batch throughput ───────────────────────────
    # Local batch < 4 starves the GPU pipeline; scale world_size up to fix it
    local_batch = max(1, model.batch_size // rec_gpus)
    if local_batch < 4 and rec_gpus < max_gpus:
        needed_gpus = math.ceil(model.batch_size / 4)
        needed_gpus = math.ceil(needed_gpus / cluster.gpus_per_node) * cluster.gpus_per_node
        rec_gpus = min(needed_gpus, max_gpus)

    rec_nodes = math.ceil(rec_gpus / cluster.gpus_per_node)
    local_batch = max(1, model.batch_size // rec_gpus)

    # ── Step 3: Re-estimate memory with correct world_size ───────────────────
    mem = estimate_training_memory(model, world_size=rec_gpus,
                                   local_batch_override=local_batch)

    # ── Step 4: CPUs per task ─────────────────────────────────────────────────
    # Allocate DataLoader workers: total_cores / gpus_per_node
    # Cap at 16 — beyond this, multiprocessing overhead exceeds benefit
    cpus_per_task = min(
        math.floor(cluster.cpu_cores_per_node / cluster.gpus_per_node) * 2,
        16,
    )
    cpus_per_task = max(cpus_per_task, 2)  # always at least 2

    # ── Step 5: Node memory ───────────────────────────────────────────────────
    # Reserve 15% for OS, NCCL shared memory, and burst buffers
    mem_gb = int(cluster.ram_per_node_gb * 0.85)

    # ── Step 6: Communication efficiency ─────────────────────────────────────
    comm = estimate_comm_efficiency(
        cluster.interconnect, rec_gpus,
        model.params_millions, rec_nodes,
    )

    # ── Step 7: Collect warnings ──────────────────────────────────────────────
    mem_util = mem.total_gb / cluster.gpu_vram_gb
    if mem_util > 0.92:
        warnings.append(
            f"VRAM usage {mem.total_gb:.1f}GB is {mem_util*100:.0f}% of "
            f"{cluster.gpu_vram_gb}GB — enable gradient checkpointing or reduce batch size."
        )
    if local_batch < 4:
        warnings.append(
            f"Local batch size {local_batch} is very small. "
            "Consider gradient accumulation to maintain effective batch size."
        )
    if not cluster.has_nvlink and rec_gpus > cluster.gpus_per_node:
        warnings.append(
            f"{cluster.gpu_name} lacks NVLink — multi-node all-reduce will "
            "bottleneck throughput significantly."
        )
    if cluster.interconnect == Interconnect.ETHERNET100 and rec_nodes > 1:
        warnings.append(
            "100GbE is slow for multi-node training. "
            "Expect 25–40% throughput reduction versus InfiniBand."
        )
    if comm.overall_comm_efficiency < 0.70:
        warnings.append(
            f"Estimated communication efficiency is low "
            f"({comm.overall_comm_efficiency*100:.0f}%). "
            "Consider gradient compression or reducing world size."
        )

    params = SlurmParams(
        nodes=rec_nodes,
        ntasks_per_node=cluster.gpus_per_node,
        gpus_per_node=cluster.gpus_per_node,
        cpus_per_task=cpus_per_task,
        mem_gb=mem_gb,
        ntasks_total=rec_gpus,
        local_batch_size=local_batch,
        world_size=rec_gpus,
        use_fsdp=(model.strategy == ParallelStrategy.FSDP),
        use_gradient_checkpointing=model.grad_checkpointing,
        warnings=warnings,
    )

    script = _emit_script(params, model, cluster, comm, mem, job_time, partition)
    return params, script


def _emit_script(
    p: SlurmParams,
    model: ModelSpec,
    cluster: ClusterSpec,
    comm,
    mem: MemoryBreakdown,
    job_time: str,
    partition: str,
) -> str:
    """Render the complete SLURM batch script."""
    ic = INTERCONNECT_SPECS[cluster.interconnect]

    framework_launch = _framework_launch(model, p, cluster)
    nccl_exports = "\n".join(
        f"export {k}={v}" for k, v in comm.nccl_flags.items()
    )

    warn_comments = ""
    if p.warnings:
        warn_comments = "\n" + "\n".join(f"# WARNING: {w}" for w in p.warnings)

    extra_flags = []
    if p.use_fsdp:
        extra_flags.append("    --fsdp_sharding_strategy FULL_SHARD \\")
    if p.use_gradient_checkpointing:
        extra_flags.append("    --gradient_checkpointing \\")
    if model.precision in (Precision.FP16, Precision.BF16):
        extra_flags.append("    --bf16 \\")
    extra_flags_str = "\n".join(extra_flags) + ("\n" if extra_flags else "")

    dl_workers = min(p.cpus_per_task - 1, 8)

    script = f"""#!/bin/bash
# ============================================================
# Auto-generated by hpc-auto-resource-planner
# Model  : {model.params_millions:.0f}M params  |  {model.precision.value}  |  {model.strategy.value}
# GPU    : {p.world_size}× {cluster.gpu_name}  ({p.nodes} node{"s" if p.nodes>1 else ""})
# Memory : {mem.total_gb:.1f} GB / {cluster.gpu_vram_gb:.0f} GB per GPU
# Comm   : {ic.name}  |  efficiency ≈ {comm.overall_comm_efficiency*100:.0f}%
# ============================================================
{warn_comments}

#SBATCH --job-name=train_{model.architecture.value}_{model.params_millions:.0f}M
#SBATCH --nodes={p.nodes}
#SBATCH --ntasks-per-node={p.ntasks_per_node}
#SBATCH --gres=gpu:{p.gpus_per_node}
#SBATCH --cpus-per-task={p.cpus_per_task}
#SBATCH --mem={p.mem_gb}G
#SBATCH --ntasks={p.ntasks_total}
#SBATCH --time={job_time}
#SBATCH --partition={partition}
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ── Environment ─────────────────────────────────────────────
module purge
module load cuda/12.1 nccl/2.18 cudnn/8.9

{nccl_exports}
export OMP_NUM_THREADS={p.cpus_per_task}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# ── Verify GPU allocation ────────────────────────────────────
echo "Allocated GPUs: $SLURM_GPUS_ON_NODE"
echo "Nodes: $SLURM_JOB_NODELIST"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

mkdir -p logs

# ── Launch ───────────────────────────────────────────────────
{framework_launch}
    --batch_size {p.local_batch_size} \\
    --global_batch_size {model.batch_size} \\
{extra_flags_str}    --num_workers {dl_workers} \\
    --output_dir checkpoints/
"""
    return script.strip()


def _framework_launch(model: ModelSpec, p: SlurmParams, cluster: ClusterSpec) -> str:
    from .memory_model import ModelSpec as _MS

    fw = model.architecture  # we don't have a framework field — infer from strategy
    strategy = model.strategy

    if strategy == ParallelStrategy.FSDP:
        return (f"torchrun \\\n"
                f"    --nnodes={p.nodes} \\\n"
                f"    --nproc-per-node={cluster.gpus_per_node} \\\n"
                f"    --rdzv-backend=c10d \\\n"
                f"    --rdzv-endpoint=$SLURM_NODELIST:29500 \\\n"
                f"    train.py \\")
    elif strategy == ParallelStrategy.DDP:
        return (f"torchrun \\\n"
                f"    --nnodes={p.nodes} \\\n"
                f"    --nproc-per-node={cluster.gpus_per_node} \\\n"
                f"    --rdzv-backend=c10d \\\n"
                f"    --rdzv-endpoint=$SLURM_NODELIST:29500 \\\n"
                f"    train.py \\")
    else:
        return (f"srun python -m torch.distributed.run \\\n"
                f"    --nproc-per-node={cluster.gpus_per_node} \\\n"
                f"    train.py \\")
