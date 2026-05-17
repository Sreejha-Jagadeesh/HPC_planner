#!/usr/bin/env python3
"""
hpc_planner/__main__.py
-----------------------
CLI entrypoint for the HPC Auto Resource Planner.

Usage
-----
  python -m hpc_planner \\
      --params-m 7000 \\
      --precision fp16 \\
      --batch 256 \\
      --seq-len 2048 \\
      --strategy fsdp \\
      --gpu-type a100_80 \\
      --gpus-per-node 8 \\
      --cpu-per-node 128 \\
      --ram-gb 512 \\
      --interconnect infiniband400 \\
      --max-nodes 32 \\
      --output-script job.sh
"""

import argparse
import sys
from pathlib import Path

from .planner.memory_model import (
    ModelSpec, ModelArch, Precision, Optimizer, ParallelStrategy,
    estimate_training_memory, estimate_inference_memory,
)
from .planner.comm_model import Interconnect
from .planner.slurm_emitter import ClusterSpec, derive_slurm_params


GPU_PRESETS = {
    "a100_80":  dict(vram=80,  name="A100 80GB",   has_nvlink=True),
    "a100_40":  dict(vram=40,  name="A100 40GB",   has_nvlink=True),
    "h100_80":  dict(vram=80,  name="H100 80GB",   has_nvlink=True),
    "v100_32":  dict(vram=32,  name="V100 32GB",   has_nvlink=True),
    "rtx4090":  dict(vram=24,  name="RTX 4090",    has_nvlink=False),
    "mi300x":   dict(vram=192, name="MI300X",      has_nvlink=False),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="hpc_planner",
        description="Derive optimal SLURM parameters for your training job.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Model
    mg = p.add_argument_group("Model")
    mg.add_argument("--params-m", type=float, default=7000,
                    help="Model parameters in millions (default: 7000)")
    mg.add_argument("--arch", choices=[a.value for a in ModelArch],
                    default="transformer")
    mg.add_argument("--precision", choices=[pr.value for pr in Precision],
                    default="fp16")
    mg.add_argument("--optimizer", choices=[o.value for o in Optimizer],
                    default="adamw")
    mg.add_argument("--strategy", choices=[s.value for s in ParallelStrategy],
                    default="ddp")
    mg.add_argument("--batch", type=int, default=256)
    mg.add_argument("--seq-len", type=int, default=2048)
    mg.add_argument("--grad-checkpointing", action="store_true")

    # Cluster
    cg = p.add_argument_group("Cluster")
    cg.add_argument("--gpu-type", choices=list(GPU_PRESETS.keys()), default="a100_80")
    cg.add_argument("--gpus-per-node", type=int, default=8)
    cg.add_argument("--cpu-per-node", type=int, default=128)
    cg.add_argument("--ram-gb", type=float, default=512)
    cg.add_argument("--interconnect",
                    choices=[ic.value for ic in Interconnect],
                    default="infiniband400")
    cg.add_argument("--max-nodes", type=int, default=32)

    # Output
    og = p.add_argument_group("Output")
    og.add_argument("--output-script", type=Path, default=None,
                    help="Write SLURM script to this path")
    og.add_argument("--json", action="store_true",
                    help="Also print memory breakdown as JSON")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    gpu_preset = GPU_PRESETS[args.gpu_type]

    model = ModelSpec(
        params_millions=args.params_m,
        architecture=ModelArch(args.arch),
        precision=Precision(args.precision),
        optimizer=Optimizer(args.optimizer),
        strategy=ParallelStrategy(args.strategy),
        batch_size=args.batch,
        seq_len=args.seq_len,
        grad_checkpointing=args.grad_checkpointing,
    )

    cluster = ClusterSpec(
        gpu_vram_gb=gpu_preset["vram"],
        gpu_name=gpu_preset["name"],
        gpus_per_node=args.gpus_per_node,
        cpu_cores_per_node=args.cpu_per_node,
        ram_per_node_gb=args.ram_gb,
        interconnect=Interconnect(args.interconnect),
        max_nodes=args.max_nodes,
        has_nvlink=gpu_preset["has_nvlink"],
    )

    print("\n" + "═" * 60)
    print("  HPC Auto Resource Planner")
    print("═" * 60)
    print(f"  Model  : {args.params_m:.0f}M params | {args.precision} | {args.strategy}")
    print(f"  Cluster: {gpu_preset['name']} × {args.gpus_per_node}/node | "
          f"{args.interconnect}")
    print("═" * 60 + "\n")

    params, script = derive_slurm_params(model, cluster)

    # Memory breakdown
    mem_train = estimate_training_memory(model, world_size=params.world_size,
                                         local_batch_override=params.local_batch_size)
    mem_infer = estimate_inference_memory(model, world_size=params.world_size)

    print("Memory breakdown (per GPU — training):")
    print(mem_train)
    print(f"\nInference VRAM estimate: {mem_infer.total_gb:.2f} GB")

    print("\nRecommended SLURM parameters:")
    for d in params.slurm_directives:
        print(f"  {d}")
    print(f"  # local batch size : {params.local_batch_size}")
    print(f"  # world size       : {params.world_size}")

    if params.warnings:
        print("\n⚠  Warnings:")
        for w in params.warnings:
            print(f"  • {w}")

    print("\n" + "─" * 60)
    print("SLURM Job Script")
    print("─" * 60)
    print(script)

    if args.output_script:
        args.output_script.parent.mkdir(parents=True, exist_ok=True)
        args.output_script.write_text(script)
        print(f"\n✓ Script written to {args.output_script}")

    if args.json:
        import json
        print("\nMemory breakdown (JSON):")
        print(json.dumps(mem_train.as_dict(), indent=2))


if __name__ == "__main__":
    main()
