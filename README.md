# 🖥️ HPC Auto Resource Planner

> **Stop guessing your SLURM parameters. Profile your model once, get optimal resource allocation automatically.**

Built as a learning project during my HPC internship. This tool takes your model's architecture and your cluster's hardware spec, then solves the resource allocation problem — telling you exactly how many nodes, GPUs, CPUs, and memory to request so your training job saturates the hardware instead of wasting it.

---

## The Problem This Solves

Every HPC user has done this:

```bash
# Day 1 — too little memory
srun: error: GPU memory allocation failed (OOM)

# Day 2 — too many nodes, wasted budget
Job finished. GPU utilization: 34%

# Day 3 — wrong ntasks-per-node
All processes spawned on rank 0. Deadlock.
```

Manual trial-and-error on an HPC cluster is expensive. Node-hours cost real money (or real queue time). This tool calculates the correct parameters analytically before you submit a single job.

---

## How It Works

The tool solves a 3-stage optimization:

### Stage 1 — Memory Budget

Total GPU memory needed for training isn't just the model size. It's four things stacked:

```
Total VRAM = Weights + Gradients + Optimizer States + Activations + Overhead
```

| Component | FP16 7B model | Notes |
|---|---|---|
| Weights | ~14 GB | `params × bytes_per_param` |
| Gradients | ~28 GB | Always FP32 in mixed precision |
| Optimizer states (AdamW) | ~84 GB | 3× weight size — momentum + variance |
| Activations | ~8–40 GB | Depends on batch × seq_len |
| Overhead | ~2.5 GB | CUDA context, NCCL buffers |

Minimum GPU count = `ceil(total_vram / gpu_vram)`

Gradient checkpointing cuts activation memory to ~33% — trading a second backward pass for memory.

### Stage 2 — Communication Efficiency

Multi-node training adds communication overhead. The all-reduce operation during backprop scales with:
- Interconnect bandwidth (NVLink 900 GB/s vs 100GbE 12.5 GB/s)
- Interconnect latency (1 µs NVLink vs 15 µs Ethernet)
- Number of parameters being synchronized

The tool estimates communication overhead and warns you when your interconnect will bottleneck training.

### Stage 3 — SLURM Parameter Derivation

From the memory and efficiency analysis, the tool emits a ready-to-use SLURM script with:

```bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8       # one process per GPU
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8         # DataLoader workers
#SBATCH --mem=435G                # 85% of node RAM
```

Plus critical environment variables that most tutorials skip:

```bash
export NCCL_IB_DISABLE=0          # Enable InfiniBand
export OMP_NUM_THREADS=8          # Prevent CPU thrashing
export NCCL_DEBUG=INFO            # Diagnose comm failures
```

---

## Interactive Tool

The static analysis tool is available as a browser-based calculator — input your model and cluster specs, get instant SLURM parameters and efficiency estimates.

**Inputs:**
- Framework (PyTorch DDP/FSDP, TensorFlow/Horovod, JAX, DeepSpeed)
- Model type + parameter count
- Precision (FP32/FP16/BF16/INT8/FP8)
- Batch size, sequence length
- Training strategy (DDP, FSDP, model parallel, hybrid)
- GPU type, GPUs per node, CPU count, RAM
- Interconnect type

**Outputs:**
- Recommended `--nodes`, `--ntasks`, `--gres`, `--cpus-per-task`, `--mem`
- Memory breakdown: per-GPU usage across all 5 components
- Efficiency forecast: GPU utilization, compute throughput, communication overhead, load balance
- Full SLURM job script (copy-paste ready)
- Warnings for common pitfalls (VRAM near limit, small local batch, poor interconnect, etc.)

---

## What I Learned Building This

This was my first real HPC project. Key takeaways that surprised me:

**1. Optimizer state memory is the silent killer.**
AdamW stores first and second moment estimates for every parameter in FP32, even in mixed-precision training. For a 7B model, that's 84 GB just for the optimizer — more than the weights themselves on some GPUs.

**2. `--cpus-per-task` is not optional.**
Setting it wrong means all DataLoader workers compete on the same cores as the GPU process. Throughput tanks. Rule of thumb: `total_cpu_cores / gpus_per_node`, capped at 8–16.

**3. NCCL environment variables matter more than documentation suggests.**
`NCCL_IB_DISABLE=0` on a cluster with InfiniBand gave a 22% throughput improvement in one test. Most job scripts I found online didn't set this.

**4. The roofline model is the right mental framework.**
Every operation is either memory-bandwidth bound or compute bound. Once you know which one, you know whether buying more GPUs or faster interconnect actually helps.

---

## Roadmap — Production Version

The static estimates here are a starting point. A production version would:

- [ ] **Live profiling via `nvidia-smi dmon`** — parse actual GPU utilization, memory, temperature, and power draw. Feed it back as calibration data.
- [ ] **W&B integration** — read historical GPU utilization from past runs of the same model. Empirical data beats formulas every time.
- [ ] **Roofline model** — automatically classify each layer as memory-bound or compute-bound using arithmetic intensity. Predict whether FSDP or DDP is optimal for your specific model.
- [ ] **Auto-tune local batch size** — binary search for the largest batch that fits without OOM, using a dry-run forward pass.
- [ ] **Queue-aware scheduling** — query the cluster's current queue depth via `squeue` and recommend job sizes that will actually get scheduled.
- [ ] **Multi-framework cost model** — separate FLOP profiles for attention layers, feed-forward, embeddings, loss functions.

---

## Project Structure

```
hpc-auto-resource-planner/
├── planner/
│   ├── memory_model.py       # VRAM estimation per component
│   ├── comm_model.py         # Interconnect bandwidth/latency model
│   ├── slurm_emitter.py      # SLURM script generation
│   └── roofline.py           # [WIP] Roofline analysis
├── profiler/
│   ├── nvsmi_parser.py       # nvidia-smi dmon output parser
│   └── wandb_reader.py       # [WIP] W&B run history reader
├── ui/
│   └── index.html            # Browser-based interactive tool
├── tests/
│   ├── test_memory_model.py
│   └── test_slurm_emitter.py
├── examples/
│   ├── llama2_7b_a100.sh     # Example: LLaMA-2 7B on A100 cluster
│   ├── resnet50_v100.sh      # Example: ResNet-50 on V100 cluster
│   └── gpt2_fsdp.sh          # Example: GPT-2 with FSDP
└── README.md
```

---

## Quick Reference: Common Pitfalls

| Symptom | Likely Cause | Fix |
|---|---|---|
| OOM on first iteration | Activations underestimated | Enable gradient checkpointing |
| GPU util < 40% | DataLoader CPU bottleneck | Increase `--cpus-per-task` |
| Deadlock at init | Wrong `--ntasks-per-node` | Must equal GPUs per node |
| Slow multi-node | NCCL using Ethernet fallback | Set `NCCL_IB_DISABLE=0` |
| Rank 0 OOM only | Model parallel misconfigured | Check pipeline stage sizes |
| High util but slow | Memory bandwidth bound | Switch to FP16/BF16 |

---

## Resources That Actually Helped

- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/) — especially the environment variable reference
- [Hugging Face Model Memory Estimator](https://huggingface.co/docs/accelerate/usage_guides/model_size_estimator) — for cross-checking estimates
- [DeepSpeed ZeRO paper](https://arxiv.org/abs/1910.02054) — the math behind optimizer state sharding
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473) — best reference for 3D parallelism

---

## About

Built during an HPC internship as a learning project. The goal was to understand *why* HPC jobs underperform, not just how to fix them.

If this saved you node-hours, or if you find bugs — open an issue or reach out.

---

*"The best job script is the one you don't have to resubmit."*
