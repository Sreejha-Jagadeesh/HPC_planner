# I Built a Tool That Tells You Exactly How Many GPUs Your Training Job Needs — Here's the Math Behind It

*What I learned as an HPC intern about the gap between "it runs" and "it runs efficiently"*

---

The first training job I submitted to our HPC cluster ran for 11 minutes before crashing with an out-of-memory error. The second one ran fine — but when I checked the monitoring dashboard the next morning, GPU utilization was sitting at 31%.

I had gone from using too little memory to wasting two-thirds of $40/hour of compute.

My supervisor, a senior AI HPC engineer, gave me a task: *"Build something that takes a model spec and outputs the correct SLURM parameters. Automate what I've been doing in my head for five years."*

This is that story.

---

## The Problem Nobody Talks About in Tutorials

Most ML tutorials stop at "it converges." HPC engineering starts at "it converges *efficiently* at scale."

When you submit a job to a cluster, you're answering a set of questions:

- How many nodes do I need? (`--nodes`)
- How many GPU processes per node? (`--ntasks-per-node`)
- How many CPU cores per process? (`--cpus-per-task`)
- How much RAM? (`--mem`)

Get these wrong and you either crash immediately or silently waste a large chunk of your resource allocation. Most people iterate by trial and error. I wanted to solve it analytically.

---

## The Math: What Actually Lives in GPU Memory?

The rookie answer is: *the model parameters*. The actual answer is four things, and they interact in ways that aren't obvious.

### 1. Model Weights

```
weight_memory = num_params × bytes_per_param
```

For a 7B parameter model in FP16: `7×10⁹ × 2 bytes = 14 GB`. Simple enough.

### 2. Gradients

Here's where it gets interesting. Even in mixed-precision training (FP16 forward pass), gradients are typically kept in FP32 to avoid underflow. So:

```
gradient_memory = num_params × 4 bytes  (always FP32)
```

For the same 7B model: **28 GB** — twice the weight memory.

### 3. Optimizer States

This one surprises almost everyone. AdamW, the default optimizer for most transformer training, stores:
- A first-moment estimate (momentum) — FP32
- A second-moment estimate (variance) — FP32

That's two FP32 tensors the same size as your model. On top of the master weights copy (also FP32), AdamW costs roughly **12 bytes per parameter**:

```
adam_memory = num_params × 12 bytes
```

For a 7B model: **84 GB**. On an A100 80GB. You're already over budget before adding anything else.

This is why FSDP (Fully Sharded Data Parallel) exists. It shards optimizer states across GPUs, so each GPU only holds `1/world_size` of the optimizer memory.

### 4. Activations

Activations — the intermediate values stored during forward pass for use in backprop — scale with batch size and sequence length. For transformers:

```
activation_memory ≈ batch_size × seq_len × hidden_dim × num_layers × bytes_per_activation
```

This is the most variable term. A batch of 256 with sequence length 2048 can easily add 20–40 GB of activations.

**Gradient checkpointing** trades compute for memory: instead of storing all activations, you recompute them during backprop. This reduces activation memory to roughly 33% at the cost of an extra forward pass.

### Putting It Together

```
Total VRAM ≈ Weights + Gradients + Optimizer States + Activations + Overhead (~2.5 GB)
```

Minimum GPU count = `ceil(total_vram / gpu_vram_available)`

Then you round up to the nearest multiple of GPUs-per-node, because you want full nodes, not partial ones.

---

## The Part That Actually Kills Your Throughput: Communication

Once you've solved the memory problem and you're running on multiple GPUs or nodes, the bottleneck shifts. Every backward pass requires an `all-reduce` — every GPU needs to synchronize gradients across all other GPUs.

The time this takes depends entirely on your interconnect:

| Interconnect | Bandwidth | Typical Training Overhead |
|---|---|---|
| NVLink 4 (within node) | 900 GB/s | < 5% |
| InfiniBand HDR 400Gb | 50 GB/s | 8–15% |
| InfiniBand HDR 200Gb | 25 GB/s | 15–25% |
| 100GbE Ethernet | 12.5 GB/s | 25–40% |

If you're doing multi-node training on Ethernet, you're potentially giving away a third of your throughput to network overhead. This is why clusters that are serious about distributed training invest in InfiniBand fabric.

One thing I learned the hard way: on a cluster with InfiniBand, you need to explicitly tell NCCL to use it:

```bash
export NCCL_IB_DISABLE=0
```

Without this, NCCL may fall back to Ethernet even when InfiniBand is available. That one line gave us a 22% throughput improvement.

---

## The SLURM Parameters, Derived

From the memory and communication analysis, you can derive each SLURM parameter with actual reasoning:

**`--ntasks-per-node`** = number of GPUs per node. One process per GPU. Not one process per CPU, not one process per node. One per GPU.

**`--gres=gpu:N`** = same as ntasks-per-node. You need both.

**`--cpus-per-task`** = `total_cpu_cores / gpus_per_node`. These are for your DataLoader workers. Too few and your GPU starves waiting for data. Too many and you create CPU contention. Cap at 8–16 in practice.

**`--mem`** = 85% of node RAM. Leave headroom for the OS and NCCL buffers. Setting it to 100% will cause your job to be killed by the OOM daemon at inconvenient moments.

And the environment variables nobody puts in their scripts:

```bash
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK   # Prevent OpenMP from spawning 128 threads
export NCCL_DEBUG=INFO                          # Diagnose communication failures
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7   # Explicit GPU assignment
```

---

## What I Built

I built a browser-based tool that takes:
- Model parameters (size, framework, precision, strategy)
- Cluster spec (GPU type, GPUs per node, CPU count, RAM, interconnect)

And outputs:
- Recommended SLURM parameters
- Per-component memory breakdown (weights, gradients, optimizer, activations, overhead)
- Efficiency forecast (GPU util, compute throughput, communication overhead)
- A complete, copy-paste SLURM job script
- Warnings for common pitfalls

The math is entirely static — no profiling, no empirical data. Which brings me to what I want to build next.

---

## Where Static Analysis Falls Short

The tool works well as a first estimate. But there are failure modes:

**It can't see your DataLoader.** If your preprocessing is slow or your storage is NFS-mounted with high latency, GPU utilization will tank regardless of how perfect your SLURM parameters are. Only `nvidia-smi dmon` watching a live run can tell you that.

**Activation estimates are approximate.** The formula I use is a rough model for transformers. Custom architectures, mixture-of-experts layers, and attention variants all have different profiles. Historical runs in W&B would give you real numbers.

**It doesn't know about your cluster's queue.** Requesting 32 nodes when the cluster has 100 jobs waiting for more than 8 nodes each is a bad strategy even if the math says 32 is optimal. A production version would query `squeue` and recommend sizes that will actually schedule.

---

## The Roadmap

Here's what the production version looks like:

1. **`nvidia-smi dmon` integration** — parse live GPU metrics (utilization, memory, temperature, power) during a short profiling run. Feed it back as calibration.

2. **W&B run history reader** — if you've trained this model before, use the actual observed GPU utilization from previous runs instead of formulas.

3. **Roofline model** — classify each layer as memory-bandwidth bound or compute bound using arithmetic intensity (`FLOPs / bytes_accessed`). This tells you whether buying more GPUs or faster interconnect actually helps for your specific model.

4. **Auto-tune batch size** — binary search for the largest batch size that fits without OOM, using a single dry-run forward pass before the real job.

---

## What This Internship Taught Me

HPC engineering is fundamentally about closing the gap between theoretical peak performance and actual utilization. A GPU with 312 TFLOPS of theoretical throughput running at 30% efficiency delivers 93 TFLOPS. The person who understands *why* it's at 30% — and knows which lever to pull — is worth a lot more than someone who just runs jobs until they stop crashing.

The math in this tool isn't new. Every piece of it is documented somewhere in NVIDIA's docs, the DeepSpeed paper, or Megatron-LM's implementation. What was missing was a single tool that assembled it into actionable SLURM parameters.

That gap is what internship projects are for.

---

*The tool and source code are available on GitHub. If you find an error in the memory model or have empirical data that contradicts my estimates — I'd genuinely like to know.*

---

**Tags:** `hpc` `deep-learning` `gpu-computing` `slurm` `distributed-training` `pytorch` `mlops` `internship`
