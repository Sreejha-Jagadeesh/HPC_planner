"""
comm_model.py
-------------
Models inter-GPU communication overhead for distributed training.

Covers:
  - All-reduce cost (DDP / FSDP gradient sync)
  - Point-to-point cost (pipeline parallelism)
  - Interconnect characterisation (NVLink, InfiniBand, Ethernet)

Reference:
  Narayanan et al., "Efficient Large-Scale Language Model Training on GPU
  Clusters Using Megatron-LM", SC'21. https://arxiv.org/abs/2104.04473
"""

from dataclasses import dataclass
from enum import Enum


class Interconnect(str, Enum):
    NVLINK4       = "nvlink4"        # NVLink 4 (H100 SXM) — 900 GB/s
    NVLINK3       = "nvlink3"        # NVLink 3 (A100 SXM) — 600 GB/s
    INFINIBAND400 = "infiniband400"  # IB HDR 400Gb — ~50 GB/s effective
    INFINIBAND200 = "infiniband200"  # IB HDR 200Gb — ~25 GB/s
    ETHERNET100   = "ethernet100"    # 100GbE — ~12.5 GB/s


@dataclass
class InterconnectSpec:
    name: str
    bandwidth_gbps: float   # Unidirectional bandwidth in GB/s
    latency_us: float       # Per-message latency in microseconds
    within_node: bool       # True for NVLink (intra-node only)


INTERCONNECT_SPECS: dict[Interconnect, InterconnectSpec] = {
    Interconnect.NVLINK4: InterconnectSpec(
        name="NVLink 4", bandwidth_gbps=900.0, latency_us=1.0, within_node=True),
    Interconnect.NVLINK3: InterconnectSpec(
        name="NVLink 3", bandwidth_gbps=600.0, latency_us=1.0, within_node=True),
    Interconnect.INFINIBAND400: InterconnectSpec(
        name="InfiniBand HDR 400Gb", bandwidth_gbps=50.0, latency_us=5.0, within_node=False),
    Interconnect.INFINIBAND200: InterconnectSpec(
        name="InfiniBand HDR 200Gb", bandwidth_gbps=25.0, latency_us=6.0, within_node=False),
    Interconnect.ETHERNET100: InterconnectSpec(
        name="100GbE Ethernet", bandwidth_gbps=12.5, latency_us=15.0, within_node=False),
}


@dataclass
class CommEfficiency:
    """Communication efficiency metrics."""
    all_reduce_efficiency: float    # 0–1; 1 = no overhead
    latency_penalty: float          # 0–1; 1 = no latency impact
    overall_comm_efficiency: float  # Combined estimate
    bottleneck: str                 # Human-readable bottleneck description
    nccl_flags: dict[str, str]      # Recommended NCCL env vars


def estimate_comm_efficiency(
    interconnect: Interconnect,
    world_size: int,
    params_millions: float,
    nodes: int,
) -> CommEfficiency:
    """
    Estimate communication efficiency for all-reduce with given interconnect.

    The all-reduce cost for ring-all-reduce is:
        time = 2 × (N-1)/N × message_size / bandwidth + 2 × (N-1) × latency

    We normalise this against pure compute time to get an efficiency ratio.
    """
    spec = INTERCONNECT_SPECS[interconnect]

    # Gradient tensor size in GB (FP32 master gradients)
    grad_size_gb = (params_millions * 1e6 * 4.0) / 1e9

    # Ring all-reduce message per GPU
    if world_size > 1:
        msg_size_gb = 2 * (world_size - 1) / world_size * grad_size_gb
    else:
        msg_size_gb = 0.0

    # Estimated all-reduce time (seconds)
    allreduce_time_s = (msg_size_gb / max(spec.bandwidth_gbps, 0.001) +
                        2 * (world_size - 1) * spec.latency_us * 1e-6)

    # Rough compute time for a transformer forward+backward at 312 TFLOPS
    # FLOPs ≈ 6 × params per token; assume 1 step = 1 token per param
    flops = 6 * params_millions * 1e6
    compute_time_s = flops / (312e12)  # A100 FP16 baseline

    # Efficiency = compute / (compute + comm)
    if compute_time_s + allreduce_time_s > 0:
        ar_efficiency = compute_time_s / (compute_time_s + allreduce_time_s)
    else:
        ar_efficiency = 1.0

    ar_efficiency = min(max(ar_efficiency, 0.40), 0.99)

    # Latency penalty — high latency hurts more with many small messages
    if spec.latency_us <= 2:
        lat_penalty = 0.99
    elif spec.latency_us <= 6:
        lat_penalty = 0.94
    elif spec.latency_us <= 10:
        lat_penalty = 0.88
    else:
        lat_penalty = 0.78

    overall = ar_efficiency * lat_penalty

    # Bottleneck identification
    if nodes == 1:
        bottleneck = "Intra-node only — NVLink saturation unlikely below 8 GPUs"
    elif spec.bandwidth_gbps >= 400:
        bottleneck = "Bandwidth not a bottleneck; watch for NCCL kernel launch overhead"
    elif spec.bandwidth_gbps >= 25:
        bottleneck = f"Cross-node bandwidth ({spec.bandwidth_gbps:.0f} GB/s) — overlap comm/compute with async all-reduce"
    else:
        bottleneck = f"Ethernet bottleneck severe at {world_size} GPUs — gradient compression (PowerSGD) recommended"

    # Recommended NCCL environment variables
    nccl_flags = {
        "NCCL_DEBUG": "INFO",
        "NCCL_IB_DISABLE": "0" if not spec.within_node and nodes > 1 else "1",
        "NCCL_SOCKET_IFNAME": "ib0" if "InfiniBand" in spec.name else "eth0",
    }
    if spec.within_node:
        nccl_flags["NCCL_P2P_DISABLE"] = "0"
    if spec.bandwidth_gbps < 25 and nodes > 1:
        nccl_flags["NCCL_ALGO"] = "Ring"  # Ring is best for low-BW large messages

    return CommEfficiency(
        all_reduce_efficiency=round(ar_efficiency, 4),
        latency_penalty=round(lat_penalty, 4),
        overall_comm_efficiency=round(overall, 4),
        bottleneck=bottleneck,
        nccl_flags=nccl_flags,
    )
