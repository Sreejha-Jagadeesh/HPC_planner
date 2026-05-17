"""
nvsmi_parser.py
---------------
Parses nvidia-smi dmon output for live GPU profiling.

nvidia-smi dmon streams one line per GPU per sample interval:
  # gpu   pwr  gtemp  mtemp    sm   mem   enc   dec  mclk  pclk
  #  Idx    W      C      C     %     %     %     %   MHz   MHz
      0   285     62      -    98    87     0     0  9751  1530

Usage
-----
  # Capture 60 seconds of samples (1-second intervals):
  nvidia-smi dmon -s u -d 1 -c 60 > profile.txt

  parser = NvSmiParser("profile.txt")
  summary = parser.summarize()
  print(summary)

  # Or stream live:
  import subprocess
  proc = subprocess.Popen(
      ["nvidia-smi", "dmon", "-s", "u", "-d", "1"],
      stdout=subprocess.PIPE, text=True,
  )
  for sample in NvSmiParser.stream(proc.stdout):
      print(sample)
"""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, IO, Optional


@dataclass
class GpuSample:
    """One time-step of metrics for one GPU."""
    gpu_idx: int
    timestamp: float          # Unix epoch (added by parser)
    sm_util_pct: float        # Streaming multiprocessor utilization
    mem_util_pct: float       # Memory bandwidth utilization
    power_w: float            # Power draw in Watts
    gpu_temp_c: float         # GPU die temperature
    mem_clock_mhz: float      # Memory clock
    sm_clock_mhz: float       # SM clock


@dataclass
class GpuProfileSummary:
    """Aggregate statistics over a profiling session."""
    gpu_idx: int
    sample_count: int
    duration_s: float

    sm_util_mean: float
    sm_util_p95: float
    sm_util_min: float

    mem_util_mean: float
    mem_util_p95: float

    power_mean_w: float
    power_peak_w: float

    temp_mean_c: float
    temp_peak_c: float

    # Derived
    is_memory_bound: bool     # mem_util >> sm_util suggests memory bottleneck
    efficiency_rating: float  # 0–1 based on sm_util

    def __str__(self) -> str:
        bound = "MEMORY-BOUND" if self.is_memory_bound else "COMPUTE-BOUND"
        return (
            f"GPU {self.gpu_idx} | {self.sample_count} samples over {self.duration_s:.0f}s\n"
            f"  SM utilization : mean {self.sm_util_mean:.1f}%  p95 {self.sm_util_p95:.1f}%  "
            f"min {self.sm_util_min:.1f}%\n"
            f"  Mem bandwidth  : mean {self.mem_util_mean:.1f}%  p95 {self.mem_util_p95:.1f}%\n"
            f"  Power          : mean {self.power_mean_w:.0f}W  peak {self.power_peak_w:.0f}W\n"
            f"  Temperature    : mean {self.temp_mean_c:.1f}°C  peak {self.temp_peak_c:.1f}°C\n"
            f"  Classification : {bound}  |  efficiency {self.efficiency_rating*100:.0f}%"
        )


class NvSmiParser:
    """
    Parses nvidia-smi dmon output from a file or a live stream.

    nvidia-smi dmon header lines start with '#' and are skipped.
    Data lines: gpu_idx  power  gtemp  mtemp  sm_util  mem_util  enc  dec  mclk  pclk
    """

    # Regex: leading whitespace, then whitespace-separated integers/dashes
    _DATA_RE = re.compile(
        r"^\s*(\d+)\s+"           # gpu index
        r"(\d+|-)\s+"             # power
        r"(\d+|-)\s+"             # gpu temp
        r"(\d+|-)\s+"             # mem temp (may be -)
        r"(\d+|-)\s+"             # sm util
        r"(\d+|-)\s+"             # mem util
        r"(\d+|-)\s+"             # enc
        r"(\d+|-)\s+"             # dec
        r"(\d+|-)\s+"             # mclk
        r"(\d+|-)"                # pclk
    )

    def __init__(self, path: Optional[str | Path] = None):
        self._path = Path(path) if path else None
        self._samples: list[GpuSample] = []
        if path:
            self._load(self._path)

    def _load(self, path: Path) -> None:
        t0 = time.time()
        with open(path) as f:
            for line in f:
                sample = self._parse_line(line, t0)
                if sample:
                    self._samples.append(sample)

    def _parse_line(self, line: str, base_ts: float) -> Optional[GpuSample]:
        line = line.strip()
        if not line or line.startswith("#"):
            return None
        m = self._DATA_RE.match(line)
        if not m:
            return None

        def _f(s: str, default: float = 0.0) -> float:
            return float(s) if s != "-" else default

        return GpuSample(
            gpu_idx=int(m.group(1)),
            timestamp=base_ts + len(self._samples) * 1.0,
            sm_util_pct=_f(m.group(5)),
            mem_util_pct=_f(m.group(6)),
            power_w=_f(m.group(2)),
            gpu_temp_c=_f(m.group(3)),
            mem_clock_mhz=_f(m.group(9)),
            sm_clock_mhz=_f(m.group(10)),
        )

    @staticmethod
    def stream(file_obj: IO[str]) -> Generator[GpuSample, None, None]:
        """Yield GpuSample objects from a live nvidia-smi dmon stream."""
        t0 = time.time()
        parser = NvSmiParser()
        for line in file_obj:
            sample = parser._parse_line(line, t0)
            if sample:
                yield sample

    def summarize(self) -> dict[int, GpuProfileSummary]:
        """Compute per-GPU summary statistics from loaded samples."""
        if not self._samples:
            raise ValueError("No samples loaded. Call load() or provide a file path.")

        import statistics

        gpu_ids = sorted(set(s.gpu_idx for s in self._samples))
        summaries: dict[int, GpuProfileSummary] = {}

        for gid in gpu_ids:
            gpu_samples = [s for s in self._samples if s.gpu_idx == gid]
            sm_vals   = [s.sm_util_pct for s in gpu_samples]
            mem_vals  = [s.mem_util_pct for s in gpu_samples]
            pow_vals  = [s.power_w for s in gpu_samples]
            tmp_vals  = [s.gpu_temp_c for s in gpu_samples]

            def _p95(vals: list[float]) -> float:
                if not vals:
                    return 0.0
                idx = int(len(vals) * 0.95)
                return sorted(vals)[min(idx, len(vals)-1)]

            duration = (gpu_samples[-1].timestamp - gpu_samples[0].timestamp
                        if len(gpu_samples) > 1 else 1.0)

            sm_mean = statistics.mean(sm_vals)
            mem_mean = statistics.mean(mem_vals)

            # Memory-bound heuristic: mem_util > 1.5× sm_util
            is_mem_bound = (mem_mean > sm_mean * 1.5 and mem_mean > 50)
            efficiency = sm_mean / 100.0

            summaries[gid] = GpuProfileSummary(
                gpu_idx=gid,
                sample_count=len(gpu_samples),
                duration_s=duration,
                sm_util_mean=sm_mean,
                sm_util_p95=_p95(sm_vals),
                sm_util_min=min(sm_vals),
                mem_util_mean=mem_mean,
                mem_util_p95=_p95(mem_vals),
                power_mean_w=statistics.mean(pow_vals),
                power_peak_w=max(pow_vals),
                temp_mean_c=statistics.mean(tmp_vals),
                temp_peak_c=max(tmp_vals),
                is_memory_bound=is_mem_bound,
                efficiency_rating=efficiency,
            )

        return summaries


def run_profiler(
    duration_s: int = 30,
    interval_s: int = 1,
    gpu_indices: Optional[list[int]] = None,
) -> dict[int, GpuProfileSummary]:
    """
    Run nvidia-smi dmon for `duration_s` seconds and return summaries.

    Requires: nvidia-smi available in PATH, running on a GPU node.
    """
    n_samples = duration_s // interval_s
    cmd = ["nvidia-smi", "dmon", "-s", "u", "-d", str(interval_s), "-c", str(n_samples)]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=duration_s + 10)
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        raise RuntimeError(f"nvidia-smi dmon failed: {e}") from e

    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi exited {result.returncode}: {result.stderr}")

    import tempfile, os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(result.stdout)
        tmp_path = f.name

    try:
        parser = NvSmiParser(tmp_path)
        return parser.summarize()
    finally:
        os.unlink(tmp_path)
