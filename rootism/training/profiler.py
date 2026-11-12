"""
Training Throughput, Resource Utilisation & Bottleneck Profiler (Step 58).

Measures across training steps:
- Net processed & presented tokens per second
- GPU memory utilisation (allocated, reserved, peak GiB) & GPU compute utilisation estimate
- Host CPU % and host RAM utilisation
- Data-loader stall latency (`data_wait_ms`)
- Forward + backward + optimizer compute latency (`compute_ms`)
- Distributed communication synchronization overhead (`comm_ms`)
- Checkpoint serialization latency (`checkpoint_ms`) and disk storage throughput (`MB/s`)
- Automated primary bottleneck diagnosis before recommending cluster scale-up
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

import psutil
import torch


@dataclass(frozen=True)
class StepProfileRecord:
    step: int
    tokens_processed_in_step: int
    tokens_per_second: float
    data_wait_ms: float
    compute_ms: float
    comm_ms: float
    checkpoint_ms: float
    checkpoint_write_mb_per_s: float
    cpu_util_pct: float
    host_ram_used_gib: float
    gpu_allocated_gib: float
    gpu_reserved_gib: float
    gpu_peak_allocated_gib: float
    gpu_estimated_mfu: float


@dataclass(frozen=True)
class ThroughputSummaryReport:
    measured_steps: int
    mean_tokens_per_second: float
    p50_tokens_per_second: float
    mean_data_wait_ms: float
    mean_compute_ms: float
    mean_comm_ms: float
    mean_checkpoint_ms: float
    data_time_share_pct: float
    compute_time_share_pct: float
    comm_time_share_pct: float
    checkpoint_time_share_pct: float
    peak_gpu_memory_gib: float
    peak_host_ram_gib: float
    mean_cpu_util_pct: float
    mean_estimated_mfu: float
    primary_bottleneck: str
    engineering_recommendation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TrainingThroughputProfiler:
    """
    Low-overhead wall-clock and resource profiler for Rootism training runs.
    """

    def __init__(
        self,
        *,
        model_param_count: int,
        world_size: int = 1,
        peak_gpu_tflops_per_device: float = 0.0,
        window_size: int = 64,
    ) -> None:
        self.model_param_count = int(model_param_count)
        self.world_size = max(1, int(world_size))
        self.peak_gpu_tflops_per_device = float(peak_gpu_tflops_per_device)
        self.records: deque[StepProfileRecord] = deque(maxlen=window_size)
        self._proc = psutil.Process()
        # Prime non-blocking cpu_percent counter
        self._proc.cpu_percent(interval=None)

    @staticmethod
    def now() -> float:
        return time.perf_counter()

    def record_step(
        self,
        *,
        step: int,
        tokens_processed: int,
        data_wait_sec: float,
        compute_sec: float,
        comm_sec: float = 0.0,
        checkpoint_sec: float = 0.0,
        checkpoint_bytes_written: int = 0,
    ) -> StepProfileRecord:
        total_step_sec = max(1e-6, data_wait_sec + compute_sec + comm_sec)
        tok_per_s = float(tokens_processed) / total_step_sec

        gib = float(1024**3)
        vm = psutil.virtual_memory()
        ram_gib = vm.used / gib
        cpu_pct = psutil.cpu_percent(interval=None)

        if torch.cuda.is_available():
            gpu_alloc = torch.cuda.memory_allocated() / gib
            gpu_res = torch.cuda.memory_reserved() / gib
            gpu_peak = torch.cuda.max_memory_allocated() / gib
        else:
            gpu_alloc, gpu_res, gpu_peak = 0.0, 0.0, 0.0

        if self.peak_gpu_tflops_per_device > 0.0 and torch.cuda.is_available():
            achieved_flops_per_s = 6.0 * float(self.model_param_count) * tok_per_s
            cluster_peak = float(self.world_size) * self.peak_gpu_tflops_per_device * 1e12
            mfu = min(1.0, achieved_flops_per_s / max(1.0, cluster_peak))
        else:
            mfu = 0.0

        ckpt_mb_s = (
            (float(checkpoint_bytes_written) / (1024.0 * 1024.0)) / max(1e-6, checkpoint_sec)
            if checkpoint_sec > 0.0 and checkpoint_bytes_written > 0
            else 0.0
        )

        rec = StepProfileRecord(
            step=step,
            tokens_processed_in_step=int(tokens_processed),
            tokens_per_second=round(tok_per_s, 2),
            data_wait_ms=round(data_wait_sec * 1000.0, 3),
            compute_ms=round(compute_sec * 1000.0, 3),
            comm_ms=round(comm_sec * 1000.0, 3),
            checkpoint_ms=round(checkpoint_sec * 1000.0, 3),
            checkpoint_write_mb_per_s=round(ckpt_mb_s, 2),
            cpu_util_pct=round(cpu_pct, 2),
            host_ram_used_gib=round(ram_gib, 3),
            gpu_allocated_gib=round(gpu_alloc, 3),
            gpu_reserved_gib=round(gpu_res, 3),
            gpu_peak_allocated_gib=round(gpu_peak, 3),
            gpu_estimated_mfu=round(mfu, 4),
        )
        self.records.append(rec)
        return rec

    def summarize(self) -> ThroughputSummaryReport:
        if not self.records:
            return ThroughputSummaryReport(
                measured_steps=0,
                mean_tokens_per_second=0.0,
                p50_tokens_per_second=0.0,
                mean_data_wait_ms=0.0,
                mean_compute_ms=0.0,
                mean_comm_ms=0.0,
                mean_checkpoint_ms=0.0,
                data_time_share_pct=0.0,
                compute_time_share_pct=0.0,
                comm_time_share_pct=0.0,
                checkpoint_time_share_pct=0.0,
                peak_gpu_memory_gib=0.0,
                peak_host_ram_gib=0.0,
                mean_cpu_util_pct=0.0,
                mean_estimated_mfu=0.0,
                primary_bottleneck="NO_STEPS_RECORDED",
                engineering_recommendation="Run at least 5 warmup + measurement steps.",
            )

        n = float(len(self.records))
        tps_sorted = sorted(r.tokens_per_second for r in self.records)
        p50_tps = tps_sorted[len(tps_sorted) // 2]
        mean_tps = sum(r.tokens_per_second for r in self.records) / n
        m_data = sum(r.data_wait_ms for r in self.records) / n
        m_comp = sum(r.compute_ms for r in self.records) / n
        m_comm = sum(r.comm_ms for r in self.records) / n
        m_ckpt = sum(r.checkpoint_ms for r in self.records) / n
        tot_ms = max(1e-6, m_data + m_comp + m_comm + m_ckpt)

        s_data = 100.0 * m_data / tot_ms
        s_comp = 100.0 * m_comp / tot_ms
        s_comm = 100.0 * m_comm / tot_ms
        s_ckpt = 100.0 * m_ckpt / tot_ms

        if s_data >= 20.0:
            bottleneck = "DATA_LOADER_OR_STORAGE_STALL"
            rec = "Increase prefetch queues / NVMe read concurrency; ensure pre-tokenized uint16 .rshard mmap resides on local NVMe."
        elif s_comm >= 25.0:
            bottleneck = "DISTRIBUTED_COMMUNICATION_OVERHEAD"
            rec = "Increase gradient_accumulation_steps or switch from full ZeRO-3 to intra-node FSDP2 Hybrid Sharding."
        elif s_ckpt >= 15.0:
            bottleneck = "CHECKPOINT_IO_OVERHEAD"
            rec = "Increase checkpoint_interval_steps or stage checkpoints to local NVMe before async object-store sync."
        else:
            bottleneck = "COMPUTE_BOUND_HEALTHY"
            rec = "Compute pipeline dominates wall-clock time; safe to scale GPU count or micro_batch_size if VRAM headroom allows."

        return ThroughputSummaryReport(
            measured_steps=len(self.records),
            mean_tokens_per_second=round(mean_tps, 2),
            p50_tokens_per_second=round(p50_tps, 2),
            mean_data_wait_ms=round(m_data, 3),
            mean_compute_ms=round(m_comp, 3),
            mean_comm_ms=round(m_comm, 3),
            mean_checkpoint_ms=round(m_ckpt, 3),
            data_time_share_pct=round(s_data, 2),
            compute_time_share_pct=round(s_comp, 2),
            comm_time_share_pct=round(s_comm, 2),
            checkpoint_time_share_pct=round(s_ckpt, 2),
            peak_gpu_memory_gib=round(max(r.gpu_peak_allocated_gib for r in self.records), 3),
            peak_host_ram_gib=round(max(r.host_ram_used_gib for r in self.records), 3),
            mean_cpu_util_pct=round(sum(r.cpu_util_pct for r in self.records) / n, 2),
            mean_estimated_mfu=round(sum(r.gpu_estimated_mfu for r in self.records) / n, 4),
            primary_bottleneck=bottleneck,
            engineering_recommendation=rec,
        )
