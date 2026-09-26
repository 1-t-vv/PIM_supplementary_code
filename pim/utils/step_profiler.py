from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Tuple

import torch


class StepProfiler:
    """Low-overhead sampled training-step profiler.

    DataLoader wait time is measured with a CPU clock. GPU phases use CUDA
    events and are resolved with a single synchronization after the sampling
    window, avoiding a device synchronization at every phase boundary.
    """

    GPU_STAGES = ("h2d", "forward", "loss", "backward", "optimizer")

    def __init__(
        self,
        *,
        enabled: bool,
        device: torch.device,
        rank: int,
        profile_epoch: int = 1,
        warmup_steps: int = 5,
        num_steps: int = 20,
    ) -> None:
        if profile_epoch <= 0:
            raise ValueError(f"profile_epoch must be >= 1, got {profile_epoch}")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        if num_steps <= 0:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}")

        self.device = torch.device(device)
        self.profile_epoch = int(profile_epoch) - 1  # config is one-based
        self.warmup_steps = int(warmup_steps)
        self.num_steps = int(num_steps)
        self.enabled = bool(enabled and rank == 0 and self.device.type == "cuda")

        self._samples: List[Dict[str, Any]] = []
        self._current: Optional[Dict[str, Any]] = None
        self._done = False

    def should_profile(self, epoch: int, batch_idx: int) -> bool:
        return bool(
            self.enabled
            and not self._done
            and epoch == self.profile_epoch
            and self.warmup_steps <= batch_idx < self.warmup_steps + self.num_steps
        )

    @staticmethod
    def _event() -> torch.cuda.Event:
        return torch.cuda.Event(enable_timing=True)

    def begin_step(
        self,
        *,
        epoch: int,
        batch_idx: int,
        data_wait_ms: float,
        is_update_step: bool = True,
    ) -> bool:
        if not self.should_profile(epoch, batch_idx):
            self._current = None
            return False
        if self._current is not None:
            raise RuntimeError("StepProfiler.begin_step called before finish_step")

        step_start = self._event()
        step_start.record()
        self._current = {
            "epoch": int(epoch),
            "batch_idx": int(batch_idx),
            "data_wait_ms": max(0.0, float(data_wait_ms)),
            "is_update_step": bool(is_update_step),
            "step_start": step_start,
            "stages": {},
            "open_stage": None,
        }
        return True

    def start(self, stage: str) -> None:
        if self._current is None:
            return
        if stage not in self.GPU_STAGES:
            raise ValueError(f"Unknown profiler stage: {stage}")
        if self._current["open_stage"] is not None:
            raise RuntimeError(
                f"Cannot start {stage}; {self._current['open_stage']} is still open"
            )
        start = self._event()
        start.record()
        self._current["open_stage"] = (stage, start)

    def end(self, stage: str) -> None:
        if self._current is None:
            return
        open_stage = self._current["open_stage"]
        if open_stage is None or open_stage[0] != stage:
            raise RuntimeError(f"Profiler stage end mismatch: expected {open_stage}, got {stage}")
        end = self._event()
        end.record()
        self._current["stages"][stage] = (open_stage[1], end)
        self._current["open_stage"] = None

    def finish_step(self) -> Optional[Dict[str, Any]]:
        if self._current is None:
            return None
        if self._current["open_stage"] is not None:
            raise RuntimeError(f"Profiler stage still open: {self._current['open_stage'][0]}")

        step_end = self._event()
        step_end.record()
        self._current["step_end"] = step_end
        self._samples.append(self._current)
        self._current = None

        if len(self._samples) < self.num_steps:
            return None
        return self._finalize()

    def finish_epoch(self, epoch: int) -> Optional[Dict[str, Any]]:
        """Resolve a partial sampling window at the end of its target epoch."""
        if (
            not self.enabled
            or self._done
            or int(epoch) != self.profile_epoch
            or not self._samples
        ):
            return None
        if self._current is not None:
            raise RuntimeError("finish_epoch called while a profiled step is still open")
        return self._finalize()

    def _finalize(self) -> Dict[str, Any]:
        torch.cuda.synchronize(self.device)
        records: List[Dict[str, Any]] = []
        for sample in self._samples:
            record: Dict[str, Any] = {
                "epoch": sample["epoch"],
                "batch_idx": sample["batch_idx"],
                "data_wait": sample["data_wait_ms"],
                "is_update_step": sample["is_update_step"],
                "gpu_step": sample["step_start"].elapsed_time(sample["step_end"]),
            }
            for stage in self.GPU_STAGES:
                events: Optional[Tuple[torch.cuda.Event, torch.cuda.Event]] = sample[
                    "stages"
                ].get(stage)
                record[stage] = 0.0 if events is None else events[0].elapsed_time(events[1])
            records.append(record)

        self._done = True
        return self.summarize_records(records)

    @staticmethod
    def _percentile(values: List[float], percentile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(float(v) for v in values)
        if len(ordered) == 1:
            return ordered[0]
        position = (len(ordered) - 1) * float(percentile) / 100.0
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    @classmethod
    def _stats(cls, values: List[float]) -> Dict[str, float]:
        if not values:
            return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
        return {
            "mean_ms": sum(values) / len(values),
            "p50_ms": cls._percentile(values, 50.0),
            "p95_ms": cls._percentile(values, 95.0),
        }

    @classmethod
    def summarize_records(cls, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not records:
            raise ValueError("Cannot summarize an empty profiler record list")

        stage_names = ("data_wait",) + cls.GPU_STAGES
        stage_stats = {
            stage: cls._stats([float(record.get(stage, 0.0)) for record in records])
            for stage in stage_names
        }
        accounted_mean = sum(stage_stats[stage]["mean_ms"] for stage in stage_names)
        for stage in stage_names:
            share = (
                100.0 * stage_stats[stage]["mean_ms"] / accounted_mean
                if accounted_mean > 0.0
                else 0.0
            )
            stage_stats[stage]["share_pct"] = share

        update_optimizer = [
            float(record.get("optimizer", 0.0))
            for record in records
            if bool(record.get("is_update_step", False))
        ]
        optimizer_update_stats = cls._stats(update_optimizer)
        optimizer_update_stats["samples"] = len(update_optimizer)

        return {
            "epoch": int(records[0]["epoch"]) + 1,
            "steps": len(records),
            "first_batch": int(records[0]["batch_idx"]) + 1,
            "last_batch": int(records[-1]["batch_idx"]) + 1,
            "accounted_step": cls._stats(
                [sum(float(record.get(stage, 0.0)) for stage in stage_names) for record in records]
            ),
            "gpu_step": cls._stats([float(record["gpu_step"]) for record in records]),
            "stages": stage_stats,
            "optimizer_update": optimizer_update_stats,
        }

    @staticmethod
    def format_summary(summary: Dict[str, Any]) -> str:
        lines = [
            "[StepProfiler] "
            f"epoch={summary['epoch']} batches={summary['first_batch']}-{summary['last_batch']} "
            f"samples={summary['steps']}"
        ]
        for stage in ("data_wait",) + StepProfiler.GPU_STAGES:
            stats = summary["stages"][stage]
            lines.append(
                "[StepProfiler] "
                f"{stage:>10s}: mean={stats['mean_ms']:.3f} ms "
                f"p50={stats['p50_ms']:.3f} ms p95={stats['p95_ms']:.3f} ms "
                f"share={stats['share_pct']:.1f}%"
            )
        gpu = summary["gpu_step"]
        accounted = summary["accounted_step"]
        update = summary["optimizer_update"]
        lines.append(
            "[StepProfiler] "
            f"accounted_step: mean={accounted['mean_ms']:.3f} ms; "
            f"gpu_span: mean={gpu['mean_ms']:.3f} ms p95={gpu['p95_ms']:.3f} ms"
        )
        lines.append(
            "[StepProfiler] "
            f"optimizer_update: mean={update['mean_ms']:.3f} ms "
            f"p95={update['p95_ms']:.3f} ms samples={update['samples']}"
        )
        lines.append(
            "[StepProfilerJSON] " + json.dumps(summary, ensure_ascii=False, separators=(",", ":"))
        )
        return "\n".join(lines)
