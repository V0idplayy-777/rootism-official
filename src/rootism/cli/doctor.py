"""Environment/hardware doctor: fails loudly on unsupported configurations."""

from __future__ import annotations

import argparse
import json

from ..config import ModelConfig, TrainingTokenBudget
from ..hardware import (
    ASSUMPTIONS,
    detect_hardware,
    estimate_memory,
    estimate_storage,
    estimate_training,
    validate_hardware_for_training,
)
from ..model.parameters import analytic_breakdown, format_breakdown


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shards", type=int, default=8)
    p.add_argument("--microbatch-tokens", type=int, default=32_768)
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    cfg = ModelConfig()
    cfg.validate()
    budget = TrainingTokenBudget()
    budget.validate()
    hw = detect_hardware()
    report = {
        "rootism_version": "0.1.0",
        "hardware": hw,
        "model": {
            "config_id": cfg.config_id,
            "parameters": analytic_breakdown(cfg)["total"],
            "context_length": cfg.max_position_embeddings,
        },
        "budget": {"target_tokens": budget.target_tokens, "hard_minimum": budget.hard_minimum_tokens},
        "memory": estimate_memory(cfg, shards=args.shards, microbatch_tokens=args.microbatch_tokens),
        "training": estimate_training(cfg, target_tokens=budget.target_tokens, gpu_count=args.shards),
        "storage": estimate_storage(target_tokens=budget.target_tokens),
        "assumptions": ASSUMPTIONS,
        "warnings": validate_hardware_for_training(
            cfg, shards=args.shards, microbatch_tokens=args.microbatch_tokens
        ),
    }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("Rootism environment report")
        print(f"  torch {hw['torch']} | CUDA: {hw['cuda_available']} | GPUs: {hw['cuda_device_count']}")
        print(f"  RAM: {hw['total_ram_gb']} GiB | BF16: {hw['bf16_supported']}")
        print(format_breakdown(analytic_breakdown(cfg)))
        print(f"  training budget: {budget.target_tokens:,} actual tokens (> {budget.hard_minimum_tokens:,})")
        print(f"  estimated peak/shard: {report['memory']['peak_per_shard_gb']:.1f} GiB")
        print(f"  estimated duration: {report['training']['days']:.1f} days on {args.shards} H100-class GPUs")
        for warning in report["warnings"]:
            print(f"  WARNING: {warning}")
        print("  NOTE: all figures are analytical estimates, not measurements.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
