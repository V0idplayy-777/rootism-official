"""Print the exact Rootism parameter breakdown and verify it against tensors."""

from __future__ import annotations

import argparse
import json

import torch

from ..config import ModelConfig
from ..hardware import estimate_memory, estimate_training
from ..model.config_bridge import build_model
from ..model.parameters import (
    analytic_breakdown,
    empirical_breakdown,
    format_breakdown,
    kv_cache_bytes_per_token,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", help="path to a serialized ModelConfig JSON")
    p.add_argument("--verify-meta", action="store_true", help="instantiate on meta device and compare counts")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    cfg = ModelConfig.load(args.config) if args.config else ModelConfig()
    cfg.validate()
    analytic = analytic_breakdown(cfg)
    result: dict = {
        "config_id": cfg.config_id,
        "config": cfg.to_dict(),
        "analytic": analytic,
    }
    if args.verify_meta:
        model = build_model(cfg, device="meta", init_weights=False)
        empirical = empirical_breakdown(model)
        result["empirical"] = empirical
        result["counts_match"] = all(empirical[k] == analytic[k] for k in (
            "embeddings", "attention", "feed_forward", "normalization", "output_projection", "total"
        ))
        if not result["counts_match"]:
            raise SystemExit("FATAL: analytic and empirical parameter counts disagree")
    result["kv_cache_bytes_per_token_bf16"] = kv_cache_bytes_per_token(cfg, torch.bfloat16)
    result["memory_estimate_8_shards"] = estimate_memory(cfg)
    result["training_estimate_8_h100"] = estimate_training(cfg)

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(format_breakdown(analytic, title=f"Rootism {cfg.config_id}"))
        print()
        print(f"  KV cache (bf16): {kv_cache_bytes_per_token(cfg, torch.bfloat16) / 1024:.1f} KiB/token")
        if args.verify_meta:
            print(f"  meta-device verification: {'MATCH' if result['counts_match'] else 'MISMATCH'}")
        print(f"  estimated static state (8 shards): {estimate_memory(cfg)['static_per_shard_gb']:.1f} GiB/shard")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
