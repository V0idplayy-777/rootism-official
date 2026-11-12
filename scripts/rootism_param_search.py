"""Enumerate depths near the ~3B parameter target and print the trade-off table.

Usage: PYTHONPATH=src python scripts/rootism_param_search.py
"""

from __future__ import annotations

from rootism.config import ModelConfig
from rootism.model.parameters import analytic_breakdown, search_depth_for_target


def main() -> None:
    print("Vocabulary-size / depth trade-off at a ~3.0B parameter budget")
    print("(d=3072, 24 query heads, 8 KV heads, head_dim=128, ffn=8192, untied head)\n")
    for vocab in (32_768, 49_152, 65_536):
        rows = search_depth_for_target(
            vocab_size=vocab,
            hidden_size=3_072,
            num_attention_heads=24,
            num_key_value_heads=8,
            head_dim=128,
            intermediate_size=8_192,
        )
        best = min(rows, key=lambda r: r["abs_error"])
        cfg = ModelConfig(vocab_size=vocab, num_hidden_layers=int(best["layers"]))
        counts = analytic_breakdown(cfg)
        print(
            f"vocab={vocab:>6,}  best_depth={int(best['layers']):>2}  "
            f"total={int(best['total']):,}  "
            f"emb+head={counts['embeddings'] + counts['output_projection']:,}  "
            f"body={counts['attention'] + counts['feed_forward']:,}"
        )
    print("\nChosen preset: vocab=49,152, depth=28 (see docs/MODEL.md).")
    print("Rationale: ~3.12B total, maximum depth at this budget, 48K vocabulary")
    print("keeps 100M parameters in transformer body rather than a second 64K head.")


if __name__ == "__main__":
    main()
