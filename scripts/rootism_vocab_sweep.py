"""Empirical vocabulary-size analysis (requirement 23).

Trains independent Rootism BPE tokenizers at several vocabulary sizes on the
original pilot corpus, measures compression and identifier fragmentation, fits
a log-linear curve, and combines it with the exact parameter cost of each
vocabulary size at a fixed ~3B model budget.

LIMITATION (must be read): the pilot corpus is ~4 MB of ORIGINAL synthetic code,
not the production corpus. A 4 MB corpus cannot support 48K learned merges, so
values at 32K/48K/64K are EXTRAPOLATIONS, not measurements. Re-run this script
against the licensed production corpus before freezing the tokenizer.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from rootism.config import TokenizerConfig
from rootism.model.parameters import analytic_breakdown, search_depth_for_target
from rootism.tokenizer.evaluation import evaluate_tokenizer
from rootism.tokenizer.pilot_corpus import build_pilot_corpus, pilot_eval_samples
from rootism.tokenizer.train import train_tokenizer

MEASURED_SIZES = (512, 1024, 2048, 4096, 8192, 16384)
CANDIDATES = (32768, 49152, 65536)


def fit_log_linear(points: list[tuple[int, float]]) -> tuple[float, float, float]:
    """Fit log(bpt) = a + b*log(v); return (a, b, R^2)."""
    xs = [math.log(v) for v, _ in points]
    ys = [math.log(y) for _, y in points]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    a = my - b * mx
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    return a, b, r2


def parameter_cost(vocab: int) -> dict:
    rows = search_depth_for_target(
        vocab_size=vocab, hidden_size=3072, num_attention_heads=24,
        num_key_value_heads=8, head_dim=128, intermediate_size=8192,
    )
    best = min(rows, key=lambda r: r["abs_error"])
    from rootism.config import ModelConfig

    cfg = ModelConfig(vocab_size=vocab, num_hidden_layers=int(best["layers"]))
    counts = analytic_breakdown(cfg)
    return {
        "depth": int(best["layers"]),
        "total_parameters": int(best["total"]),
        "embedding_plus_head_parameters": counts["embeddings"] + counts["output_projection"],
        "body_parameters": counts["attention"] + counts["feed_forward"],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--documents", type=int, default=12_000)
    p.add_argument("--output", default="artifacts/vocab_sweep.json")
    args = p.parse_args()

    docs = build_pilot_corpus(generated_documents=args.documents)
    samples = pilot_eval_samples()
    corpus_bytes = sum(len(d.encode("utf-8")) for d in docs)
    print(f"pilot corpus: {len(docs):,} documents, {corpus_bytes:,} bytes\n")

    measured: list[tuple[int, float]] = []
    rows = []
    for size in MEASURED_SIZES:
        tok = train_tokenizer(docs, config=TokenizerConfig(vocab_size=size, min_pair_frequency=1))
        report = evaluate_tokenizer(tok, samples)
        achieved = tok.vocab_size
        row = {
            "requested_vocab_size": size,
            "achieved_vocab_size": achieved,
            "reached_target": achieved == size,
            "bytes_per_token": report["bytes_per_token"],
            "identifier_fragmentation_rate": report["identifier_fragmentation_rate"],
            "identifier_mean_tokens": report["identifier_mean_tokens"],
        }
        rows.append(row)
        # Fit on the ACTUAL learned vocabulary size, even when the request could
        # not be reached (the corpus ran out of sufficiently frequent pairs).
        measured.append((achieved, report["bytes_per_token"]))
        print(
            f"  requested {size:>6,} -> achieved {achieved:>6,}  "
            f"bytes/token {report['bytes_per_token']:.4f}  "
            f"id-fragmentation {report['identifier_fragmentation_rate']:.3f}"
        )

    deduplicated = sorted(set(measured))
    a, b, r2 = fit_log_linear(deduplicated)
    print(f"\nfit: log(bytes/token) = {a:.4f} + {b:.4f} * log(vocab), R^2 = {r2:.4f}")
    print("Fitted on achieved vocabulary sizes; extrapolated values are NOT measurements.")
    print("The fit is a power law and is only a weak guide outside the measured range.\n")
    extrapolated = {}
    for size in CANDIDATES:
        predicted = math.exp(a + b * math.log(size))
        cost = parameter_cost(size)
        extrapolated[size] = {
            "predicted_bytes_per_token": predicted,
            **cost,
        }
        print(
            f"  vocab {size:>6,}: predicted bytes/token {predicted:.4f}, "
            f"best depth {cost['depth']}, total {cost['total_parameters']:,}, "
            f"emb+head {cost['embedding_plus_head_parameters']:,}"
        )

    gain = extrapolated[65536]["predicted_bytes_per_token"] - extrapolated[49152]["predicted_bytes_per_token"]
    print(f"\n64K vs 48K predicted compression gain: {gain:.4f} bytes/token "
          f"({100 * gain / extrapolated[49152]['predicted_bytes_per_token']:.2f}%)")
    print("cost of 64K: ~100.7M more embedding/head parameters and 2 fewer transformer layers")
    print("at the same ~3B budget. Selected: 49,152 (see docs/TOKENIZER.md).")

    output = {
        "corpus": {"documents": len(docs), "bytes": corpus_bytes, "origin": "original Rootism pilot corpus"},
        "measured": rows,
        "fit": {"intercept": a, "slope": b, "r_squared": r2},
        "extrapolated": {str(k): v for k, v in extrapolated.items()},
        "decision": {"vocab_size": 49_152, "basis": "measured compression curve + exact parameter cost; extrapolated region"},
        "limitation": "Pilot corpus too small to learn >48K merges; production re-validation required.",
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
