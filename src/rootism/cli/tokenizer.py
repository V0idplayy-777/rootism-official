"""Tokenizer CLI: train, evaluate, inspect, verify."""

from __future__ import annotations

import argparse
import json

from ..config import TokenizerConfig
from ..tokenizer.evaluation import evaluate_tokenizer
from ..tokenizer.pilot_corpus import build_pilot_corpus, pilot_eval_samples
from ..tokenizer.tokenizer import RootismTokenizer
from ..tokenizer.train import iter_text_files, train_tokenizer


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="train a tokenizer on UTF-8 text files")
    t.add_argument("--corpus", nargs="+", required=True)
    t.add_argument("--output", required=True)
    t.add_argument("--vocab-size", type=int, default=49_152)
    t.add_argument("--min-pair-frequency", type=int, default=2)
    t.add_argument("--verbose", action="store_true")

    e = sub.add_parser("evaluate", help="evaluate a tokenizer on a sample corpus")
    e.add_argument("--tokenizer", required=True)
    e.add_argument("--pilot", action="store_true", help="use the built-in original pilot samples")
    e.add_argument("--corpus", nargs="*", default=[])

    i = sub.add_parser("info", help="print tokenizer identity and statistics")
    i.add_argument("--tokenizer", required=True)

    v = sub.add_parser("verify", help="verify tokenizer integrity")
    v.add_argument("--tokenizer", required=True)

    s = sub.add_parser("pilot", help="train on the built-in original pilot corpus")
    s.add_argument("--output", required=True)
    s.add_argument("--vocab-size", type=int, default=4_096)
    s.add_argument("--documents", type=int, default=400)

    args = p.parse_args(argv)

    if args.command == "train":
        cfg = TokenizerConfig(
            vocab_size=args.vocab_size, min_pair_frequency=args.min_pair_frequency
        )
        tok = train_tokenizer(iter_text_files(args.corpus), config=cfg, verbose=args.verbose)
        tok.save(args.output)
        print(json.dumps({
            "tokenizer_id": tok.tokenizer_id,
            "vocab_size": tok.vocab_size,
            "corpus_sha256": tok.model.corpus_sha256,
            "documents": tok.model.corpus_documents,
            "path": args.output,
        }, indent=2))
    elif args.command == "pilot":
        cfg = TokenizerConfig(vocab_size=args.vocab_size)
        docs = build_pilot_corpus(generated_documents=args.documents)
        tok = train_tokenizer(docs, config=cfg, validation_documents=docs[:20])
        tok.save(args.output)
        print(json.dumps({
            "tokenizer_id": tok.tokenizer_id,
            "vocab_size": tok.vocab_size,
            "documents": len(docs),
            "corpus_bytes": tok.model.corpus_bytes,
            "path": args.output,
            "note": "PILOT artifact only; not for production training",
        }, indent=2))
    elif args.command in {"evaluate", "info", "verify"}:
        tok = RootismTokenizer.from_file(args.tokenizer)
        if args.command == "verify":
            print(json.dumps({"valid": True, "tokenizer_id": tok.tokenizer_id}, indent=2))
        elif args.command == "info":
            print(json.dumps(tok.to_dict(), indent=2)[:4000])
        else:
            samples = list(pilot_eval_samples()) if args.pilot else []
            for path in args.corpus:
                for i, text in enumerate(iter_text_files([path])):
                    from ..tokenizer.evaluation import EvalSample

                    samples.append(EvalSample("corpus", "mixed", text))
            print(json.dumps(evaluate_tokenizer(tok, samples), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
