"""
Rootism Unified Command-Line Interface (Phases I–XII).

Provides cohesive operator access to:
- `inspect-model`: Programmatic parameter & KV-cache audit + candidate comparison
- `plan-hardware`: Memory, storage, throughput & 650B duration estimates
- `token-allocation`: 650B modality and 19-language token budgets
- `verify-licenses`: Commercial provenance registry audit & attribution export
- `train-tokenizer`: Reproducible byte-level BPE training
- `inspect-data`: `.rshard` dataset manifest & licence inspection
- `generate-synthetic`: Subprocess-verified synthetic training data generation
- `train-verify`: End-to-end resumable `RootismPretrainer` smoke/verification run
- `quantize-eval`: Non-destructive INT8/INT4 quantisation quality & memory benchmark
- `evaluate`: Executable multi-language coding, reasoning & generalisation benchmark suite
- `serve`: Launch production FastAPI + Root Developer Platform server (`0.0.0.0`)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch
import uvicorn

from rootism.config import RootismConfig, RootismModelConfig, RootismTokenizerConfig
from rootism.data.inspector import inspect_prepared_dataset
from rootism.data.registry import DatasetProvenanceRegistry
from rootism.data.synthetic import VerifiedSyntheticGenerator
from rootism.evaluation.coding_bench import CodingBenchmarkSuite
from rootism.evaluation.generalization import GeneralizationVerifier
from rootism.evaluation.reasoning_bench import ReasoningBenchmarkSuite
from rootism.evaluation.regression import EvaluationReportBuilder
from rootism.inference.engine import (
    GenerationSamplingParams,
    RootismInferenceEngine,
    bootstrap_local_serving_artifacts,
)
from rootism.inference.quantization import (
    evaluate_quantization_quality,
    quantize_model_non_destructive,
)
from rootism.languages.allocation import compute_training_token_allocation
from rootism.model.inspector import evaluate_architecture_candidates, inspect_model_parameters
from rootism.model.transformer import RootismForCausalLM
from rootism.server.app import create_rootism_app
from rootism.tokenizer.bpe import RootismTokenizer
from rootism.tokenizer.trainer import RootismTokenizerTrainer
from rootism.training.hardware import (
    GPU_SPEC_PRESETS,
    detect_hardware_capabilities,
    estimate_training_infrastructure,
)
from rootism.training.trainer import RootismPretrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rootism",
        description="Rootism Engineering, Training, Evaluation & Serving CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_model = sub.add_parser("inspect-model", help="Programmatically audit Rootism model parameters & KV cache.")
    p_model.add_argument("--config", type=str, default=None, help="Optional YAML master config path.")
    p_model.add_argument("--compare-candidates", action="store_true", help="Include candidate architecture comparison.")

    p_hw = sub.add_parser("plan-hardware", help="Calculate memory, storage, throughput & duration estimates.")
    p_hw.add_argument("--gpu", type=str, default="h100_80gb_sxm", choices=sorted(GPU_SPEC_PRESETS.keys()))
    p_hw.add_argument("--num-gpus", type=int, default=64)
    p_hw.add_argument("--seq-len", type=int, default=4096)
    p_hw.add_argument("--micro-batch", type=int, default=4)
    p_hw.add_argument("--tokens", type=int, default=650_000_000_000)
    p_hw.add_argument("--ckpt", type=str, default="selective", choices=["none", "selective", "full"])

    p_alloc = sub.add_parser("token-allocation", help="Display 650B modality and language token budgets.")
    p_alloc.add_argument("--tokens", type=int, default=650_000_000_000)

    p_lic = sub.add_parser("verify-licenses", help="Validate dataset provenance registry & optionally write notices.")
    p_lic.add_argument("--registry", type=str, default="configs/datasets/provenance_registry.yaml")
    p_lic.add_argument("--write-notices", type=str, default=None)

    p_tok = sub.add_parser("train-tokenizer", help="Train a Rootism byte-level BPE tokenizer from text files.")
    p_tok.add_argument("--input", type=str, required=True, help="Input text file or directory.")
    p_tok.add_argument("--output", type=str, required=True, help="Output tokenizer JSON path.")
    p_tok.add_argument("--vocab-size", type=int, default=65_536)

    p_data = sub.add_parser("inspect-data", help="Audit prepared .rshard dataset manifest, licences & previews.")
    p_data.add_argument("--manifest", type=str, required=True)
    p_data.add_argument("--tokenizer", type=str, required=True)
    p_data.add_argument("--registry", type=str, default="configs/datasets/provenance_registry.yaml")

    p_syn = sub.add_parser("generate-synthetic", help="Generate execution-verified synthetic training tasks.")
    p_syn.add_argument("--count-per-type", type=int, default=4)
    p_syn.add_argument("--seed", type=int, default=20261112)
    p_syn.add_argument("--output", type=str, default=None)

    p_tv = sub.add_parser("train-verify", help="Run resumable RootismPretrainer verification pass.")
    p_tv.add_argument("--config", type=str, default="configs/model/rootism_tiny_verify.yaml")
    p_tv.add_argument("--steps", type=int, default=4)
    p_tv.add_argument("--checkpoint-dir", type=str, default="artifacts/verify_train_ckpts")

    p_qe = sub.add_parser("quantize-eval", help="Evaluate non-destructive INT8 & INT4 quantisation trade-offs.")
    p_qe.add_argument("--config", type=str, default="configs/model/rootism_tiny_verify.yaml")

    p_ev = sub.add_parser("evaluate", help="Run Coding, Reasoning & Generalisation evaluation harness.")
    p_ev.add_argument("--artifact-dir", type=str, default="artifacts/local_runtime")
    p_ev.add_argument("--output-dir", type=str, default="artifacts/eval_reports")

    p_srv = sub.add_parser("serve", help="Start FastAPI server + Root Developer Platform on 0.0.0.0.")
    p_srv.add_argument("--host", type=str, default="0.0.0.0")
    p_srv.add_argument("--port", type=int, default=8000)
    p_srv.add_argument("--checkpoint-dir", type=str, default=None)
    p_srv.add_argument("--tokenizer-dir", type=str, default=None)
    p_srv.add_argument("--quantization", type=str, default="none", choices=["none", "int8", "int4"])

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "inspect-model":
        cfg = RootismConfig.from_yaml(args.config).model if args.config else RootismModelConfig()
        report = inspect_model_parameters(cfg).to_dict()
        if args.compare_candidates:
            report["candidate_comparison"] = evaluate_architecture_candidates()
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "plan-hardware":
        est = estimate_training_infrastructure(
            gpu_preset=args.gpu,
            num_gpus=args.num_gpus,
            sequence_length=args.seq_len,
            micro_batch_size=args.micro_batch,
            target_training_tokens=args.tokens,
            activation_checkpointing=args.ckpt,
        )
        out = {
            "active_local_machine": detect_hardware_capabilities().to_dict(),
            "cluster_plan": est.to_dict(),
        }
        print(json.dumps(out, indent=2))
        return 0

    if args.command == "token-allocation":
        print(json.dumps(compute_training_token_allocation(args.tokens), indent=2))
        return 0

    if args.command == "verify-licenses":
        reg = DatasetProvenanceRegistry.from_yaml(args.registry)
        notices = reg.generate_attribution_markdown()
        if args.write_notices:
            Path(args.write_notices).write_text(notices, encoding="utf-8")
        print(
            json.dumps(
                {
                    "status": "verified",
                    "registry_sha256": reg.registry_sha256(),
                    "dataset_count": len(reg.records),
                    "datasets": [r.dataset_id for r in reg.records],
                },
                indent=2,
            )
        )
        return 0

    if args.command == "train-tokenizer":
        src = Path(args.input)
        docs: list[str] = []
        if src.is_file():
            docs.append(src.read_text(encoding="utf-8", errors="ignore"))
        else:
            for fp in sorted(src.rglob("*")):
                if fp.is_file() and fp.stat().st_size <= 5_000_000:
                    docs.append(fp.read_text(encoding="utf-8", errors="ignore"))
        tok_cfg = RootismTokenizerConfig(vocab_size=args.vocab_size)
        tok, rep = RootismTokenizerTrainer(tok_cfg).train_from_iterator(docs)
        tok.save(args.output)
        print(json.dumps(rep.to_dict(), indent=2))
        return 0

    if args.command == "inspect-data":
        tok = RootismTokenizer.load(args.tokenizer)
        reg = DatasetProvenanceRegistry.from_yaml(args.registry)
        summary = inspect_prepared_dataset(args.manifest, tok, reg)
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "generate-synthetic":
        gen = VerifiedSyntheticGenerator(seed=args.seed)
        samples = gen.generate_batch(count_per_type=args.count_per_type)
        payload = [s.to_dict() for s in samples]
        if args.output:
            out_p = Path(args.output)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {
                    "generated_verified_samples": len(samples),
                    "rejected_unverified_samples": gen.rejected_verification_count,
                },
                indent=2,
            )
        )
        return 0

    if args.command == "train-verify":
        cfg = RootismConfig.from_yaml(args.config)
        train_cfg = cfg.training.model_copy(
            update={"checkpoint_dir": args.checkpoint_dir, "checkpoint_interval_steps": max(1, args.steps // 2)}
        )
        cfg = cfg.model_copy(update={"training": train_cfg})
        trainer = RootismPretrainer(cfg)
        seq_len = cfg.training.sequence_length
        docs = [
            {"token_ids": [((i * 17 + j) % 300) + 64 for j in range(seq_len // 2 + 11)]}
            for i in range(args.steps * cfg.training.micro_batch_size * 4)
        ]
        summary = trainer.train(docs, max_steps=args.steps)
        print(json.dumps(summary.to_dict(), indent=2))
        return 0

    if args.command == "quantize-eval":
        cfg = RootismConfig.from_yaml(args.config)
        model = RootismForCausalLM(cfg.model).eval()
        eval_ids = torch.randint(64, 320, (2, 64), dtype=torch.long)
        q8 = quantize_model_non_destructive(model, "int8")
        q4 = quantize_model_non_destructive(model, "int4")
        rep8 = evaluate_quantization_quality(model, q8, eval_ids, mode="int8")
        rep4 = evaluate_quantization_quality(model, q4, eval_ids, mode="int4")
        print(json.dumps({"int8": rep8.to_dict(), "int4": rep4.to_dict()}, indent=2))
        return 0

    if args.command == "evaluate":
        ckpt_dir, tok_file = bootstrap_local_serving_artifacts(args.artifact_dir)
        engine = RootismInferenceEngine(checkpoint_dir=ckpt_dir, tokenizer_dir=tok_file)
        coding_suite = CodingBenchmarkSuite()
        reasoning_suite = ReasoningBenchmarkSuite(tasks_per_faculty=2)
        gen_verifier = GeneralizationVerifier()

        coding_rep = coding_suite.run_suite(
            lambda task: engine.generate_chat(
                user_message=task.prompt,
                reasoning_mode="concise",
                params=GenerationSamplingParams(max_new_tokens=24, temperature=0.0),
            ).visible_answer
        )
        reasoning_rep = reasoning_suite.run_suite(
            lambda task: engine.generate_chat(
                user_message=task.prompt,
                reasoning_mode="structured",
                params=GenerationSamplingParams(max_new_tokens=24, temperature=0.0),
            ).visible_answer
        )
        gen_rep = gen_verifier.verify_generalization(
            training_corpus_texts=["def helper(x): return x + 1"],
            solver_fn=lambda prompt: engine.generate_chat(
                user_message=prompt,
                reasoning_mode="concise",
                params=GenerationSamplingParams(max_new_tokens=16, temperature=0.0),
            ).visible_answer,
        )
        json_p, md_p = EvaluationReportBuilder.write_reports(
            args.output_dir,
            runtime_meta=engine.runtime_metadata(),
            coding_report=coding_rep,
            reasoning_report=reasoning_rep,
            generalization_report=gen_rep.to_dict(),
        )
        print(
            json.dumps(
                {
                    "json_report": str(json_p),
                    "markdown_report": str(md_p),
                    "coding_tasks": coding_rep["total_tasks"],
                    "reasoning_tasks": reasoning_rep["total_tasks"],
                    "decontamination_clean": gen_rep.decontamination_clean,
                },
                indent=2,
            )
        )
        return 0

    if args.command == "serve":
        app = create_rootism_app(
            checkpoint_dir=args.checkpoint_dir,
            tokenizer_dir=args.tokenizer_dir,
            quantization=args.quantization,  # type: ignore[arg-type]
        )
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
