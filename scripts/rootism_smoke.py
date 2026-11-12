"""Stage-1 end-to-end smoke test on a TINY randomly initialised model.

This performs genuine gradient-based optimisation, real checkpoint round-trips
and real autoregressive generation. It is a PIPELINE VALIDATION ONLY: it does
not demonstrate that the production 3B model has learned anything.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

from rootism.accounting import TokenLedger
from rootism.config import TrainingTokenBudget, tiny_test_config
from rootism.io import checkpoint_path, load_model_checkpoint, save_model_checkpoint
from rootism.model.config_bridge import build_model
from rootism.model.generation import Generator, SamplingConfig
from rootism.model.parameters import analytic_breakdown


def main() -> None:
    torch.manual_seed(1234)
    cfg = tiny_test_config(vocab_size=257, max_position_embeddings=64)
    model = build_model(cfg, seed=1234)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ledger = TokenLedger()
    budget = TrainingTokenBudget()
    budget.validate()

    losses = []
    for step in range(6):
        input_ids = torch.randint(4, cfg.vocab_size, (2, 12))
        labels = input_ids.clone()
        labels[:, :2] = -100  # masked prompt prefix
        out = model(input_ids, labels=labels)
        opt.zero_grad(set_to_none=True)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(round(out.loss.item(), 5))
        ledger.record_step(
            presented_tokens=24, padded_tokens=0, masked_tokens=4, epoch=0, committed=True
        )

    with tempfile.TemporaryDirectory() as tmp:
        path = checkpoint_path(Path(tmp), step=6)
        save_model_checkpoint(model, path, step=6)
        restored, manifest = load_model_checkpoint(path)
        probe = torch.randint(4, cfg.vocab_size, (1, 7))
        identical = torch.equal(model(probe).logits, restored(probe).logits)

    gen = Generator(model)
    result = gen.generate([5, 6, 7], SamplingConfig(max_new_tokens=5, do_sample=False))

    report = {
        "config_id": cfg.config_id,
        "parameters_tiny": analytic_breakdown(cfg)["total"],
        "losses": losses,
        "loss_decreased": losses[-1] < losses[0],
        "ledger": ledger.to_dict(),
        "checkpoint_restored_identical": identical,
        "checkpoint_manifest_step": manifest["step"],
        "generation": {
            "token_ids": result.token_ids,
            "finish_reason": result.finish_reason,
        },
        "budget_target": budget.target_tokens,
        "note": "Tiny random-weight pipeline validation only; NOT a trained model.",
    }
    print(json.dumps(report, indent=2))
    if not all(torch.isfinite(torch.tensor(v)) for v in losses):
        raise SystemExit("non-finite loss")
    if not identical:
        raise SystemExit("checkpoint restoration mismatch")


if __name__ == "__main__":
    main()
