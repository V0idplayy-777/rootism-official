# Training-Token Budget and Accounting (requirements 05, 60)

## 1. Target

| Quantity | Value |
|---|---:|
| Initial target | **650,000,000,000 actual training tokens** |
| Hard floor | **> 600,000,000,000** (`HARD_MINIMUM_TRAINING_TOKENS = 600_000_000_000`, enforced as strictly greater) |
| Minimum deduplicated unique tokens | **162,500,000,000** (`min_unique_fraction = 0.25`) |
| Maximum epochs over the deduplicated corpus | 4 |
| Configurable larger budgets | allowed (`TrainingTokenBudget(target_tokens=1_000_000_000_000)` validates) |

`TrainingTokenBudget.validate()` rejects:
* any target ≤ 600B,
* any target < 650B without an explicit documented override (the default
  constructor takes 650B),
* a deduplicated corpus smaller than `target × min_unique_fraction`.

`preflight_budget()` must pass **before** the first training step. Stage 3's
trainer must call it. `assert_budget_complete()` refuses to declare a run complete
unless `committed_tokens ≥ target`, `committed_tokens > 600B`, ledger invariants
hold, and the repeat fraction is within bounds.

## 2. The seven counters (no inflation)

| Counter | Definition |
|---|---|
| `unique_source_tokens` | tokens in raw collected documents, before deduplication |
| `deduplicated_tokens` | tokens remaining after deduplication = size of one clean epoch |
| `presented_tokens` | every token slot fed to the model, **including padding** |
| `repeated_tokens` | processed tokens belonging to epoch ≥ 1 |
| `padded_tokens` / `masked_tokens` | presented slots excluded from the loss |
| `processed_training_tokens` | `presented − padded − masked` |
| `committed_tokens` | processed tokens in steps that completed and were checkpointed (**the only counter that counts**) |
| `failed_tokens` | processed tokens from failed/retried steps; never counted |

Invariants checked by `TokenLedger.invariant_errors()`:

```
presented - padded - masked == processed_training_tokens
committed + failed          == processed_training_tokens
sum(epochs.values())        == processed_training_tokens
committed                   <= processed_training_tokens
```

## 3. What is explicitly NOT counted

* Padding tokens (`<pad>` and any `attention_mask == 0` slot).
* Masked prompt tokens excluded from the loss (`labels == -100`).
* Tokens from steps that raised, produced NaN, or were retried.
* Tokens in a checkpoint that failed integrity validation.
* A step counter multiplied by a nominal batch size (no synthetic counting).

## 4. Wiring status

| Component | State |
|---|---|
| `TrainingTokenBudget` + validation | ✅ implemented + tested |
| `TokenLedger` + invariants + JSON persistence | ✅ implemented + tested |
| `preflight_budget` / `assert_budget_complete` | ✅ implemented + tested |
| Per-step `num_valid_tokens` from the model | ✅ implemented + tested (`RootismOutput.num_valid_tokens`) |
| Data pipeline reporting unique/dedup token counts | ❌ Stage 2 |
| Trainer wiring ledger to real distributed steps + shard manifests | ❌ Stage 3 |
| Stopping exactly at the configured budget | ❌ Stage 3 |

**No training token count is claimed.** Stage 1 validated the accounting machinery
with a 120-token, six-step smoke run; that number is a test fixture, not training
progress.

## 5. Anti-inflation rationale

With `max_epochs=4` and `min_unique_fraction=0.25`, a 650B-token run must be fed
by a deduplicated corpus of at least 162.5B unique tokens. Looping a 50B-token
corpus 13 times cannot satisfy the budget: `preflight_budget()` and
`assert_budget_complete()` both reject it. Padding cannot inflate the counter
because the model reports only non-`-100` labels and the ledger subtracts padding
and masked slots.
