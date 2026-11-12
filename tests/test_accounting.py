from __future__ import annotations

import pytest

from rootism.accounting import TokenLedger, assert_budget_complete, preflight_budget
from rootism.config import TrainingTokenBudget


def test_ledger_distinguishes_all_token_categories():
    ledger = TokenLedger()
    ledger.set_corpus_counts(unique_source_tokens=200_000, deduplicated_tokens=160_000)
    # 100 presented slots, 10 padding, 5 masked => 85 actual tokens
    assert ledger.record_step(presented_tokens=100, padded_tokens=10, masked_tokens=5, epoch=0) == 85
    assert ledger.record_step(presented_tokens=100, padded_tokens=0, masked_tokens=0, epoch=1) == 100
    assert ledger.record_step(presented_tokens=50, padded_tokens=0, masked_tokens=0, epoch=1,
                              committed=False) == 50
    assert ledger.presented_tokens == 250
    assert ledger.padded_tokens == 10
    assert ledger.masked_tokens == 5
    assert ledger.repeated_tokens == 150
    assert ledger.processed_training_tokens == 235
    assert ledger.committed_tokens == 185
    assert ledger.failed_tokens == 50
    assert ledger.invariant_errors() == []
    assert ledger.epochs == {"0": 85, "1": 150}
    assert ledger.unique_fraction == pytest.approx(35 / 185)


def test_ledger_rejects_impossible_counts():
    ledger = TokenLedger()
    with pytest.raises(ValueError):
        ledger.record_step(presented_tokens=10, padded_tokens=8, masked_tokens=5)
    with pytest.raises(ValueError):
        ledger.set_corpus_counts(unique_source_tokens=10, deduplicated_tokens=20)


def test_ledger_serialization(tmp_path):
    ledger = TokenLedger()
    ledger.record_step(presented_tokens=1000, epoch=0)
    path = tmp_path / "ledger.json"
    ledger.save(path)
    assert TokenLedger.load(path).to_dict() == ledger.to_dict()


def test_preflight_requires_enough_unique_tokens():
    budget = TrainingTokenBudget()
    report = preflight_budget(
        budget, unique_source_tokens=300_000_000_000, deduplicated_tokens=200_000_000_000
    )
    assert report["minimum_unique_tokens"] == 162_500_000_000
    assert report["target_tokens"] == 650_000_000_000
    with pytest.raises(ValueError, match="at least"):
        preflight_budget(budget, unique_source_tokens=100e9, deduplicated_tokens=100e9)
    with pytest.raises(ValueError, match="impossible"):
        preflight_budget(budget, unique_source_tokens=200e9, deduplicated_tokens=300e9)


def test_budget_completion_uses_actual_committed_tokens():
    budget = TrainingTokenBudget()
    ledger = TokenLedger()
    ledger.set_corpus_counts(unique_source_tokens=200e9, deduplicated_tokens=170e9)
    # 600B presented but padded/masked tokens must not count
    ledger.presented_tokens = 650_000_000_000
    ledger.padded_tokens = 50_000_000_000
    ledger.masked_tokens = 1_000_000_000  # deliberately inconsistent ledger
    ledger.processed_training_tokens = 600_000_000_000
    ledger.committed_tokens = 600_000_000_000
    ledger.epochs = {"0": 170e9, "1": 170e9, "2": 170e9, "3": 90e9}
    with pytest.raises(ValueError, match="invariants broken"):
        assert_budget_complete(budget, ledger)

    good = TokenLedger()
    good.set_corpus_counts(unique_source_tokens=200e9, deduplicated_tokens=170e9)
    good.presented_tokens = 700_000_000_000
    good.padded_tokens = 50_000_000_000
    good.processed_training_tokens = 650_000_000_000
    good.committed_tokens = 650_000_000_000
    good.repeated_tokens = 480_000_000_000
    good.epochs = {"0": 170e9, "1": 170e9, "2": 170e9, "3": 140e9}
    assert_budget_complete(budget, good)  # exactly 650B committed, no exception
    assert good.progress(budget)["fraction_complete"] == 1.0


def test_completion_rejects_early_stop_and_inflation():
    budget = TrainingTokenBudget()
    ledger = TokenLedger()
    ledger.presented_tokens = 600_000_000_001
    ledger.committed_tokens = 600_000_000_001
    ledger.processed_training_tokens = 600_000_000_001
    ledger.epochs = {"0": 600_000_000_001}
    with pytest.raises(ValueError, match="stopped early"):
        assert_budget_complete(budget, ledger)
    inflated = TokenLedger()
    inflated.presented_tokens = 650_000_000_000
    inflated.committed_tokens = 650_000_000_000
    inflated.processed_training_tokens = 650_000_000_000
    inflated.repeated_tokens = 600_000_000_000
    inflated.epochs = {"1": 650_000_000_000}
    with pytest.raises(ValueError, match="repeated epochs"):
        assert_budget_complete(budget, inflated)
