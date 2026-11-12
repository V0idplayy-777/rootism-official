"""
Configurable Multi-Domain Curriculum Data Mixture Engine (Step 44).

Implements deterministic weighted domain sampling over `.rshard` datasets,
tracks exact per-domain epoch rollover so `repeated_epoch_tokens` are distinguished
from first-pass tokens (Step 05), and enforces `max_epochs` saturation limits.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from rootism.config import CurriculumStageConfig
from rootism.data.sharding import RootismStreamingShardDataset


class DataMixtureSampler:
    """
    Deterministic weighted mixture sampler across named domain shard datasets.
    """

    def __init__(
        self,
        stage_config: CurriculumStageConfig,
        domain_datasets: dict[str, RootismStreamingShardDataset],
        *,
        seed: int = 20261112,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.stage_config = stage_config
        self.domain_datasets = domain_datasets
        self.seed = seed
        self.rank = rank
        self.world_size = world_size

        missing = [d.domain_id for d in stage_config.domains if d.domain_id not in domain_datasets]
        if missing:
            raise ValueError(f"Missing shard datasets for mixture domains: {missing}")

        self.domain_ids: list[str] = [d.domain_id for d in stage_config.domains]
        self.weights = np.array([d.weight for d in stage_config.domains], dtype=np.float64)
        self.weights /= self.weights.sum()
        self.max_epochs: dict[str, float] = {d.domain_id: d.max_epochs for d in stage_config.domains}

        self.domain_epochs: dict[str, int] = {dom: 0 for dom in self.domain_ids}
        self.domain_emitted_tokens: dict[str, int] = {dom: 0 for dom in self.domain_ids}

    def stream(self) -> Iterator[dict[str, Any]]:
        """
        Yield documents sampled according to `stage_config` domain weights,
        tagging each sample with `is_repeated_epoch` (`epoch > 0`).
        """
        rng = np.random.default_rng(self.seed + self.rank * 7919)
        iterators: dict[str, Iterator[dict[str, Any]]] = {
            dom: self.domain_datasets[dom].iter_documents(
                seed=self.seed, epoch=0, rank=self.rank, world_size=self.world_size
            )
            for dom in self.domain_ids
        }

        while True:
            dom_idx = int(rng.choice(len(self.domain_ids), p=self.weights))
            dom = self.domain_ids[dom_idx]
            try:
                doc = next(iterators[dom])
            except StopIteration:
                self.domain_epochs[dom] += 1
                next_ep = self.domain_epochs[dom]
                if float(next_ep) >= self.max_epochs[dom]:
                    raise RuntimeError(
                        f"Domain '{dom}' exceeded configured max_epochs ({self.max_epochs[dom]}) "
                        "to prevent memorisation overfitting."
                    ) from None
                iterators[dom] = self.domain_datasets[dom].iter_documents(
                    seed=self.seed,
                    epoch=next_ep,
                    rank=self.rank,
                    world_size=self.world_size,
                )
                doc = next(iterators[dom])

            n_toks = int(len(doc["token_ids"]))
            self.domain_emitted_tokens[dom] += n_toks
            yield {
                "domain_id_name": dom,
                "token_ids": doc["token_ids"],
                "epoch": self.domain_epochs[dom],
                "is_repeated_epoch": self.domain_epochs[dom] > 0,
            }
