"""
Pre-Training Dataset, Provenance & Quality Inspection Toolkit (Step 45).

Allows users to audit shard manifests, decoded sample previews, domain/language
distributions, commercial license compliance, and token statistics before launching
a multi-hundred-billion-token training run.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from rootism.data.registry import DatasetProvenanceRegistry
from rootism.data.sharding import RootismStreamingShardDataset
from rootism.tokenizer.bpe import RootismTokenizer


def inspect_prepared_dataset(
    manifest_path: str | Path,
    tokenizer: RootismTokenizer,
    provenance_registry: DatasetProvenanceRegistry,
    *,
    num_preview_samples: int = 3,
    max_preview_chars: int = 240,
) -> dict[str, Any]:
    """
    Validate shard integrity, tokenizer binding, and provenance hash, and compile
    human-readable summary statistics and decoded previews.
    """
    ds = RootismStreamingShardDataset(
        manifest_path=manifest_path,
        expected_tokenizer_fingerprint=tokenizer.fingerprint,
        verify_checksums_on_open=True,
    )

    manifest_prov_sha = ds.manifest["provenance_registry_sha256"]
    live_prov_sha = provenance_registry.registry_sha256()
    provenance_match = manifest_prov_sha == live_prov_sha

    domain_token_counts: Counter[int] = Counter()
    language_token_counts: Counter[int] = Counter()
    doc_lengths: list[int] = []
    previews: list[dict[str, Any]] = []

    for idx, doc in enumerate(ds.iter_documents(seed=None)):
        tids = doc["token_ids"].tolist()
        t_len = len(tids)
        doc_lengths.append(t_len)
        domain_token_counts[int(doc["domain_id"])] += t_len
        language_token_counts[int(doc["language_id"])] += t_len

        if idx < num_preview_samples:
            decoded = tokenizer.decode(tids, skip_special_tokens=True)
            previews.append(
                {
                    "doc_index": idx,
                    "token_count": t_len,
                    "domain_id": int(doc["domain_id"]),
                    "language_id": int(doc["language_id"]),
                    "preview": decoded[:max_preview_chars],
                }
            )

    unique_src = int(ds.manifest["unique_source_tokens"])
    dedup_toks = int(ds.manifest["deduplicated_tokens"])

    return {
        "manifest_path": str(manifest_path),
        "dataset_name": ds.manifest["dataset_name"],
        "tokenizer_fingerprint_verified": True,
        "provenance_registry_sha256": manifest_prov_sha,
        "provenance_matches_active_registry": provenance_match,
        "cleared_datasets_in_registry": [
            {
                "dataset_id": r.dataset_id,
                "licence": r.licence,
                "commercial_use_permission": r.commercial_use_permission,
            }
            for r in provenance_registry.records
        ],
        "total_shards": int(ds.manifest["total_shards"]),
        "total_documents": ds.total_documents,
        "unique_source_tokens": unique_src,
        "deduplicated_tokens": dedup_toks,
        "dedup_retention_ratio": round(dedup_toks / float(max(1, unique_src)), 4),
        "min_doc_tokens": min(doc_lengths) if doc_lengths else 0,
        "max_doc_tokens": max(doc_lengths) if doc_lengths else 0,
        "mean_doc_tokens": round(sum(doc_lengths) / float(max(1, len(doc_lengths))), 2),
        "domain_token_distribution": dict(sorted(domain_token_counts.items())),
        "language_token_distribution": dict(sorted(language_token_counts.items())),
        "sample_previews": previews,
    }
