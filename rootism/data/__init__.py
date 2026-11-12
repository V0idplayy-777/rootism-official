"""Rootism Commercial Training Data, Licensing, Deduplication & Sharding Package."""

from rootism.data.contamination import (
    BenchmarkItem,
    ContaminationCheckResult,
    DecontaminationEngine,
)
from rootism.data.dedup import (
    DedupDecision,
    DeduplicationEngine,
    extract_word_shingles,
    normalize_code_structure,
)
from rootism.data.inspector import inspect_prepared_dataset
from rootism.data.licensing import (
    APPROVED_COMMERCIAL_SPDX,
    DeepFileLicenseVerifier,
    FileLicenseScanResult,
    LicenseDecision,
    SpdxPolicyRule,
    evaluate_spdx_expression,
)
from rootism.data.mixture import DataMixtureSampler
from rootism.data.quality import DataQualityPipeline, QualityAssessment, shannon_char_entropy
from rootism.data.registry import (
    DatasetProvenanceError,
    DatasetProvenanceRecord,
    DatasetProvenanceRegistry,
)
from rootism.data.sharding import (
    RawCorpusDocument,
    RootismShardWriter,
    RootismStreamingShardDataset,
    ShardDescriptor,
    build_shards_parallel,
)
from rootism.data.synthetic import VerifiedSyntheticGenerator, VerifiedSyntheticSample

__all__ = [
    "APPROVED_COMMERCIAL_SPDX",
    "SpdxPolicyRule",
    "LicenseDecision",
    "FileLicenseScanResult",
    "evaluate_spdx_expression",
    "DeepFileLicenseVerifier",
    "DatasetProvenanceError",
    "DatasetProvenanceRecord",
    "DatasetProvenanceRegistry",
    "DataQualityPipeline",
    "QualityAssessment",
    "shannon_char_entropy",
    "DedupDecision",
    "DeduplicationEngine",
    "extract_word_shingles",
    "normalize_code_structure",
    "BenchmarkItem",
    "ContaminationCheckResult",
    "DecontaminationEngine",
    "VerifiedSyntheticGenerator",
    "VerifiedSyntheticSample",
    "RawCorpusDocument",
    "ShardDescriptor",
    "RootismShardWriter",
    "RootismStreamingShardDataset",
    "build_shards_parallel",
    "DataMixtureSampler",
    "inspect_prepared_dataset",
]
