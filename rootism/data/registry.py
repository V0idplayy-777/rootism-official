"""
Dataset Provenance Registry & Commercial Compliance Gate (Steps 36 & 38).

Maintains the 13 mandatory provenance attributes for every training data source,
blocks any unverified or commercially incompatible dataset, and generates
the third-party attribution notice file automatically.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from rootism.data.licensing import APPROVED_COMMERCIAL_SPDX, evaluate_spdx_expression


class DatasetProvenanceError(RuntimeError):
    """Raised when a dataset fails provenance completeness or commercial-use verification."""


class DatasetProvenanceRecord(BaseModel):
    """
    Complete 13-field provenance record mandated by Phase V Step 36.
    """

    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(min_length=2)
    dataset_name: str = Field(min_length=2)
    original_publisher: str = Field(min_length=2)
    original_source: str = Field(min_length=4)
    download_location: str = Field(min_length=4)
    exact_version: str = Field(min_length=1)
    licence: str = Field(min_length=2)
    licence_url: str = Field(min_length=8)
    commercial_use_permission: bool
    attribution_requirements: str = Field(min_length=2)
    redistribution_restrictions: str = Field(min_length=2)
    collection_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    processing_history: list[str] = Field(min_length=1)
    known_limitations: list[str] = Field(min_length=1)
    modality: str = Field(default="source_code_curated")

    @model_validator(mode="after")
    def _verify_commercial_compliance(self) -> "DatasetProvenanceRecord":
        if not self.commercial_use_permission:
            raise ValueError(
                f"Dataset '{self.dataset_name}' has commercial_use_permission=False and is strictly excluded."
            )
        decision = evaluate_spdx_expression(self.licence)
        if not decision.approved_for_commercial_training:
            raise ValueError(
                f"Dataset '{self.dataset_name}' license '{self.licence}' failed commercial compliance: "
                f"{decision.rejection_reason}"
            )
        lower_restr = self.redistribution_restrictions.lower()
        forbidden_phrases = ("non-commercial", "research only", "no commercial", "academic only")
        for phrase in forbidden_phrases:
            if phrase in lower_restr:
                raise ValueError(
                    f"Dataset '{self.dataset_name}' contains prohibited restriction '{phrase}' "
                    "in redistribution_restrictions."
                )
        return self


class DatasetProvenanceRegistry:
    """Validated registry of commercially cleared datasets and attribution notices."""

    def __init__(self, records: list[DatasetProvenanceRecord] | None = None) -> None:
        self._records: dict[str, DatasetProvenanceRecord] = {}
        for rec in records or []:
            self.register(rec)

    def register(self, record: DatasetProvenanceRecord) -> None:
        if record.dataset_id in self._records:
            raise DatasetProvenanceError(f"Duplicate dataset_id in provenance registry: {record.dataset_id}")
        self._records[record.dataset_id] = record

    def get(self, dataset_id: str) -> DatasetProvenanceRecord:
        if dataset_id not in self._records:
            raise DatasetProvenanceError(
                f"Dataset '{dataset_id}' is unregistered. Unregistered datasets cannot enter training shards."
            )
        return self._records[dataset_id]

    @property
    def records(self) -> list[DatasetProvenanceRecord]:
        return [self._records[k] for k in sorted(self._records.keys())]

    def registry_sha256(self) -> str:
        serialized = json.dumps(
            [r.model_dump(mode="json") for r in self.records],
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def generate_attribution_markdown(self) -> str:
        """Render complete third-party dataset attribution ledger for commercial compliance."""
        lines = [
            "# Rootism — Third-Party Dataset Provenance & Attribution Notices",
            "",
            f"**Registry SHA-256:** `{self.registry_sha256()}`  ",
            f"**Total Cleared Sources:** {len(self.records)}",
            "",
            "Every external dataset listed below has been verified against Rootism's SPDX commercial-use policy.",
            "",
        ]
        for rec in self.records:
            rule = APPROVED_COMMERCIAL_SPDX.get(rec.licence)
            attr_flag = "Required" if (rule and rule.requires_attribution) else "Not strictly required (documented for provenance)"
            lines.extend(
                [
                    f"## {rec.dataset_name} (`{rec.dataset_id}`)",
                    f"- **Publisher:** {rec.original_publisher}",
                    f"- **Original Source:** {rec.original_source}",
                    f"- **Download Location:** {rec.download_location}",
                    f"- **Exact Version:** `{rec.exact_version}`",
                    f"- **SPDX Licence:** `{rec.licence}` ({rec.licence_url})",
                    f"- **Commercial-Use Permission:** `{rec.commercial_use_permission}`",
                    f"- **Attribution Obligation ({attr_flag}):** {rec.attribution_requirements}",
                    f"- **Redistribution Conditions:** {rec.redistribution_restrictions}",
                    f"- **Collection Date:** `{rec.collection_date}`",
                    "",
                ]
            )
        return "\n".join(lines)

    def save_yaml(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"datasets": [r.model_dump(mode="json") for r in self.records]}
        with out.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(payload, fh, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DatasetProvenanceRegistry":
        src = Path(path)
        with src.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        entries = raw.get("datasets", [])
        if not isinstance(entries, list):
            raise DatasetProvenanceError(f"Malformed provenance registry YAML in {src}")
        parsed: list[DatasetProvenanceRecord] = []
        for item in entries:
            try:
                parsed.append(DatasetProvenanceRecord.model_validate(item))
            except Exception as exc:
                raise DatasetProvenanceError(
                    f"Dataset provenance validation failed in {src}: {exc}"
                ) from exc
        return cls(records=parsed)
