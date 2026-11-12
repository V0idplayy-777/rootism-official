"""
5-Level Multi-Granularity Deduplication Engine (Step 42).

Implements:
1. Exact document deduplication (SHA-256)
2. Near-duplicate detection via MinHash + Locality-Sensitive Hashing (LSH) + exact Jaccard check
3. Code-specific structural deduplication (comment-stripped & whitespace-normalized clone detection)
4. Repeated non-idiomatic snippet suppression (while preserving legitimate idiomatic patterns!)
5. Repository-level file-tree overlap detection (catches forked repositories)
"""

from __future__ import annotations

import hashlib
import re
import zlib
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any

_MERSENNE_61: int = (1 << 61) - 1
_MAX_HASH_64: int = (1 << 64) - 1

# Idiomatic lines exempt from snippet-boilerplate suppression so common patterns remain intact
_IDIOMATIC_LINE_EXEMPTIONS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*if\s+__name__\s*==\s*['\"]__main__['\"]\s*:"),
    re.compile(r"^\s*(?:import|from)\s+[A-Za-z0-9_.*,\s]+$"),
    re.compile(r"^\s*#include\s+[<\"][^>\"]+[>\"]$"),
    re.compile(r"^\s*use\s+[A-Za-z0-9_:{},\s*]+;$"),
    re.compile(r"^\s*package\s+[A-Za-z0-9_.]+;?$"),
    re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\b"),
    re.compile(r"^\s*return\s+(?:None|null|nil|0|true|false|Ok\(.+\));?$"),
)

_C_STYLE_COMMENTS_RE = re.compile(r"//[^\n]*|/\*[\s\S]*?\*/")
_PY_HASH_COMMENTS_RE = re.compile(r"#[^\n]*")
_WS_COLLAPSE_RE = re.compile(r"\s+")


_CODE_TOKEN_NORM_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+(?:\.[0-9]+)?|'[\\s\\S]*?'|\"[\\s\\S]*?\"|==|!=|<=|>=|->|=>|::|&&|\|\||[^\s]")


def normalize_code_structure(source: str, language: str | None = None) -> str:
    """Normalize comments and cosmetic formatting while keeping identifiers, literals, and operators."""
    lang = (language or "").lower()
    stripped = source
    if lang in ("python", "bash", "ruby", "powershell"):
        stripped = _PY_HASH_COMMENTS_RE.sub("", stripped)
    else:
        stripped = _C_STYLE_COMMENTS_RE.sub("", stripped)
    tokens = _CODE_TOKEN_NORM_RE.findall(stripped)
    return " ".join(tokens)


def extract_word_shingles(text: str, k: int = 5) -> set[int]:
    """Extract deterministic 32-bit CRC32 hashes of k-token shingles."""
    tokens = re.findall(r"\w+|[^\w\s]", text.lower())
    if not tokens:
        return set()
    if len(tokens) < k:
        joined = " ".join(tokens).encode("utf-8")
        return {zlib.crc32(joined) & 0xFFFFFFFF}
    shingles: set[int] = set()
    for i in range(len(tokens) - k + 1):
        gram = " ".join(tokens[i : i + k]).encode("utf-8")
        shingles.add(zlib.crc32(gram) & 0xFFFFFFFF)
    return shingles


@dataclass(frozen=True)
class DedupDecision:
    kept: bool
    reason: str | None
    matched_doc_id: str | None
    jaccard_similarity: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DeduplicationEngine:
    """
    Stateful multi-level deduplicator supporting streaming document & repository ingestion.
    """

    def __init__(
        self,
        *,
        num_perm: int = 128,
        lsh_bands: int = 16,
        jaccard_threshold: float = 0.85,
        enable_code_structural_dedup: bool = True,
        repo_overlap_threshold: float = 0.70,
        max_non_idiomatic_block_repeats: int = 5,
    ) -> None:
        if num_perm % lsh_bands != 0:
            raise ValueError(f"num_perm ({num_perm}) must be divisible by lsh_bands ({lsh_bands})")
        self.num_perm = num_perm
        self.lsh_bands = lsh_bands
        self.rows_per_band = num_perm // lsh_bands
        self.jaccard_threshold = jaccard_threshold
        self.enable_code_structural_dedup = enable_code_structural_dedup
        self.repo_overlap_threshold = repo_overlap_threshold
        self.max_non_idiomatic_block_repeats = max_non_idiomatic_block_repeats

        # Deterministic universal hash coefficients (a_i * x + b_i) mod (2^61 - 1)
        self._coeff_a: list[int] = []
        self._coeff_b: list[int] = []
        for i in range(num_perm):
            digest = hashlib.sha256(f"rootism-minhash-perm-{i}".encode("utf-8")).digest()
            a = (int.from_bytes(digest[:8], "little") % (_MERSENNE_61 - 1)) + 1
            b = int.from_bytes(digest[8:16], "little") % _MERSENNE_61
            self._coeff_a.append(a)
            self._coeff_b.append(b)

        self._exact_hashes: dict[str, str] = {}
        self._structural_code_hashes: dict[str, str] = {}
        self._lsh_buckets: list[dict[bytes, list[str]]] = [
            defaultdict(list) for _ in range(lsh_bands)
        ]
        self._doc_shingles: dict[str, set[int]] = {}
        self._block_seen_counts: dict[str, int] = defaultdict(int)
        self._repo_file_hashes: dict[str, set[str]] = {}

        self.stats: dict[str, int] = defaultdict(int)

    def _compute_minhash_signature(self, shingles: set[int]) -> list[int]:
        if not shingles:
            return [_MERSENNE_61] * self.num_perm
        sig: list[int] = []
        p = _MERSENNE_61
        for a, b in zip(self._coeff_a, self._coeff_b, strict=True):
            min_val = min(((a * s + b) % p) for s in shingles)
            sig.append(min_val)
        return sig

    def check_repository_overlap(self, repo_id: str, file_contents: dict[str, str]) -> DedupDecision:
        """
        Level 5: Detect near-duplicate forked repositories via file SHA-256 overlap,
        without falsely rejecting repos that only share tiny config/empty `__init__.py` files.
        """
        substantive_hashes = {
            hashlib.sha256(txt.strip().encode("utf-8")).hexdigest()
            for txt in file_contents.values()
            if len(txt.strip()) >= 64
        }
        if not substantive_hashes:
            return DedupDecision(True, None, None)

        for prev_repo, prev_set in self._repo_file_hashes.items():
            inter = len(substantive_hashes & prev_set)
            union = len(substantive_hashes | prev_set)
            overlap = inter / float(max(1, min(len(substantive_hashes), len(prev_set))))
            jaccard = inter / float(max(1, union))
            if overlap >= self.repo_overlap_threshold:
                self.stats["rejected_repo_overlap"] += 1
                return DedupDecision(
                    False,
                    "REPO_FORK_OVERLAP",
                    prev_repo,
                    round(jaccard, 4),
                )

        self._repo_file_hashes[repo_id] = substantive_hashes
        return DedupDecision(True, None, None)

    def filter_repeated_snippets(self, text: str) -> str:
        """
        Level 4: Suppress verbatim non-idiomatic 6-line boilerplate blocks repeated excessively
        across documents while exempting standard language idioms/imports/entrypoints.
        """
        lines = text.splitlines()
        if len(lines) < 10:
            return text

        out_lines: list[str] = []
        i = 0
        window = 6
        while i < len(lines):
            if i + window <= len(lines):
                block = lines[i : i + window]
                non_trivial = [ln for ln in block if len(ln.strip()) >= 12]
                is_exempt = any(
                    any(pat.search(ln) for pat in _IDIOMATIC_LINE_EXEMPTIONS)
                    for ln in block
                )
                if len(non_trivial) == window and not is_exempt:
                    block_hash = hashlib.sha256(
                        "\n".join(ln.strip() for ln in block).encode("utf-8")
                    ).hexdigest()
                    self._block_seen_counts[block_hash] += 1
                    if self._block_seen_counts[block_hash] > self.max_non_idiomatic_block_repeats:
                        self.stats["suppressed_repeated_snippets"] += 1
                        i += window
                        continue
            out_lines.append(lines[i])
            i += 1
        return "\n".join(out_lines)

    def check_and_register_document(
        self,
        doc_id: str,
        text: str,
        *,
        is_code: bool = True,
        language: str | None = None,
    ) -> DedupDecision:
        """
        Run Levels 1, 2, and 3 on a document and index it if unique.
        """
        self.stats["total_seen"] += 1

        # Level 1: Exact SHA-256
        exact_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if exact_sha in self._exact_hashes:
            self.stats["rejected_exact"] += 1
            return DedupDecision(False, "EXACT_SHA256_DUPLICATE", self._exact_hashes[exact_sha], 1.0)

        # Level 3: Code structural clone check (whitespace & comment invariant)
        struct_sha: str | None = None
        if is_code and self.enable_code_structural_dedup:
            norm_code = normalize_code_structure(text, language=language)
            if len(norm_code) >= 48:
                struct_sha = hashlib.sha256(norm_code.encode("utf-8")).hexdigest()
                if struct_sha in self._structural_code_hashes:
                    self.stats["rejected_code_structural"] += 1
                    return DedupDecision(
                        False,
                        "CODE_STRUCTURAL_DUPLICATE",
                        self._structural_code_hashes[struct_sha],
                        1.0,
                    )

        # Level 2: MinHash + LSH + exact Jaccard verification
        shingles = extract_word_shingles(text, k=5)
        sig = self._compute_minhash_signature(shingles)
        candidate_ids: set[str] = set()
        band_keys: list[bytes] = []

        for band_idx in range(self.lsh_bands):
            start = band_idx * self.rows_per_band
            band_slice = sig[start : start + self.rows_per_band]
            key = hashlib.blake2b(
                ",".join(str(x) for x in band_slice).encode("ascii"), digest_size=12
            ).digest()
            band_keys.append(key)
            for cand in self._lsh_buckets[band_idx].get(key, ()):
                candidate_ids.add(cand)

        for cand_id in candidate_ids:
            cand_shingles = self._doc_shingles[cand_id]
            union_sz = len(shingles | cand_shingles)
            jaccard = len(shingles & cand_shingles) / float(max(1, union_sz))
            if jaccard >= self.jaccard_threshold:
                self.stats["rejected_minhash_near_dup"] += 1
                return DedupDecision(
                    False,
                    "MINHASH_NEAR_DUPLICATE",
                    cand_id,
                    round(jaccard, 4),
                )

        # Document passed all checks -> register in all indices
        self._exact_hashes[exact_sha] = doc_id
        if struct_sha is not None:
            self._structural_code_hashes[struct_sha] = doc_id
        self._doc_shingles[doc_id] = shingles
        for band_idx, key in enumerate(band_keys):
            self._lsh_buckets[band_idx][key].append(doc_id)

        self.stats["kept_unique"] += 1
        return DedupDecision(True, None, None, None)
