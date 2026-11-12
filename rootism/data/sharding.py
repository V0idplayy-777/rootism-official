"""
Scalable Memory-Mapped Binary Shard Storage & Streaming Engine (Step 46).

Designed so a 650-billion-token corpus (~1.30 TB in uint16) is preprocessed, sharded,
verified, and streamed with O(1) host RAM overhead.

Binary Formats:
- `.rshard`:
    - 16-byte header: magic `b"RSHARD01"` (8B) + version `uint32` (4B) + token_dtype_bytes `uint32` (4B)
    - Contiguous `uint16` (for vocab <= 65,536) little-endian token stream
- `.ridx`:
    - 16-byte header: magic `b"RIDX0001"` (8B) + doc_count `uint64` (8B)
    - Fixed 24-byte records per document: `<Q I H H I I`
      `(token_offset_u64, token_length_u32, domain_id_u16, language_id_u16, crc32_u32, flags_u32)`
- `.rmanifest.json`:
    - Cryptographically binds shard files to `tokenizer_fingerprint`, `provenance_registry_sha256`,
      `unique_source_tokens`, `deduplicated_tokens`, and per-shard SHA-256 checksums.
"""

from __future__ import annotations

import hashlib
import json
import struct
import zlib
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rootism.tokenizer.bpe import RootismTokenizer, TokenizerCompatibilityError

SHARD_MAGIC = b"RSHARD01"
INDEX_MAGIC = b"RIDX0001"
SHARD_HEADER_STRUCT = struct.Struct("<8sII")
INDEX_HEADER_STRUCT = struct.Struct("<8sQ")
INDEX_RECORD_STRUCT = struct.Struct("<QIHHI I")


@dataclass(frozen=True)
class RawCorpusDocument:
    doc_id: str
    text: str
    dataset_id: str
    domain_id: int = 0
    language_id: int = 0


@dataclass(frozen=True)
class ShardDescriptor:
    shard_index: int
    shard_filename: str
    index_filename: str
    document_count: int
    token_count: int
    shard_sha256: str
    index_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


_WORKER_TOKENIZER: RootismTokenizer | None = None


def _init_worker(tok_path: str) -> None:
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = RootismTokenizer.load(tok_path)


def _tokenize_worker_batch(
    args: tuple[str, list[tuple[str, int, int, str]], bool],
) -> list[tuple[list[int], int, int, str]]:
    global _WORKER_TOKENIZER
    tok_path, docs, allow_special = args
    if _WORKER_TOKENIZER is None:
        _WORKER_TOKENIZER = RootismTokenizer.load(tok_path)
    tok = _WORKER_TOKENIZER
    results: list[tuple[list[int], int, int, str]] = []
    for text, dom_id, lang_id, dataset_id in docs:
        tids = tok.encode(text, add_bos=True, add_eos=True, allow_special=allow_special)
        results.append((tids, dom_id, lang_id, dataset_id))
    return results


class RootismShardWriter:
    """
    Streaming binary `.rshard` / `.ridx` writer with atomic shard finalization
    and automatic rollover at `max_tokens_per_shard`.
    """

    def __init__(
        self,
        output_dir: str | Path,
        tokenizer: RootismTokenizer,
        provenance_registry_sha256: str,
        *,
        dataset_name: str = "rootism-corpus",
        max_tokens_per_shard: int = 50_000_000,
    ) -> None:
        if tokenizer.vocab_size > 65_536:
            raise ValueError("uint16 shard writer requires vocab_size <= 65,536")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = tokenizer
        self.provenance_registry_sha256 = provenance_registry_sha256
        self.dataset_name = dataset_name
        self.max_tokens_per_shard = max_tokens_per_shard

        self.shards: list[ShardDescriptor] = []
        self.total_documents: int = 0
        self.total_tokens: int = 0
        # Per-dataset attribution: dataset_id -> (doc_count, token_count)
        self.dataset_attribution: dict[str, dict[str, int]] = {}

        self._cur_shard_idx: int = 0
        self._cur_tokens_written: int = 0
        self._cur_doc_records: list[tuple[int, int, int, int, int, int]] = []
        self._cur_shard_tmp_path: Path | None = None
        self._cur_shard_fh: Any = None

    def _open_new_shard(self) -> None:
        tmp_path = self.output_dir / f"shard_{self._cur_shard_idx:06d}.rshard.tmp"
        fh = tmp_path.open("wb")
        fh.write(SHARD_HEADER_STRUCT.pack(SHARD_MAGIC, 1, 2))
        self._cur_shard_tmp_path = tmp_path
        self._cur_shard_fh = fh
        self._cur_tokens_written = 0
        self._cur_doc_records = []

    def _flush_current_shard(self) -> None:
        if self._cur_shard_fh is None or self._cur_shard_tmp_path is None:
            return
        self._cur_shard_fh.flush()
        self._cur_shard_fh.close()
        self._cur_shard_fh = None

        if self._cur_tokens_written == 0:
            self._cur_shard_tmp_path.unlink(missing_ok=True)
            self._cur_shard_tmp_path = None
            return

        shard_name = f"shard_{self._cur_shard_idx:06d}.rshard"
        idx_name = f"shard_{self._cur_shard_idx:06d}.ridx"
        final_shard = self.output_dir / shard_name
        final_idx = self.output_dir / idx_name
        tmp_idx = self.output_dir / (idx_name + ".tmp")

        self._cur_shard_tmp_path.replace(final_shard)
        self._cur_shard_tmp_path = None

        with tmp_idx.open("wb") as ifh:
            ifh.write(INDEX_HEADER_STRUCT.pack(INDEX_MAGIC, len(self._cur_doc_records)))
            for rec in self._cur_doc_records:
                ifh.write(INDEX_RECORD_STRUCT.pack(*rec))
        tmp_idx.replace(final_idx)

        self.shards.append(
            ShardDescriptor(
                shard_index=self._cur_shard_idx,
                shard_filename=shard_name,
                index_filename=idx_name,
                document_count=len(self._cur_doc_records),
                token_count=self._cur_tokens_written,
                shard_sha256=_sha256_file(final_shard),
                index_sha256=_sha256_file(final_idx),
            )
        )
        self._cur_shard_idx += 1
        self._cur_tokens_written = 0
        self._cur_doc_records = []

    def add_token_sequence(
        self,
        token_ids: list[int] | np.ndarray,
        *,
        domain_id: int = 0,
        language_id: int = 0,
        flags: int = 0,
        dataset_id: str = "",
    ) -> None:
        arr = np.asarray(token_ids, dtype=np.int64)
        if arr.size == 0:
            return
        # Validate every token ID against the uint16 range before conversion
        # to prevent silent wrapping of negative or oversized IDs.
        if arr.size > 0:
            min_val = int(arr.min())
            max_val = int(arr.max())
            if min_val < 0 or min_val > 65_535:
                raise ValueError(
                    f"Token ID {min_val} out of uint16 range [0, 65535] "
                    f"(vocab_size={self.tokenizer.vocab_size})"
                )
            if max_val > 65_535:
                raise ValueError(
                    f"Token ID {max_val} out of uint16 range [0, 65535] "
                    f"(vocab_size={self.tokenizer.vocab_size})"
                )
        arr = arr.astype("<u2")
        if self._cur_shard_fh is None:
            self._open_new_shard()
        elif self._cur_tokens_written > 0 and (
            self._cur_tokens_written + int(arr.size) > self.max_tokens_per_shard
        ):
            self._flush_current_shard()
            self._open_new_shard()

        raw_bytes = arr.tobytes()
        crc = zlib.crc32(raw_bytes) & 0xFFFFFFFF
        offset = self._cur_tokens_written
        length = int(arr.size)

        assert self._cur_shard_fh is not None
        self._cur_shard_fh.write(raw_bytes)
        self._cur_doc_records.append((offset, length, domain_id, language_id, crc, flags))
        self._cur_tokens_written += length
        self.total_documents += 1
        self.total_tokens += length

        # Track per-dataset attribution for provenance auditing
        if dataset_id:
            attr = self.dataset_attribution.setdefault(dataset_id, {"documents": 0, "tokens": 0})
            attr["documents"] += 1
            attr["tokens"] += length

    def add_document_text(
        self,
        text: str,
        *,
        domain_id: int = 0,
        language_id: int = 0,
        allow_special: bool = False,
    ) -> int:
        ids = self.tokenizer.encode(text, add_bos=True, add_eos=True, allow_special=allow_special)
        self.add_token_sequence(ids, domain_id=domain_id, language_id=language_id)
        return len(ids)

    def finalize(self, *, unique_source_tokens: int | None = None) -> Path:
        self._flush_current_shard()
        manifest_path = self.output_dir / "dataset.rmanifest.json"
        src_toks = unique_source_tokens if unique_source_tokens is not None else self.total_tokens
        manifest = {
            "format": "rootism-shard-manifest-v1",
            "dataset_name": self.dataset_name,
            "tokenizer_fingerprint_sha256": self.tokenizer.fingerprint,
            "vocab_size": self.tokenizer.vocab_size,
            "provenance_registry_sha256": self.provenance_registry_sha256,
            "total_shards": len(self.shards),
            "total_documents": self.total_documents,
            "unique_source_tokens": int(src_toks),
            "deduplicated_tokens": int(self.total_tokens),
            "dataset_attribution": {
                ds_id: {"documents": att["documents"], "tokens": att["tokens"]}
                for ds_id, att in self.dataset_attribution.items()
            },
            "shards": [s.to_dict() for s in self.shards],
        }
        tmp = manifest_path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
        tmp.replace(manifest_path)
        return manifest_path


def build_shards_parallel(
    documents: Iterable[RawCorpusDocument],
    output_dir: str | Path,
    tokenizer_path: str | Path,
    provenance_registry_sha256: str,
    *,
    num_workers: int = 2,
    batch_size: int = 64,
    max_tokens_per_shard: int = 50_000_000,
    allow_special: bool = False,
) -> Path:
    """Parallel multi-process corpus tokenization & sharding pipeline.

    Streams documents through bounded batches so the full raw corpus is never
    held in memory simultaneously. Retains ``dataset_id`` provenance through
    the pipeline for per-source attribution.
    """
    tok = RootismTokenizer.load(tokenizer_path)
    writer = RootismShardWriter(
        output_dir=output_dir,
        tokenizer=tok,
        provenance_registry_sha256=provenance_registry_sha256,
        max_tokens_per_shard=max_tokens_per_shard,
    )

    tok_path_str = str(tokenizer_path)

    def _batch_generator() -> Iterator[tuple[str, list[tuple[str, int, int, str]], bool]]:
        """Yield bounded batches without accumulating the full corpus."""
        cur: list[tuple[str, int, int, str]] = []
        for doc in documents:
            cur.append((doc.text, doc.domain_id, doc.language_id, doc.dataset_id))
            if len(cur) >= batch_size:
                yield (tok_path_str, cur, allow_special)
                cur = []
        if cur:
            yield (tok_path_str, cur, allow_special)

    if num_workers <= 1:
        for batch in _batch_generator():
            for tids, dom_id, lang_id, ds_id in _tokenize_worker_batch(batch):
                writer.add_token_sequence(
                    tids, domain_id=dom_id, language_id=lang_id, dataset_id=ds_id
                )
    else:
        with ProcessPoolExecutor(
            max_workers=num_workers, initializer=_init_worker, initargs=(tok_path_str,)
        ) as pool:
            for encoded_batch in pool.map(_tokenize_worker_batch, _batch_generator()):
                for tids, dom_id, lang_id, ds_id in encoded_batch:
                    writer.add_token_sequence(
                        tids, domain_id=dom_id, language_id=lang_id, dataset_id=ds_id
                    )

    return writer.finalize()


class RootismStreamingShardDataset:
    """
    Zero-RAM-bloat `mmap` reader over `.rshard` / `.ridx` binary datasets.
    Verifies tokenizer fingerprint compatibility and CRC32 integrity on access.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        expected_tokenizer_fingerprint: str | None = None,
        verify_checksums_on_open: bool = False,
        required_provenance_registry_sha256: str | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.base_dir = self.manifest_path.parent
        with self.manifest_path.open("r", encoding="utf-8") as fh:
            self.manifest = json.load(fh)

        if self.manifest.get("format") != "rootism-shard-manifest-v1":
            raise ValueError(f"Unsupported shard manifest format in {self.manifest_path}")

        manifest_tok_fp = self.manifest["tokenizer_fingerprint_sha256"]
        if (
            expected_tokenizer_fingerprint is not None
            and manifest_tok_fp != expected_tokenizer_fingerprint
        ):
            raise TokenizerCompatibilityError(
                f"Shard manifest locked to tokenizer {manifest_tok_fp}, "
                f"expected {expected_tokenizer_fingerprint}"
            )

        if not self.manifest.get("provenance_registry_sha256"):
            raise ValueError("Shard manifest missing required commercial provenance_registry_sha256")

        # Enforce provenance registry hash equality when a required hash is supplied
        if required_provenance_registry_sha256 is not None:
            manifest_reg = self.manifest.get("provenance_registry_sha256", "")
            if manifest_reg != required_provenance_registry_sha256:
                raise RuntimeError(
                    f"Provenance registry mismatch: manifest has {manifest_reg!r}, "
                    f"expected {required_provenance_registry_sha256!r}"
                )

        # Validate unique_source_tokens consistency
        unique_src = int(self.manifest.get("unique_source_tokens", self.manifest["deduplicated_tokens"]))
        dedup_toks = int(self.manifest["deduplicated_tokens"])
        if unique_src < dedup_toks:
            raise ValueError(
                f"unique_source_tokens ({unique_src}) cannot be less than "
                f"deduplicated_tokens ({dedup_toks})"
            )

        self.total_tokens: int = dedup_toks
        self.total_documents: int = int(self.manifest["total_documents"])
        self.unique_source_tokens: int = unique_src

        self._shard_mmaps: list[np.memmap] = []
        self._shard_indices: list[list[tuple[int, int, int, int, int, int]]] = []

        for s_desc in self.manifest["shards"]:
            # Validate that shard paths stay within the dataset directory
            s_path = (self.base_dir / s_desc["shard_filename"]).resolve()
            i_path = (self.base_dir / s_desc["index_filename"]).resolve()
            try:
                s_path.relative_to(self.base_dir.resolve())
            except ValueError:
                raise ValueError(
                    f"Shard filename {s_desc['shard_filename']!r} resolves outside "
                    f"dataset directory {self.base_dir}"
                )
            try:
                i_path.relative_to(self.base_dir.resolve())
            except ValueError:
                raise ValueError(
                    f"Index filename {s_desc['index_filename']!r} resolves outside "
                    f"dataset directory {self.base_dir}"
                )
            if verify_checksums_on_open:
                if _sha256_file(s_path) != s_desc["shard_sha256"]:
                    raise RuntimeError(f"Corrupted shard file {s_path}: SHA-256 mismatch")
                if _sha256_file(i_path) != s_desc["index_sha256"]:
                    raise RuntimeError(f"Corrupted index file {i_path}: SHA-256 mismatch")

            with s_path.open("rb") as sfh:
                magic, ver, dtype_bytes = SHARD_HEADER_STRUCT.unpack(sfh.read(16))
                if magic != SHARD_MAGIC or ver != 1 or dtype_bytes != 2:
                    raise RuntimeError(f"Invalid .rshard binary header in {s_path}")

            mm = np.memmap(
                s_path,
                dtype="<u2",
                mode="r",
                offset=SHARD_HEADER_STRUCT.size,
                shape=(int(s_desc["token_count"]),),
            )
            self._shard_mmaps.append(mm)

            records: list[tuple[int, int, int, int, int, int]] = []
            with i_path.open("rb") as ifh:
                imagic, doc_cnt = INDEX_HEADER_STRUCT.unpack(ifh.read(INDEX_HEADER_STRUCT.size))
                if imagic != INDEX_MAGIC:
                    raise RuntimeError(f"Invalid .ridx binary header in {i_path}")
                for _ in range(doc_cnt):
                    rec = INDEX_RECORD_STRUCT.unpack(ifh.read(INDEX_RECORD_STRUCT.size))
                    records.append(rec)
            self._shard_indices.append(records)

    def get_document(self, shard_idx: int, doc_idx: int, *, verify_crc: bool = True) -> dict[str, Any]:
        mm = self._shard_mmaps[shard_idx]
        offset, length, dom_id, lang_id, crc_expected, flags = self._shard_indices[shard_idx][doc_idx]
        view = mm[offset : offset + length]
        if verify_crc:
            actual_crc = zlib.crc32(view.tobytes()) & 0xFFFFFFFF
            if actual_crc != crc_expected:
                raise RuntimeError(
                    f"CRC32 integrity error in shard {shard_idx} doc {doc_idx}: "
                    f"{actual_crc:#x} != {crc_expected:#x}"
                )
        return {
            "token_ids": np.array(view, dtype=np.int64, copy=True),
            "domain_id": dom_id,
            "language_id": lang_id,
            "flags": flags,
        }

    def iter_documents(
        self,
        *,
        seed: int | None = None,
        epoch: int = 0,
        rank: int = 0,
        world_size: int = 1,
    ) -> Iterator[dict[str, Any]]:
        """
        Deterministically stream documents across shards (optionally shuffled per epoch
        and partitioned across distributed `rank` / `world_size` workers).
        """
        coords: list[tuple[int, int]] = []
        for s_idx, recs in enumerate(self._shard_indices):
            for d_idx in range(len(recs)):
                coords.append((s_idx, d_idx))

        if seed is not None:
            rng = np.random.default_rng(seed + epoch * 1_000_003)
            order = rng.permutation(len(coords))
            coords = [coords[int(i)] for i in order]

        for global_pos in range(rank, len(coords), world_size):
            s_idx, d_idx = coords[global_pos]
            doc = self.get_document(s_idx, d_idx, verify_crc=False)
            doc["epoch"] = epoch
            yield doc
