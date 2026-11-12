# Rootism — Commercial Licensing, Provenance & Decontamination Protocol

## 1. Software Dependency Licensing Audit

Rootism itself is proprietary (`LICENSE`: *Copyright (c) 2026 Dubnium (V0idplayy-777). All Rights Reserved.*). Every third-party runtime dependency in `pyproject.toml` has been audited to ensure commercial compatibility without viral copyleft obligations:

| Package | SPDX Licence | Commercial Distribution Compatible |
|---|---|---|
| `torch` | `BSD-3-Clause` | Yes (retain copyright notice) |
| `numpy` | `BSD-3-Clause` | Yes (retain copyright notice) |
| `pydantic` | `MIT` | Yes (retain copyright notice) |
| `pyyaml` | `MIT` | Yes (retain copyright notice) |
| `safetensors` | `Apache-2.0` | Yes (retain license & NOTICE) |
| `psutil` | `BSD-3-Clause` | Yes (retain copyright notice) |
| `fastapi` / `uvicorn` *(Stage 2)* | `MIT` / `BSD-3-Clause` | Yes |
| `pytest` *(dev/CI)* | `MIT` | Yes |

Zero `GPL`, `AGPL`, `LGPL`, `SSPL`, or `BUSL` libraries are depended upon.

---

## 2. Training Dataset Commercial-Use Policy (`rootism/data/licensing.py`)

### Approved SPDX Allowlist
- `MIT`, `Apache-2.0`, `BSD-2-Clause`, `BSD-3-Clause`, `0BSD`, `ISC`, `CC0-1.0`, `Unlicense`, `Python-2.0`, `Zlib`, `BSL-1.0`, `CC-BY-4.0` (with mandatory attribution ledger tracking in `THIRD_PARTY_DATA_NOTICES.md`), and `Rootism-Proprietary-Original`.

### Strictly Excluded / Quarantined Categories
- Viral copyleft: `GPL-*`, `AGPL-*`, `LGPL-*`, `CC-BY-SA-*` (e.g., raw Wikipedia / StackExchange dumps).
- Non-commercial or research-only: `CC-BY-NC-*`, academic "for non-commercial research only" corpora, and LLM-distilled datasets violating upstream provider Terms of Service.
- Source-available commercial restrictions: `SSPL-1.0`, `BUSL-1.1`, `Commons Clause`.
- Ambiguous / missing licenses: `UNLICENSED`, `NOASSERTION`, `UNKNOWN`, or unverified public git repositories lacking an explicit `LICENSE` file.

### Deep Per-File & Submodule Verification (`DeepFileLicenseVerifier`)
Repository-level metadata is **never** trusted blindly:
1. `.gitmodules` and vendored directories (`vendor/`, `third_party/`, `external/`, `node_modules/`, `submodules/`) are quarantined unless accompanied by an independent verified clearance record.
2. Every source file's header and footer are scanned for `SPDX-License-Identifier:` overrides and embedded GPL/AGPL/CC-BY-NC/research-only notices before admission into `.rshard` files.

---

## 3. Benchmark Decontamination Protocol (`rootism/data/contamination.py`)

Evaluation benchmark suites are stored exclusively outside training directories and indexed into `DecontaminationEngine` prior to sharding using three independent layers:
1. Exact canonical SHA-256 document/prompt/solution hashes.
2. Normalized `13`-word n-gram collision detection across prompts and docstrings.
3. Whitespace-stripped `50`-character strided code span detection across canonical reference solutions.

Any training candidate matching any layer is quarantined and logged in the decontamination audit summary.

---

## 4. Legal Review Disclaimer (Directive H)

Technical enforcement of SPDX filtering, file-header scanning, attribution notice generation (`THIRD_PARTY_DATA_NOTICES.md`), and provenance hashing substantially reduces licensing risk and provides complete auditability; however, automated engineering controls do not constitute formal legal advice. Before commercial release of trained Rootism weights, qualified legal counsel should review the final `provenance_registry.yaml` and `THIRD_PARTY_DATA_NOTICES.md` generated from the exact production training shards.
