# Licensing, Provenance and Commercial-Use Policy

> **This repository is NOT legally cleared for commercial model deployment.**
> The project licence is proprietary "all rights reserved"; no dataset registry
> has been populated; dependency licences have not been reviewed by counsel.
> This document states the policy and records what has been observed, not legal
> advice or a clearance claim.

## 1. Project licence

`LICENSE` (copyright 2026 Dubnium / V0idplayy-777) grants **no permission** to
use, copy, modify, merge, publish, distribute, sublicense or sell the software
without prior express written permission. Any commercial deployment, model-weight
distribution or sublicensing requires an explicit decision by the copyright
holder. Model weights produced from this codebase inherit whatever terms the
holder sets; none are set here.

## 2. Third-party dependencies (observed, not verified)

Installed in the Stage 1 environment (from `importlib.metadata`):

| Package | Version | Declared licence metadata |
|---|---|---|
| torch | 2.15.0+cu132 | no `License` field in wheel metadata (upstream project declares BSD-style terms; **verify**) |
| numpy | 2.4.6 | no `License` field in wheel metadata (upstream declares BSD-3-Clause; **verify**) |
| pytest | 9.2.0 | no `License` field in wheel metadata (upstream declares MIT; **verify**) |
| triton | 3.9.0 | `MIT` |
| pluggy | 1.7.0 | `MIT` |
| sympy / mpmath / Jinja2 | 1.14.0 / 1.3.0 / 3.1.6 | BSD (classifier) |
| `nvidia-cusparselt-cu13` | 0.8.1.22 | **`NVIDIA Proprietary Software`** |
| other `nvidia-*` runtime wheels | 13.x | licence metadata absent; **verify before redistribution** |

**Action required (Stage 8 / legal review):** pin every dependency, capture its
licence text and URL, and assess redistribution obligations. The proprietary
NVIDIA runtime wheels pulled in by the PyPI `torch` wheel are a concrete
commercial-deployment consideration, not a hypothetical one. Optional planned
dependencies (`flash-attn`, `fastapi`, `uvicorn`, `pydantic`) must be reviewed
when they are added.

## 3. Dataset policy (requirement 35 — ABSOLUTE)

1. **No external dataset has been downloaded, registered, processed or shipped
   in Stage 1.** The only corpus in the repository is `pilot_corpus.py`: original
   text authored for Rootism, used exclusively for tokenizer tests.
2. Every externally sourced dataset must have a **verified legal basis permitting
   commercial use**. Public accessibility is not permission.
3. Datasets with unknown, unclear, non-commercial, contractually restricted,
   unresolved-ownership or redistribution-prohibited terms are **excluded**, not
   silently included.
4. Nested repositories, submodules, vendored files and individual file headers
   must be checked; a parent repository licence is not automatically inherited.
5. Synthetic/original data must be evaluated for correctness and is not assumed
   high quality.
6. Benchmark/evaluation data is kept separate from training data; contamination
   detection is part of Stage 2/8.

## 4. Dataset registry schema (Stage 2 deliverable, currently EMPTY)

Each entry must record, at minimum (requirement 36):

```
dataset_id, name, publisher, original_source, download_location, exact_version,
licence, licence_url, commercial_use_permission, attribution_requirements,
redistribution_restrictions, collection_date, processing_history,
known_limitations, licence_verification_evidence, nested_component_checks,
sha256_of_source_manifest, excluded_reason (if excluded)
```

Format ID: `rootism-dataset-manifest-v1`, schema version 1, canonical JSON with a
`manifest_id` content hash. Checkpoint manifests will bind `manifest_id` so a
training run is reproducible to an exact data snapshot.

## 5. Attribution obligations

None are currently asserted because no external data is used. When Stage 2 adds
datasets, required attributions must be generated into a distributable
`ATTRIBUTIONS.md` and included in any release that contains derived artifacts.

## 6. Outstanding legal work

* Dependency licence audit + lockfile.
* Dataset licence verification with evidence for every source.
* Review of framework/example code provenance (React, Node.js, FastAPI, Django,
  Flask, Express, etc.) before any such examples enter the corpus.
* Model licence, output-ownership terms and third-party notice file.
* Security/privacy review of collected code (secrets, PII, licence headers).
