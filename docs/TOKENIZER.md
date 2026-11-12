# Rootism Tokenizer

**Algorithm ID:** `rootism-byte-bpe-v1` · **Vocabulary:** 49,152 slots ·
**Special tokens:** 49 (IDs 0–48) · **Base byte symbols:** 256 (IDs 49–304) ·
**Learned merges:** up to 48,847 · **Normalisation:** identity (byte-exact)

The tokenizer is designed and implemented for Rootism. It does not use, load or
adapt any pretrained tokenizer, vocabulary, merge table or sentencepiece model.
`tests/test_tokenizer.py::test_no_pretrained_tokenizer_library_is_used` enforces
that no `transformers`, `tokenizers` or `sentencepiece` import appears in the
package.

## 1. Algorithm

1. **Pre-tokenisation** (`pretokenizer.py`) — a hand-written scanner, not a regex:
   * newline runs (`\n`, `\r\n`) as their own chunks,
   * horizontal whitespace runs (spaces, tabs, Unicode spaces) as their own
     chunks — **indentation is never folded into a word**,
   * identifiers (Unicode-aware: `isalpha`, `Lm/Lt/Nl`, ZWNJ/ZWJ) kept whole,
   * numbers including hex/`_`/`.`/exponent signs,
   * punctuation/operator runs kept together so `=>`, `===`, `::`, `->`, `<<=`,
     `??=`, `|>` can become single learned tokens,
   * oversized chunks split at UTF-8 character boundaries (`max_pretoken_bytes`).
   * Invariant: `"".join(pretokenize(text)) == text` (tested on every sample).
2. **Byte alphabet** — each byte maps to a printable Unicode codepoint
   (GPT-2-style reversible mapping), so token strings are always valid Unicode
   while every possible byte sequence remains representable.
3. **BPE merge learning** (`bpe.py`) — frequency-weighted adjacent-pair merging
   with a pair→word inverted index and a lazy max-heap; deterministic tie-break
   `(count desc, token strings asc)`; `min_pair_frequency` gate; SQLite spill for
   pre-token counting so the corpus need not fit in RAM.
4. **Encoding** — per pre-token, apply merges in learned rank order (all
   occurrences of the best-ranked pair at once, matching training), with an LRU
   cache.
5. **Decoding** — map token strings back to bytes, concatenate, UTF-8 decode.

Training is reproducible: two runs over the same corpus produce the same
`corpus_sha256`, merge list and `tokenizer_id` (tested).

## 2. Special tokens (requirement 25)

IDs are permanent. Normative semantics live in
`src/rootism/tokenizer/special_tokens.py` (the `semantics` field of each
`SpecialToken`).

| IDs | Tokens | Kind |
|---|---|---|
| 0–4 | `<pad>`, `<bos>`, `<eos>`, `<unk>`, `<sep>` | padding / sequence boundaries / fallback |
| 5–7 | `<fim_prefix>`, `<fim_middle>`, `<fim_suffix>` | fill-in-the-middle (PSM order) |
| 8–9 | `<code>`, `</code>` | code span boundary |
| 10–13 | `<file>`, `</file>`, `<file_path>`, `</file_path>` | workspace file payload + path |
| 14–17 | `<diff>`, `</diff>`, `<workspace>`, `</workspace>` | diff + workspace context |
| 18–23 | `<system>`, `</system>`, `<user>`, `</user>`, `<assistant>`, `</assistant>` | role boundaries |
| 24–25 | `<instruction>`, `</instruction>` | instruction boundary |
| 26–29 | `<reasoning>`, `</reasoning>`, `<task>`, `</task>` | **user-visible** structured reasoning summary + task metadata. This is generated explanatory text, not a raw private deliberation trace |
| 30–35 | `<plan>`, `</plan>`, `<observation>`, `</observation>`, `<agent>`, `</agent>` | agent communication |
| 36–43 | `<tool_call>`, `</tool_call>`, `<tool_name>`, `</tool_name>`, `<tool_args>`, `</tool_args>`, `<tool_result>`, `</tool_result>` | structured tool calls (JSON payload validated by the agent in Stage 6) |
| 44–47 | `<test_output>`, `</test_output>`, `<error>`, `</error>` | captured output + structured errors |
| 48 | `<mask>` | reserved for auxiliary masked objectives |

**Security decision:** `encode(text)` does *not* interpret special-token strings
found in raw text. A repository file containing `<tool_call>` cannot inject a
tool call. Structured input builders must opt in via `allowed_special="all"` or
an explicit list. Tested in
`test_special_token_injection_is_blocked_by_default`.

`encode_fim(prefix, suffix)` emits `<fim_prefix> prefix <fim_suffix> suffix
<fim_middle>` (PSM), ready for code-infilling training objectives.

## 3. Vocabulary-size analysis (requirement 23)

Executed: `python scripts/rootism_vocab_sweep.py --documents 12000`
(pilot corpus: 12,022 original documents, 4,045,604 bytes). Output saved to
`artifacts/vocab_sweep.json` (gitignored, regenerable).

| Requested | Achieved | Bytes/token | Identifier fragmentation |
|---:|---:|---:|---:|
| 512 | 512 | 1.3700 | 0.884 |
| 1,024 | 1,024 | 1.5479 | 0.762 |
| 2,048 | 2,048 | 1.5621 | 0.752 |
| 4,096 | 4,096 | 1.8034 | 0.530 |
| 8,192 | 8,192 | 1.9259 | 0.434 |
| 16,384 | 15,948 | 2.5126 | 0.059 |

Power-law fit on achieved sizes:
`log(bytes/token) = -0.7023 + 0.1585 · log(vocab)`, `R² = 0.9134`.

| Candidate | Predicted bytes/token | Emb+head params at ~3B | Depth at ~3B |
|---:|---:|---:|---:|
| 32,768 | 2.5754 | 201,326,592 | 28 |
| 49,152 | 2.7464 | 301,989,888 | 27 |
| 65,536 | 2.8745 | 402,653,184 | 26 |

**Decision: 49,152.** Predicted 64K→48K difference is 4.67% compression, but
64K costs 100,663,296 extra embedding/head parameters and two transformer layers
at the same ~3B budget. 32K is below the evaluated 48K–64K design range and
fragments code identifiers more. 48K is the balanced choice.

**Honest limitation:** the pilot corpus is 4 MB of *original synthetic* code, far
too small to learn 48K real merges (it saturates around 15,948). Values at
32K/48K/64K are **extrapolations, not measurements**, and the power-law fit is
only a weak guide (R² 0.913). `scripts/rootism_vocab_sweep.py` must be re-run
against the licensed production corpus before the tokenizer is frozen; once
training begins, changing the vocabulary invalidates all checkpoints.

## 4. Evaluation (requirement 27)

`evaluate_tokenizer()` reports: bytes/token, tokens per document, NL
bytes/token, identifier mean tokens + fragmentation rate, Unicode round-trip
rate, indentation round-trip rate, special-token injection safety, per-language
bytes/token, cross-language standard deviation and max−min spread. On the
22-sample pilot evaluation set (all 19 languages + Unicode/paths + NL docs) the
session test tokenizer achieved 100% Unicode and indentation round-trip and
passed the injection-safety check. These are **tokenizer measurements on a pilot
artifact**, not model-quality results.

## 5. Versioning (requirement 28)

`tokenizer_id = "tok-" + sha256(canonical JSON of {schema, algorithm, config,
special tokens, vocab, merges, corpus_sha256})[:16]`. `save()` records the
fingerprint; `from_file()` recomputes it and refuses to load on mismatch (tested
with deliberate tampering). Checkpoint manifests store `tokenizer_id`, and
`compatibility_report()` rejects a checkpoint/tokenizer mismatch.

## 6. Training a production tokenizer

```bash
export PYTHONPATH=src
.venv/bin/python -m rootism.cli.tokenizer train \
    --corpus /path/to/licensed/corpus --output artifacts/rootism-tokenizer.json \
    --vocab-size 49152
.venv/bin/python -m rootism.cli.tokenizer verify --tokenizer artifacts/rootism-tokenizer.json
.venv/bin/python -m rootism.cli.tokenizer evaluate --tokenizer artifacts/rootism-tokenizer.json --pilot
```

`train` streams UTF-8 files (or directories of `*.txt`) and never requires the
whole corpus in RAM. **The caller is responsible for the corpus satisfying the
commercial-use licensing policy in `LICENSING.md`**; the tokenizer pipeline does
not verify licences (that is Stage 2's dataset registry).

## 7. Known limitations

* Merge learning is single-machine with `max_unique_pretokens=2,000,000` kept in
  the frequency table. Rarer pre-tokens remain encodable via byte symbols, but
  production-scale corpora need sharded counting (Stage 2).
* No NFKC normalisation by design; visually identical Unicode sequences are
  distinct tokens.
* Long punctuation runs are split at 128 bytes, which can split very long string
  literals (round-trip remains exact).
* The shipped pilot tokenizer is a **test artifact**, not the production
  tokenizer; it is not committed to Git.
