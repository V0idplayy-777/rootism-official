"""Original, self-authored pilot corpus for tokenizer experiments and tests.

IMPORTANT: this is NOT the production training corpus. It exists so the
tokenizer pipeline, compression analysis and regression tests are fully
reproducible without shipping third-party source code. Everything below is
original text written for Rootism; the generated portion is deterministic
synthetic code used only to give the BPE trainer enough token variety to
produce meaningful statistics. See docs/TOKENIZER.md (Known limitations).
"""

from __future__ import annotations

import random
from typing import Sequence

from .evaluation import EvalSample

PYTHON = '''"""Rootism sample: bounded retry with typed errors."""
from __future__ import annotations
import asyncio, logging
from dataclasses import dataclass
from typing import Callable, Awaitable

@dataclass(frozen=True)
class RetryPolicy:
    attempts: int = 3
    backoff_seconds: float = 0.25

async def call_with_retry(fn: Callable[[], Awaitable[str]], policy: RetryPolicy) -> str:
    for attempt in range(1, policy.attempts + 1):
        try:
            return await fn()
        except (TimeoutError, ConnectionError) as exc:
            logging.warning("attempt %d failed: %s", attempt, exc)
            if attempt == policy.attempts:
                raise
            await asyncio.sleep(policy.backoff_seconds * 2 ** (attempt - 1))
    raise AssertionError("unreachable")
'''

JAVASCRIPT = '''// Debounce with cancellation, ES2022.
export function debounce(fn, delayMs = 250) {
  let timer = null;
  const wrapped = (...args) => {
    if (timer !== null) clearTimeout(timer);
    timer = setTimeout(() => { timer = null; fn(...args); }, delayMs);
  };
  wrapped.cancel = () => { if (timer !== null) clearTimeout(timer); timer = null; };
  return wrapped;
}
const status = { ok: true, code: 200, payload: { items: [1, 2, 3] } };
console.log(`status=${status.code} => ${status.ok ? "ready" : "failed"}`);
'''

TYPESCRIPT = '''type Result<T, E = Error> = { ok: true; value: T } | { ok: false; error: E };
interface Repository<T extends { id: string }> {
  find(id: string): Promise<Result<T>>;
  upsert(entity: T): Promise<void>;
}
export class MemoryRepository<T extends { id: string }> implements Repository<T> {
  #store = new Map<string, T>();
  async find(id: string): Promise<Result<T>> {
    const value = this.#store.get(id);
    return value ? { ok: true, value } : { ok: false, error: new Error(`missing ${id}`) };
  }
  async upsert(entity: T): Promise<void> { this.#store.set(entity.id, entity); }
}
'''

HTML = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Rootism</title>
<link rel="stylesheet" href="/static/app.css"></head>
<body data-theme="dark"><main id="app" class="layout">
  <section aria-label="chat"><h1>Rootism</h1><p>Ask about your code.</p></section>
  <script type="module" src="/static/app.js" defer></script>
</main></body></html>
'''

CSS = ''':root { --bg: #0b0e14; --fg: #e6edf3; --accent: oklch(0.72 0.14 250); }
@media (prefers-color-scheme: light) { :root { --bg: #ffffff; --fg: #101418; } }
.layout { display: grid; grid-template-columns: minmax(220px, 18rem) 1fr; gap: 1rem; }
button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
'''

SQL = '''-- Rolling 7-day revenue by customer.
WITH daily AS (
  SELECT customer_id, DATE_TRUNC('day', created_at) AS day, SUM(amount_cents) / 100.0 AS revenue
  FROM payments WHERE status = 'settled' AND created_at >= CURRENT_DATE - INTERVAL '30 days'
  GROUP BY 1, 2
)
SELECT customer_id, day, revenue,
       AVG(revenue) OVER (PARTITION BY customer_id ORDER BY day ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS ma7
FROM daily ORDER BY customer_id, day;
'''

C = '''#include <stdio.h>
#include <stdlib.h>
typedef struct { size_t len; unsigned char *data; } Buffer;
static Buffer *buffer_new(size_t n) {
  Buffer *b = malloc(sizeof(*b));
  if (!b) return NULL;
  b->data = calloc(n, sizeof(unsigned char));
  if (!b->data) { free(b); return NULL; }
  b->len = n; return b;
}
int main(int argc, char **argv) { printf("argc=%d\\n", argc); return EXIT_SUCCESS; }
'''

CPP = '''#include <algorithm>
#include <memory>
#include <vector>
template <typename T, typename Compare = std::less<T>>
class SortedView {
public:
  explicit SortedView(std::vector<T> values, Compare cmp = {})
      : values_(std::move(values)), cmp_(cmp) { std::sort(values_.begin(), values_.end(), cmp_); }
  auto begin() const noexcept { return values_.cbegin(); }
  auto end() const noexcept { return values_.cend(); }
private:
  std::vector<T> values_; Compare cmp_;
};
'''

RUST = '''use std::collections::HashMap;
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token<'a> { pub text: &'a str, pub offset: usize }
pub fn index_tokens(input: &str) -> HashMap<&str, Vec<usize>> {
    let mut index: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, word) in input.split_whitespace().enumerate() {
        index.entry(word).or_default().push(i);
    }
    index
}
'''

GO = '''package worker
import (
    "context"
    "errors"
    "sync"
)
type Job func(context.Context) error
func Run(ctx context.Context, jobs <-chan Job, parallelism int) error {
    var wg sync.WaitGroup
    errs := make(chan error, parallelism)
    for i := 0; i < parallelism; i++ {
        wg.Add(1)
        go func() { defer wg.Done(); for job := range jobs { if err := job(ctx); err != nil { errs <- err } } }()
    }
    wg.Wait()
    close(errs)
    return errors.Join()
}
'''

SECONDARY = {
    "java": '''public final class RateLimiter {\n    private final long nanosPerToken;\n    private long next = System.nanoTime();\n    public RateLimiter(double tokensPerSecond) { this.nanosPerToken = (long) (1e9 / tokensPerSecond); }\n    public synchronized boolean tryAcquire() { long now = System.nanoTime(); if (now < next) return false; next = now + nanosPerToken; return true; }\n}\n''',
    "csharp": '''using System;\nusing System.Collections.Generic;\npublic static class EnumerableExtensions {\n  public static IEnumerable<T> WhereNotNull<T>(this IEnumerable<T?> source) where T : class {\n    foreach (var item in source) { if (item is not null) yield return item; }\n  }\n}\n''',
    "php": '''<?php\ndeclare(strict_types=1);\nfunction slugify(string $title): string {\n    $normalized = preg_replace('/[^a-z0-9]+/i', '-', strtolower($title));\n    return trim((string) $normalized, '-');\n}\necho slugify("Rootism 3B Model");\n''',
    "bash": '''#!/usr/bin/env bash\nset -euo pipefail\nROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\nfor file in "$ROOT"/src/**/*.py; do\n  [[ -f "$file" ]] || continue\n  printf '%s: %s lines\\n' "$file" "$(wc -l < "$file")"\ndone\n''',
    "powershell": '''param([string]$Path = ".")\nGet-ChildItem -Path $Path -Recurse -Filter *.ps1 | ForEach-Object {\n  $hash = (Get-FileHash -Algorithm SHA256 $_.FullName).Hash\n  [PSCustomObject]@{ Name = $_.Name; Sha256 = $hash.Substring(0, 12) }\n} | Sort-Object Name\n''',
    "lua": '''local M = {}\nfunction M.memoize(fn)\n  local cache = setmetatable({}, { __mode = "kv" })\n  return function(...)\n    local key = table.concat({...}, "\\0")\n    if cache[key] == nil then cache[key] = fn(...) end\n    return cache[key]\n  end\nend\nreturn M\n''',
    "kotlin": '''data class Page(val number: Int, val size: Int) { val offset: Int get() = (number - 1) * size }\nfun <T> List<T>.paginate(page: Page): List<T> = drop(page.offset).take(page.size)\nfun main() { println((1..100).toList().paginate(Page(2, 10))) }\n''',
    "swift": '''import Foundation\nstruct Cache<Key: Hashable, Value> {\n    private var storage: [Key: Value] = [:]\n    mutating func insert(_ value: Value, forKey key: Key) { storage[key] = value }\n    func value(forKey key: Key) -> Value? { storage[key] }\n}\nlet queue = DispatchQueue(label: "rootism.io", attributes: .concurrent)\n''',
    "ruby": '''# frozen_string_literal: true\nmodule Rootism\n  class Registry\n    def initialize = @entries = {}\n    def register(name, &block) = @entries[name.to_sym] = block\n    def fetch(name) = @entries.fetch(name.to_sym) { raise KeyError, "unknown: #{name}" }\n  end\nend\n''',
}

UNICODE_AND_PATHS = '''Paths: C:\\Users\\dev\\project\\src\\main.rs and /home/user/rootism/src/model/attention.py
Unicode identifiers: café_π = 3.14159; Δt = 1e-6; 日本語の変数 = "こんにちは"; emoji 🚀 ✅ ⚙️
Operators: => === !== <=> :: -> <<= ??= |>> |> << |> &&= ||= ~> 
String: "he said \\"hello\\" and 'goodbye'"; raw = r"C:\\temp\\new"; template = `${a + b}px`
Comment: // TODO(ünïcode): refactor before 2027-01-01
'''

NL_DOC = '''Rootism is a decoder-only transformer trained from randomly initialised weights.
Pre-normalisation with RMSNorm keeps activations stable in bfloat16. Grouped-query
attention reduces the KV cache by a factor of three relative to multi-head attention:
eight key/value heads serve twenty-four query heads. Rotary embeddings encode relative
position; the initial training context is 32,768 tokens. Sequence packing concatenates
multiple documents into one row while a document-identity mask prevents cross-document
attention. The loss is summed over non-padding tokens and divided by the global token
count, so gradient accumulation is exactly equivalent to a single large batch.
'''

NL_TUTORIAL = '''To add a new tool to the coding agent, first define its JSON schema and a handler
function. Validate every argument at the boundary; never pass untrusted strings to a
shell. Register the handler in the tool registry, then add a unit test that asserts
the handler rejects malformed arguments. Run the test suite before enabling the tool
in production. If a command can delete files or reach the network, require explicit
user confirmation and record the decision in the audit log.
'''

BASE_DOCUMENTS: list[tuple[str, str, str]] = [
    ("python", "code", PYTHON),
    ("javascript", "code", JAVASCRIPT),
    ("typescript", "code", TYPESCRIPT),
    ("html", "code", HTML),
    ("css", "code", CSS),
    ("sql", "code", SQL),
    ("c", "code", C),
    ("cpp", "code", CPP),
    ("rust", "code", RUST),
    ("go", "code", GO),
    *((lang, "code", text) for lang, text in SECONDARY.items()),
    ("mixed", "code", UNICODE_AND_PATHS),
    ("english", "nl", NL_DOC),
    ("english", "nl", NL_TUTORIAL),
]

WORDS = [
    "account", "adapter", "aggregate", "allocator", "archive", "attribute", "batch", "binding",
    "buffer", "cache", "catalog", "chunk", "client", "cluster", "codec", "component", "config",
    "context", "cursor", "dataset", "decoder", "delta", "descriptor", "dispatcher", "document",
    "encoder", "endpoint", "entity", "event", "executor", "factory", "feature", "fieldset",
    "filter", "fragment", "gateway", "handler", "header", "index", "iterator", "journal",
    "kernel", "layout", "listener", "manifest", "mapper", "message", "metadata", "middleware",
    "module", "namespace", "network", "observer", "operation", "packet", "parameter", "partition",
    "payload", "pipeline", "pointer", "policy", "processor", "project", "provider", "queue",
    "reader", "record", "registry", "renderer", "repository", "request", "resolver", "resource",
    "response", "result", "router", "runtime", "scheduler", "schema", "segment", "serializer",
    "session", "shard", "signal", "snapshot", "socket", "source", "spec", "statement", "storage",
    "stream", "symbol", "synchronizer", "table", "template", "thread", "token", "transaction",
    "transport", "tree", "validator", "vector", "viewport", "volume", "watcher", "worker", "writer",
]


def _generated_document(rng: random.Random, index: int) -> str:
    """Deterministic, original, syntactically plausible code/doc text."""
    a, b, c = rng.sample(WORDS, 3)
    kind = index % 4
    number = rng.randint(0, 9_999)
    if kind == 0:
        return (
            f"def {a}_{b}({c}_list: list[int], limit: int = {number}) -> dict[str, int]:\n"
            f"    \"\"\"Aggregate {a} values for the {b} pipeline (sample {index}).\"\"\"\n"
            f"    totals: dict[str, int] = {{}}\n"
            f"    for position, value in enumerate({c}_list[:limit]):\n"
            f"        key = f\"{a}:{{position % 7}}\"\n"
            f"        totals[key] = totals.get(key, 0) + value * {rng.randint(2, 9)}\n"
            f"    return sorted(totals.items())  # type: ignore[return-value]\n"
        )
    if kind == 1:
        return (
            f"export interface {a.title()}{b.title()}Options {{ {c}Id: string; retries?: number; }}\n"
            f"export async function create{a.title()}(options: {a.title()}{b.title()}Options): Promise<void> {{\n"
            f"  const endpoint = `/api/v2/{a}/${{options.{c}Id}}`;\n"
            f"  const response = await fetch(endpoint, {{ method: 'POST', headers: {{ 'X-Trace': '{number}' }} }});\n"
            f"  if (!response.ok) throw new Error(`${{endpoint}} failed with ${{response.status}}`);\n"
            f"}}\n"
        )
    if kind == 2:
        return (
            f"// {a.title()} {b} notes, revision {number}\n"
            f"The {c} resolver validates the {a} manifest before the {b} worker starts.\n"
            f"When a {c} shard is unavailable, the scheduler retries with exponential backoff\n"
            f"and records the failure in the {b} journal. Do not bypass the {a} validator.\n"
        )
    return (
        f"SELECT {a}_id, COUNT(*) AS {b}_count, MAX({c}_at) AS last_seen\n"
        f"FROM analytics.{a}_events\n"
        f"WHERE {b}_status IN ('queued', 'running') AND tenant_id = {number}\n"
        f"GROUP BY {a}_id HAVING COUNT(*) > {rng.randint(1, 20)} ORDER BY last_seen DESC;\n"
    )


def build_pilot_corpus(*, seed: int = 20261112, generated_documents: int = 400) -> list[str]:
    """Return base documents plus deterministic synthetic documents."""
    rng = random.Random(seed)
    docs = [text for _, _, text in BASE_DOCUMENTS]
    docs.extend(_generated_document(rng, i) for i in range(generated_documents))
    return docs


def pilot_eval_samples() -> Sequence[EvalSample]:
    return [EvalSample(lang, kind, text) for lang, kind, text in BASE_DOCUMENTS]
