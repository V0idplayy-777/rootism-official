(() => {
  "use strict";

  const STORE_KEY = "rootism_local_workspace_v1";
  const MAX_FILE_BYTES = 500000;
  const MAX_UNDO = 50;
  const SEED = {
  "pyproject.toml": "[project]\nname = \"workspace-sample\"\nversion = \"0.1.0\"\nrequires-python = \">=3.11\"\n\n[tool.pytest.ini_options]\npythonpath = [\".\"]\n",
  "src/__init__.py": "",
  "src/metrics.py": "\"\"\"Latency percentile & token throughput utilities.\"\"\"\n\nfrom __future__ import annotations\n\n\ndef compute_p95_latency(samples_ms: list[float]) -> float:\n    \"\"\"Compute nearest-rank P95 latency in milliseconds.\"\"\"\n    if not samples_ms:\n        raise ValueError(\"samples_ms must not be empty\")\n    ordered = sorted(samples_ms)\n    idx = max(0, min(len(ordered) - 1, round(0.95 * (len(ordered) - 1))))\n    return float(ordered[idx])\n\n\ndef tokens_per_second(token_count: int, elapsed_sec: float) -> float:\n    if elapsed_sec <= 0:\n        raise ValueError(\"elapsed_sec must be positive\")\n    return round(float(token_count) / float(elapsed_sec), 3)\n",
  "src/ui_helpers.ts": "export interface StreamStats {\n  promptTokens: number;\n  completionTokens: number;\n  ttftMs: number;\n}\n\nexport function formatThroughput(tokens: number, ms: number): string {\n  if (ms <= 0) return '0.0 tok/s';\n  return `${((tokens * 1000) / ms).toFixed(1)} tok/s`;\n}\n",
  "tests/test_metrics.py": "from src.metrics import compute_p95_latency, tokens_per_second\n\n\ndef test_metrics_calculations() -> None:\n    assert compute_p95_latency([10.0, 20.0, 30.0, 40.0, 100.0]) == 100.0\n    assert tokens_per_second(250, 2.0) == 125.0\n"
};

  function httpError(status, message) {
    return Object.assign(new Error(message), { status });
  }

  function loadStore() {
    try {
      const raw = localStorage.getItem(STORE_KEY);
      if (raw) {
        const parsed = JSON.parse(raw);
        if (parsed && parsed.files && Array.isArray(parsed.undo)) return parsed;
      }
    } catch (_e) {
      return { files: { ...SEED }, undo: [] };
    }
    return { files: { ...SEED }, undo: [] };
  }

  let store = loadStore();

  function persist() {
    try {
      localStorage.setItem(STORE_KEY, JSON.stringify(store));
    } catch (_e) {
      throw httpError(507, "Browser storage is full or blocked, so the change was not kept.");
    }
  }

  function cleanPath(raw) {
    const p = String(raw || "").trim().replace(/\\/g, "/").replace(/^\.\//, "");
    if (!p || p.startsWith("/") || p.split("/").some((seg) => seg === ".." || seg === "")) {
      throw httpError(400, `Invalid workspace path: ${raw}`);
    }
    return p;
  }

  function buildSymbolIndex() {
    const map = {};
    const add = (name, path) => {
      if (!map[name]) map[name] = [];
      if (!map[name].includes(path)) map[name].push(path);
    };
    Object.entries(store.files).forEach(([path, content]) => {
      let re = null;
      if (path.endsWith(".py")) {
        re = /^[ \t]*(?:async[ \t]+)?(?:def|class)[ \t]+([A-Za-z_]\w*)/gm;
      } else if (/\.(ts|tsx|js|jsx)$/.test(path)) {
        re = /^[ \t]*(?:export[ \t]+)?(?:async[ \t]+)?(?:function|class|interface|type|const)[ \t]+([A-Za-z_$][\w$]*)/gm;
      }
      if (!re) return;
      let m;
      while ((m = re.exec(content)) !== null) add(m[1], path);
    });
    return map;
  }

  function unifiedDiff(path, before, after) {
    const toLines = (text) => {
      const parts = text.split("\n");
      if (parts.length && parts[parts.length - 1] === "") parts.pop();
      return parts;
    };
    const a = before === null ? [] : toLines(before);
    const b = toLines(after);
    const n = a.length;
    const m = b.length;
    const ops = [];

    if (n * m > 4000000) {
      a.forEach((line) => ops.push(["-", line]));
      b.forEach((line) => ops.push(["+", line]));
    } else {
      const dp = Array.from({ length: n + 1 }, () => new Uint32Array(m + 1));
      for (let i = n - 1; i >= 0; i--) {
        for (let j = m - 1; j >= 0; j--) {
          dp[i][j] = a[i] === b[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
        }
      }
      let i = 0;
      let j = 0;
      while (i < n && j < m) {
        if (a[i] === b[j]) {
          ops.push([" ", a[i]]);
          i++;
          j++;
        } else if (dp[i + 1][j] >= dp[i][j + 1]) {
          ops.push(["-", a[i++]]);
        } else {
          ops.push(["+", b[j++]]);
        }
      }
      while (i < n) ops.push(["-", a[i++]]);
      while (j < m) ops.push(["+", b[j++]]);
    }

    const changed = [];
    ops.forEach((op, k) => {
      if (op[0] !== " ") changed.push(k);
    });
    if (!changed.length) return "";

    const aPos = [];
    const bPos = [];
    let ai = 1;
    let bi = 1;
    ops.forEach((op) => {
      aPos.push(ai);
      bPos.push(bi);
      if (op[0] !== "+") ai++;
      if (op[0] !== "-") bi++;
    });

    const ctx = 3;
    const ranges = [];
    let start = Math.max(0, changed[0] - ctx);
    let end = Math.min(ops.length, changed[0] + ctx + 1);
    changed.slice(1).forEach((k) => {
      if (k - ctx <= end) {
        end = Math.min(ops.length, k + ctx + 1);
      } else {
        ranges.push([start, end]);
        start = Math.max(0, k - ctx);
        end = Math.min(ops.length, k + ctx + 1);
      }
    });
    ranges.push([start, end]);

    const out = [before === null ? "--- /dev/null" : `--- a/${path}`, `+++ b/${path}`];
    ranges.forEach(([s, e]) => {
      const slice = ops.slice(s, e);
      const aCount = slice.filter((o) => o[0] !== "+").length;
      const bCount = slice.filter((o) => o[0] !== "-").length;
      out.push(`@@ -${aCount ? aPos[s] : aPos[s] - 1},${aCount} +${bCount ? bPos[s] : bPos[s] - 1},${bCount} @@`);
      slice.forEach((o) => out.push(o[0] + o[1]));
    });
    return out.join("\n");
  }

  function parseBody(options) {
    try {
      return options && options.body ? JSON.parse(options.body) : {};
    } catch (_e) {
      throw httpError(400, "Request body was not valid JSON.");
    }
  }

  const NO_BACKEND = "No backend is connected, so there is no model to run this. Set a Backend URL in Settings once one is hosted.";

  async function handle(relativePath, options = {}) {
    const url = new URL(relativePath, "http://local/");
    const match = /api\/v1\/(.*)$/.exec(url.pathname);
    const route = match ? match[1] : "";
    const method = String(options.method || "GET").toUpperCase();

    if (route === "workspace/files" && method === "GET") {
      return {
        workspace_root: "this browser",
        files: Object.keys(store.files).sort(),
        overview: { symbol_to_files: buildSymbolIndex() },
        rollback_depth: store.undo.length,
      };
    }

    if (route === "workspace/file" && method === "GET") {
      const path = cleanPath(url.searchParams.get("path"));
      if (!(path in store.files)) throw httpError(404, `No such file in the workspace: ${path}`);
      return { path, content: store.files[path] };
    }

    if (route === "workspace/file" && method === "POST") {
      const body = parseBody(options);
      const path = cleanPath(body.path);
      const content = String(body.content ?? "");
      if (content.length > MAX_FILE_BYTES) throw httpError(400, "File is too large for browser storage.");
      const before = path in store.files ? store.files[path] : null;
      if (before === content) {
        return { ok: true, summary: `No changes to ${path}`, payload: { path, unified_diff: "" } };
      }
      store.undo.push({ path, before });
      if (store.undo.length > MAX_UNDO) store.undo.shift();
      store.files[path] = content;
      persist();
      return { ok: true, summary: `Saved ${path}`, payload: { path, unified_diff: unifiedDiff(path, before, content) } };
    }

    if (route === "workspace/revert" && method === "POST") {
      const last = store.undo.pop();
      if (!last) throw httpError(400, "Nothing to undo.");
      if (last.before === null) {
        delete store.files[last.path];
      } else {
        store.files[last.path] = last.before;
      }
      persist();
      return { ok: true, summary: `Reverted ${last.path}`, payload: { reverted_path: last.path } };
    }

    if (route === "agent/tool" && method === "POST") {
      const body = parseBody(options);
      if (body.tool_name === "inspect_project") {
        const symbols = buildSymbolIndex();
        const fileCount = Object.keys(store.files).length;
        return {
          ok: true,
          summary: `${fileCount} files and ${Object.keys(symbols).length} symbols in the browser workspace`,
          payload: { files: Object.keys(store.files).sort(), symbol_to_files: symbols },
        };
      }
      if (body.tool_name === "run_tests") {
        return { ok: false, summary: "pytest can't run in the browser. Run the tests locally or in CI.", payload: {} };
      }
      return { ok: false, summary: `${body.tool_name || "That tool"} needs the Python backend.`, payload: {} };
    }

    if (route === "agent/run" && method === "POST") {
      const body = parseBody(options);
      return {
        status: "failed",
        goal: body.goal || "",
        stop_reason: "The agent needs a model and a Python sandbox. Neither is available without a backend.",
        traces: [],
      };
    }

    if (route === "agent/confirm" && method === "POST") {
      return { message: "Nothing is pending.", executed_result: null };
    }

    if (route === "code/complete" && method === "POST") {
      throw httpError(503, NO_BACKEND);
    }

    if (/^requests\/[^/]+\/cancel$/.test(route) && method === "POST") {
      return { ok: true };
    }

    throw httpError(404, `Not available without a backend: ${route || relativePath}`);
  }

  window.RootismLocal = {
    active: false,
    handle,
    chatNotice: () => NO_BACKEND,
  };
})();vvvvvvvvv
