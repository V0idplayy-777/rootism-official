(() => {
  "use strict";

  const STORAGE_FILES_KEY = "rootism_studio_local_files_v1";
  const STORAGE_ROLLBACK_KEY = "rootism_studio_local_rollback_v1";

  const JS_TS_SYM_RE = /^\s*(?:export\s+)?(?:async\s+)?(?:function|class|interface|type|enum|const)\s+([A-Za-z_$][\w$]*)/gm;
  const PY_SYM_RE = /^(?:async\s+)?(?:def|class)\s+([A-Za-z_]\w*)/gm;
  const RUST_SYM_RE = /^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:fn|struct|enum|trait|mod)\s+([A-Za-z_]\w*)/gm;
  const GO_SYM_RE = /^\s*(?:func(?:\s+\([^)]+\))?|type)\s+([A-Za-z_]\w*)/gm;

  function loadFiles() {
    try {
      const raw = localStorage.getItem(STORAGE_FILES_KEY);
      if (raw) {
        const parsed = JSON.parse(raw);
        if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
          return parsed;
        }
      }
    } catch (_e) {
      // ignore corrupted storage
    }
    return {};
  }

  function saveFiles(filesMap) {
    try {
      localStorage.setItem(STORAGE_FILES_KEY, JSON.stringify(filesMap));
    } catch (_e) {
      // ignore storage quota issues
    }
  }

  function loadRollback() {
    try {
      const raw = localStorage.getItem(STORAGE_ROLLBACK_KEY);
      if (raw) {
        const parsed = JSON.parse(raw);
        if (Array.isArray(parsed)) return parsed;
      }
    } catch (_e) {
      // ignore
    }
    return [];
  }

  function saveRollback(stack) {
    try {
      localStorage.setItem(STORAGE_ROLLBACK_KEY, JSON.stringify(stack.slice(-30)));
    } catch (_e) {
      // ignore
    }
  }

  function normalizePath(rawPath) {
    return String(rawPath || "")
      .trim()
      .replace(/\\/g, "/")
      .replace(/^\/+/, "")
      .replace(/\/+/g, "/");
  }

  function extractSymbols(filePath, content) {
    const ext = (filePath.split(".").pop() || "").toLowerCase();
    let rx = null;
    if (ext === "py") rx = PY_SYM_RE;
    else if (["js", "jsx", "mjs", "cjs", "ts", "tsx"].includes(ext)) rx = JS_TS_SYM_RE;
    else if (ext === "rs") rx = RUST_SYM_RE;
    else if (ext === "go") rx = GO_SYM_RE;
    if (!rx) return [];
    rx.lastIndex = 0;
    const found = [];
    let m;
    while ((m = rx.exec(content)) !== null) {
      if (m[1] && !found.includes(m[1])) {
        found.push(m[1]);
      }
      if (found.length >= 40) break;
    }
    return found;
  }

  function buildOverview(filesMap) {
    const paths = Object.keys(filesMap).sort();
    const symbolToFiles = {};
    const indexedFiles = [];
    const languageCounts = {};

    paths.forEach((p) => {
      const content = String(filesMap[p] ?? "");
      const ext = (p.split(".").pop() || "txt").toLowerCase();
      languageCounts[ext] = (languageCounts[ext] || 0) + 1;
      const syms = extractSymbols(p, content);
      syms.forEach((s) => {
        if (!symbolToFiles[s]) symbolToFiles[s] = [];
        symbolToFiles[s].push(p);
      });
      indexedFiles.push({
        path: p,
        language: ext,
        line_count: content ? content.split("\n").length : 0,
        symbols: syms,
        imports: [],
      });
    });

    return {
      workspace_root: "browser://local-workspace",
      total_files: paths.length,
      language_file_counts: languageCounts,
      detected_manifests: paths.filter((p) =>
        ["pyproject.toml", "package.json", "Cargo.toml", "go.mod", "requirements.txt"].includes(
          p.split("/").pop()
        )
      ),
      declared_dependencies: {},
      indexed_files: indexedFiles,
      symbol_to_files: symbolToFiles,
    };
  }

  function computeSimpleDiff(relPath, before, after) {
    const oldLines = (before || "").split("\n");
    const newLines = (after || "").split("\n");
    if (before === after) return "";
    const out = [`--- a/${relPath}`, `+++ b/${relPath}`];
    const maxLen = Math.max(oldLines.length, newLines.length);
    out.push(`@@ -1,${oldLines.length} +1,${newLines.length} @@`);
    for (let i = 0; i < maxLen; i++) {
      const o = i < oldLines.length ? oldLines[i] : undefined;
      const n = i < newLines.length ? newLines[i] : undefined;
      if (o === n) {
        out.push(` ${o}`);
      } else {
        if (o !== undefined) out.push(`-${o}`);
        if (n !== undefined) out.push(`+${n}`);
      }
    }
    return out.join("\n");
  }

  function parseBody(options) {
    if (!options || !options.body) return {};
    if (typeof options.body === "string") {
      try {
        return JSON.parse(options.body);
      } catch (_e) {
        return {};
      }
    }
    return options.body;
  }

  const RootismLocal = {
    active: false,

    chatNotice() {
      return (
        "No model weights are currently connected (running in local workspace mode).\n\n" +
        "You can create, edit, diff, undo, and organize files in the Workspace right now. " +
        "Once your Rootism model server is running with trained weights (or initialized verification weights), " +
        "start `rootism serve --checkpoint-dir <path-to-step_XXXXXXXX> --tokenizer-dir <path-to-tokenizer>` " +
        "or set the **Backend URL** in **Settings** to connect live model inference."
      );
    },

    resetWorkspace() {
      saveFiles({});
      saveRollback([]);
    },

    async handle(relativePath, options = {}) {
      const method = String(options.method || "GET").toUpperCase();
      const cleanUrl = String(relativePath || "").replace(/^\./, "");
      const [pathname, queryString] = cleanUrl.split("?");
      const params = new URLSearchParams(queryString || "");
      const body = parseBody(options);

      if (pathname === "/api/v1/health") {
        return {
          status: "local_only",
          ready: true,
          uptime_seconds: 0,
          in_flight_generations: 0,
          max_concurrent_generations: 1,
          auth_required: false,
          runtime: { device: "local-browser" },
        };
      }

      if (pathname === "/api/v1/workspace/files" && method === "GET") {
        const filesMap = loadFiles();
        const files = Object.keys(filesMap).sort();
        const rollback = loadRollback();
        return {
          workspace_root: "browser://local-workspace",
          files,
          overview: buildOverview(filesMap),
          rollback_depth: rollback.length,
        };
      }

      if (pathname === "/api/v1/workspace/file" && method === "GET") {
        const reqPath = normalizePath(params.get("path") || "");
        const filesMap = loadFiles();
        if (!reqPath || !Object.prototype.hasOwnProperty.call(filesMap, reqPath)) {
          return {
            path: reqPath,
            exists: false,
            total_lines: 0,
            start_line: 1,
            end_line: 0,
            content: "",
            numbered_content: "",
          };
        }
        const content = String(filesMap[reqPath] ?? "");
        const lines = content.split("\n");
        return {
          path: reqPath,
          exists: true,
          total_lines: lines.length,
          start_line: 1,
          end_line: lines.length,
          content,
          numbered_content: lines.map((l, idx) => `${String(idx + 1).padStart(4, " ")} | ${l}`).join("\n"),
        };
      }

      if (pathname === "/api/v1/workspace/file" && method === "POST") {
        const rel = normalizePath(body.path || "untitled.py");
        const content = String(body.content ?? "");
        const filesMap = loadFiles();
        const existed = Object.prototype.hasOwnProperty.call(filesMap, rel);
        const prev = existed ? filesMap[rel] : null;
        const diff = computeSimpleDiff(rel, prev || "", content);

        filesMap[rel] = content;
        saveFiles(filesMap);

        const rollback = loadRollback();
        rollback.push({
          relative_path: rel,
          existed_before: existed,
          previous_content: prev,
          new_content: content,
        });
        saveRollback(rollback);

        return {
          tool_name: "create_file",
          ok: true,
          requires_confirmation: false,
          confirmation_token: null,
          summary: `${existed ? "Updated" : "Created"} ${rel} (${content.length} chars).`,
          payload: { path: rel, unified_diff: diff },
        };
      }

      if (pathname === "/api/v1/workspace/revert" && method === "POST") {
        const rollback = loadRollback();
        if (rollback.length === 0) {
          return {
            tool_name: "revert_last_edit",
            ok: true,
            requires_confirmation: false,
            confirmation_token: null,
            summary: "Nothing to undo yet.",
            payload: { reverted_path: null, reverted: false },
          };
        }
        const entry = rollback.pop();
        saveRollback(rollback);
        const filesMap = loadFiles();
        if (entry.existed_before && entry.previous_content !== null) {
          filesMap[entry.relative_path] = entry.previous_content;
        } else {
          delete filesMap[entry.relative_path];
        }
        saveFiles(filesMap);
        return {
          tool_name: "revert_last_edit",
          ok: true,
          requires_confirmation: false,
          confirmation_token: null,
          summary: `Reverted ${entry.relative_path}.`,
          payload: { reverted_path: entry.relative_path, reverted: true, existed_before: entry.existed_before },
        };
      }

      if (pathname === "/api/v1/code/complete" && method === "POST") {
        return {
          id: `local-fim-${Date.now()}`,
          completion: "",
          finish_reason: "no_backend",
          standby: true,
          note: "No model backend connected — set a Backend URL in Settings to enable code completion.",
          usage: { prompt_tokens: 0, completion_tokens: 0 },
          telemetry: { ttft_ms: 0, total_latency_ms: 0, tokens_per_second: 0 },
        };
      }

      if (pathname === "/api/v1/chat/completions" && method === "POST") {
        return {
          id: `local-chat-${Date.now()}`,
          object: "rootism.chat.completion",
          model: "rootism-local-offline",
          checkpoint_step: 0,
          reasoning_mode: body.reasoning_mode || "structured",
          message: {
            role: "assistant",
            content: this.chatNotice(),
            reasoning_summary: null,
          },
          finish_reason: "stop",
          usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
          telemetry: {
            ttft_ms: 0,
            total_latency_ms: 0,
            tokens_per_second: 0,
            kv_cache_bytes: 0,
            context_window: {
              original_prompt_tokens: 0,
              effective_prompt_tokens: 0,
              max_context_limit: 65536,
              reserved_generation_tokens: 96,
              was_truncated: false,
              used_extended_yarn_window: false,
            },
          },
        };
      }

      if (pathname === "/api/v1/agent/tool" && method === "POST") {
        const toolName = body.tool_name || "inspect_project";
        const filesMap = loadFiles();
        const ov = buildOverview(filesMap);
        if (toolName === "inspect_project") {
          return {
            tool_name: "inspect_project",
            ok: true,
            requires_confirmation: false,
            confirmation_token: null,
            summary: `Indexed ${ov.total_files} local workspace file(s) (${Object.keys(ov.symbol_to_files).length} symbols).`,
            payload: ov,
          };
        }
        if (toolName === "run_tests") {
          return {
            tool_name: "run_tests",
            ok: true,
            requires_confirmation: false,
            confirmation_token: null,
            summary: "No backend connected — start `rootism serve` to run sandboxed pytest suites.",
            payload: {},
          };
        }
        return {
          tool_name: toolName,
          ok: true,
          requires_confirmation: false,
          confirmation_token: null,
          summary: `Local mode handled ${toolName}.`,
          payload: {},
        };
      }

      if (pathname === "/api/v1/agent/run" && method === "POST") {
        const filesMap = loadFiles();
        const ov = buildOverview(filesMap);
        return {
          goal: body.goal || "",
          status: "completed",
          stop_reason:
            "Local workspace inspected. Connect a Rootism backend (`rootism serve`) to execute sandboxed tests and model passes.",
          iterations_used: 1,
          tests_verified: false,
          modified_files: [],
          unified_diffs: [],
          pending_confirmation_token: null,
          pending_action: null,
          traces: [
            {
              step_index: 1,
              phase: "inspect",
              rationale: `Scanned local browser workspace (${ov.total_files} file(s), ${Object.keys(ov.symbol_to_files).length} symbol(s)).`,
              tool_name: "inspect_project",
              tool_result: {
                ok: true,
                requires_confirmation: false,
                summary: `${ov.total_files} file(s) in workspace.`,
              },
            },
          ],
        };
      }

      if (pathname === "/api/v1/agent/confirm" && method === "POST") {
        return {
          approved: Boolean(body.approve),
          confirmation_token: body.confirmation_token || "",
          message: body.approve ? "Action confirmed." : "Action rejected.",
          executed_result: { summary: "Completed in local mode." },
        };
      }

      if (pathname.startsWith("/api/v1/requests/") && pathname.endsWith("/cancel")) {
        return { cancelled: true };
      }

      return { ok: true };
    },
  };

  window.RootismLocal = RootismLocal;
})();
