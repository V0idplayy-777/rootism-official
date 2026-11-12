/* Rootism Developer Platform Client Application (Steps 81–90)
   Strictly uses relative `/api/v1/*` URLs so it runs seamlessly in local & proxied preview hosts. */

(() => {
  "use strict";

  const STORAGE_SESSIONS_KEY = "rootism_studio_sessions_v1";
  const STORAGE_THEME_KEY = "rootism_studio_theme_v1";

  const state = {
    sessions: [],
    activeSessionId: null,
    activeFilePath: "src/metrics.py",
    activeFileContent: "",
    activeRequestId: null,
    abortController: null,
    modelInfo: null,
    pendingConfirmation: null,
  };

  // --- DOM Helpers ---
  const $ = (id) => document.getElementById(id);

  function authHeaders(extra = {}) {
    const headers = { "Content-Type": "application/json", ...extra };
    const apiKey = $("ctrl-api-key")?.value?.trim();
    if (apiKey) {
      headers["Authorization"] = `Bearer ${apiKey}`;
    }
    return headers;
  }

  async function apiFetch(relativePath, options = {}) {
    const resp = await fetch(relativePath, {
      ...options,
      headers: authHeaders(options.headers || {}),
    });
    if (!resp.ok) {
      let errMsg = `HTTP ${resp.status}`;
      try {
        const errData = await resp.json();
        errMsg = errData?.error?.message || errData?.detail || errMsg;
      } catch (_e) {
        // ignore non-json body
      }
      throw new Error(errMsg);
    }
    return resp.json();
  }

  function showAlert(msg) {
    $("global-alert-text").textContent = msg;
    $("global-alert").classList.remove("hidden");
  }

  function escapeHtml(raw) {
    return String(raw)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function highlightCode(escapedCode) {
    return escapedCode
      .replace(
        /\b(def|class|return|if|elif|else|for|while|import|from|try|except|finally|with|async|await|function|const|let|export|interface|type|struct|fn|impl|pub|SELECT|FROM|WHERE|ORDER|BY)\b/g,
        '<span class="tok-kw">$1</span>'
      )
      .replace(/\b(\d+(?:\.\d+)?)\b/g, '<span class="tok-num">$1</span>')
      .replace(/(#[^\n]*|\/\/[^\n]*)/g, '<span class="tok-cmt">$1</span>');
  }

  function renderRichContent(text) {
    if (!text) return '<span class="muted-small">(empty output)</span>';
    const fenceRegex = /```([a-zA-Z0-9_+#-]*)\n([\s\S]*?)```/g;
    let lastIndex = 0;
    let html = "";
    let match;

    while ((match = fenceRegex.exec(text)) !== null) {
      const prose = text.slice(lastIndex, match.index);
      if (prose) {
        html += `<div>${escapeHtml(prose)}</div>`;
      }
      const lang = match[1] || "code";
      const rawSnippet = match[2];
      const encodedSnippet = encodeURIComponent(rawSnippet);
      html += `
        <div class="code-fence">
          <div class="code-fence-header">
            <span>${escapeHtml(lang)}</span>
            <div>
              <button type="button" class="link-btn copy-code-btn" data-code="${encodedSnippet}">Copy</button>
              &nbsp;&bull;&nbsp;
              <button type="button" class="link-btn insert-code-btn" data-code="${encodedSnippet}">Send to Editor</button>
            </div>
          </div>
          <pre><code>${highlightCode(escapeHtml(rawSnippet))}</code></pre>
        </div>`;
      lastIndex = fenceRegex.lastIndex;
    }
    const tail = text.slice(lastIndex);
    if (tail) {
      html += `<div>${escapeHtml(tail)}</div>`;
    }
    return html;
  }

  // --- Theme & View Layout ---
  function initThemeAndTabs() {
    const savedTheme = localStorage.getItem(STORAGE_THEME_KEY) || "dark";
    setTheme(savedTheme);

    $("theme-toggle-btn").addEventListener("click", () => {
      const cur = document.documentElement.getAttribute("data-theme") || "dark";
      setTheme(cur === "dark" ? "light" : "dark");
    });

    $("sidebar-toggle-btn").addEventListener("click", () => {
      $("session-sidebar").classList.toggle("open");
    });

    $("global-alert-dismiss").addEventListener("click", () => {
      $("global-alert").classList.add("hidden");
    });

    document.querySelectorAll(".tab-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".tab-btn").forEach((b) => {
          b.classList.remove("active");
          b.setAttribute("aria-selected", "false");
        });
        btn.classList.add("active");
        btn.setAttribute("aria-selected", "true");
        const view = btn.dataset.view;
        const shell = $("main-workspace");
        shell.setAttribute("data-layout", view);

        const wsCard = $("subpanel-workspace");
        const agCard = $("subpanel-agent");
        const arCard = $("subpanel-architecture");

        if (view === "split") {
          wsCard.classList.remove("hidden");
          agCard.classList.remove("hidden");
          arCard.classList.add("hidden");
        } else if (view === "workspace") {
          wsCard.classList.remove("hidden");
          agCard.classList.add("hidden");
          arCard.classList.add("hidden");
        } else if (view === "agent") {
          wsCard.classList.add("hidden");
          agCard.classList.remove("hidden");
          arCard.classList.add("hidden");
        } else if (view === "architecture") {
          wsCard.classList.add("hidden");
          agCard.classList.add("hidden");
          arCard.classList.remove("hidden");
        }
      });
    });

    // Settings drawer
    const drawer = $("settings-drawer");
    $("settings-toggle-btn").addEventListener("click", () => {
      const hidden = drawer.classList.toggle("hidden");
      $("settings-toggle-btn").setAttribute("aria-expanded", String(!hidden));
    });
    $("close-settings-btn").addEventListener("click", () => {
      drawer.classList.add("hidden");
      $("settings-toggle-btn").setAttribute("aria-expanded", "false");
    });

    const sliderBindings = [
      ["ctrl-max-tokens", "val-max-tokens", (v) => v],
      ["ctrl-temperature", "val-temperature", (v) => Number(v).toFixed(2)],
      ["ctrl-top-p", "val-top-p", (v) => Number(v).toFixed(2)],
      ["ctrl-top-k", "val-top-k", (v) => v],
      ["ctrl-rep-pen", "val-rep-pen", (v) => Number(v).toFixed(2)],
    ];
    sliderBindings.forEach(([ctrlId, valId, fmt]) => {
      const el = $(ctrlId);
      el.addEventListener("input", () => {
        $(valId).textContent = fmt(el.value);
      });
    });
  }

  function setTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem(STORAGE_THEME_KEY, theme);
    $("theme-toggle-btn").textContent = theme === "dark" ? "Light Theme" : "Dark Theme";
  }

  // --- Session Persistence ---
  function loadSessions() {
    try {
      const raw = localStorage.getItem(STORAGE_SESSIONS_KEY);
      if (raw) {
        state.sessions = JSON.parse(raw);
      }
    } catch (_e) {
      state.sessions = [];
    }
    if (!Array.isArray(state.sessions) || state.sessions.length === 0) {
      createNewSession("Session 1 — Engineering Studio");
    } else {
      state.activeSessionId = state.sessions[0].id;
      renderSessionList();
      renderMessages();
    }
  }

  function saveSessions() {
    localStorage.setItem(STORAGE_SESSIONS_KEY, JSON.stringify(state.sessions));
  }

  function activeSession() {
    return state.sessions.find((s) => s.id === state.activeSessionId) || state.sessions[0];
  }

  function createNewSession(title) {
    const sess = {
      id: `sess-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
      title: title || `Session ${state.sessions.length + 1}`,
      createdAt: new Date().toISOString(),
      messages: [],
    };
    state.sessions.unshift(sess);
    state.activeSessionId = sess.id;
    saveSessions();
    renderSessionList();
    renderMessages();
  }

  function renderSessionList() {
    const ul = $("session-list");
    ul.innerHTML = "";
    state.sessions.forEach((sess) => {
      const li = document.createElement("li");
      li.className = `session-item ${sess.id === state.activeSessionId ? "active" : ""}`;
      li.innerHTML = `
        <span class="session-title">${escapeHtml(sess.title)}</span>
        <button type="button" class="link-btn del-sess-btn" title="Delete session" aria-label="Delete session">&times;</button>
      `;
      li.addEventListener("click", (ev) => {
        if (ev.target.closest(".del-sess-btn")) {
          ev.stopPropagation();
          state.sessions = state.sessions.filter((x) => x.id !== sess.id);
          if (state.sessions.length === 0) {
            createNewSession("Session 1");
          } else {
            state.activeSessionId = state.sessions[0].id;
            saveSessions();
            renderSessionList();
            renderMessages();
          }
          return;
        }
        state.activeSessionId = sess.id;
        renderSessionList();
        renderMessages();
      });
      ul.appendChild(li);
    });
  }

  // --- Chat Rendering & Streaming ---
  function renderMessages() {
    const container = $("chat-messages");
    const sess = activeSession();
    if (!sess || sess.messages.length === 0) {
      container.innerHTML = `
        <div class="msg-card">
          <div class="msg-header"><span class="msg-role">Rootism Developer Platform</span></div>
          <div class="msg-body">Welcome to <strong>Rootism</strong>.
&bull; Every response streams live from <code>RootismInferenceEngine</code> (zero external APIs or canned strings).
&bull; Select a reasoning mode above, inspect/edit files in the right-hand IDE, or run the sandboxed Coding Agent with pytest verification.</div>
        </div>`;
      return;
    }

    container.innerHTML = "";
    sess.messages.forEach((m, idx) => {
      const card = document.createElement("article");
      card.className = `msg-card ${m.role}`;
      const summaryBlock = m.reasoning_summary
        ? `<details class="reasoning-details" open>
             <summary>Reasoning Summary (${escapeHtml(m.reasoning_mode || "structured")})</summary>
             <div>${escapeHtml(m.reasoning_summary)}</div>
           </details>`
        : "";
      const metaBadge = m.telemetry
        ? `<span class="muted-small">${m.telemetry.tokens_per_second} tok/s &bull; TTFT ${m.telemetry.ttft_ms}ms</span>`
        : "";

      card.innerHTML = `
        <div class="msg-header">
          <span class="msg-role">${m.role === "user" ? "Developer" : "Rootism"}</span>
          <div class="msg-actions">
            ${metaBadge}
            ${m.role === "user" ? `<button type="button" class="link-btn edit-msg-btn" data-idx="${idx}">Edit</button>` : ""}
            <button type="button" class="link-btn copy-msg-btn" data-idx="${idx}">Copy</button>
          </div>
        </div>
        ${summaryBlock}
        <div class="msg-body">${m.role === "assistant" ? renderRichContent(m.content) : escapeHtml(m.content)}</div>
      `;
      container.appendChild(card);
    });
    container.scrollTop = container.scrollHeight;
  }

  async function streamChatCompletion(userPrompt) {
    const sess = activeSession();
    if (sess.messages.length === 0) {
      sess.title = userPrompt.slice(0, 36);
      renderSessionList();
    }

    sess.messages.push({ role: "user", content: userPrompt });
    const assistantMsg = {
      role: "assistant",
      content: "…",
      reasoning_summary: null,
      reasoning_mode: $("reasoning-mode-select").value,
      telemetry: null,
    };
    sess.messages.push(assistantMsg);
    saveSessions();
    renderMessages();

    const reqId = `ui-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
    state.activeRequestId = reqId;
    state.abortController = new AbortController();

    $("send-btn").disabled = true;
    $("stop-stream-btn").classList.remove("hidden");

    const sysPrompt = $("ctrl-system-prompt").value.trim();
    const apiMessages = [];
    if (sysPrompt) {
      apiMessages.push({ role: "system", content: sysPrompt });
    }
    sess.messages.slice(0, -1).forEach((m) => {
      apiMessages.push({ role: m.role, content: m.content });
    });

    let wsContext = null;
    if ($("attach-workspace-toggle").checked && state.activeFilePath && state.activeFileContent) {
      wsContext = `Active File: ${state.activeFilePath}\n${state.activeFileContent.slice(0, 8000)}`;
    }

    const payload = {
      request_id: reqId,
      messages: apiMessages,
      reasoning_mode: $("reasoning-mode-select").value,
      stream: true,
      max_tokens: Number($("ctrl-max-tokens").value),
      temperature: Number($("ctrl-temperature").value),
      top_p: Number($("ctrl-top-p").value),
      top_k: Number($("ctrl-top-k").value),
      repetition_penalty: Number($("ctrl-rep-pen").value),
      workspace_context: wsContext,
    };

    try {
      const resp = await fetch("./api/v1/chat/completions", {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify(payload),
        signal: state.abortController.signal,
      });

      if (!resp.ok) {
        const errBody = await resp.json().catch(() => ({}));
        throw new Error(errBody?.error?.message || `HTTP ${resp.status}`);
      }

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop() || "";

        for (const frame of parts) {
          const line = frame.trim();
          if (!line.startsWith("data:")) continue;
          const jsonStr = line.slice(5).trim();
          if (!jsonStr) continue;
          const ev = JSON.parse(jsonStr);

          if (ev.event === "meta" && ev.context_window) {
            updateContextBar(ev.context_window);
          } else if (ev.event === "delta") {
            assistantMsg.content = ev.sanitized_text_so_far || "";
            assistantMsg.reasoning_summary = ev.reasoning_summary_so_far || null;
            renderMessages();
          } else if (ev.event === "done") {
            assistantMsg.content = ev.content || assistantMsg.content || "(EOS reached)";
            assistantMsg.reasoning_summary = ev.reasoning_summary || null;
            assistantMsg.telemetry = ev.telemetry || null;
            if (ev.telemetry) {
              $("last-ttft-label").textContent = `TTFT: ${ev.telemetry.ttft_ms}ms`;
              $("last-tps-label").textContent = `Speed: ${ev.telemetry.tokens_per_second} tok/s`;
              if (ev.telemetry.context_window) {
                updateContextBar(ev.telemetry.context_window);
              }
            }
            saveSessions();
            renderMessages();
          } else if (ev.event === "error") {
            throw new Error(ev.message || "Streaming generation error");
          }
        }
      }
    } catch (err) {
      if (err.name === "AbortError") {
        assistantMsg.content = `${assistantMsg.content === "…" ? "" : assistantMsg.content}\n[Generation stopped by user]`.trim();
      } else {
        assistantMsg.content = `[Error: ${err.message}]`;
        showAlert(`Generation error: ${err.message}`);
      }
      saveSessions();
      renderMessages();
    } finally {
      state.activeRequestId = null;
      state.abortController = null;
      $("send-btn").disabled = false;
      $("stop-stream-btn").classList.add("hidden");
    }
  }

  async function cancelActiveGeneration() {
    const reqId = state.activeRequestId;
    if (state.abortController) {
      state.abortController.abort();
    }
    if (reqId) {
      try {
        await apiFetch(`./api/v1/requests/${encodeURIComponent(reqId)}/cancel`, { method: "POST" });
      } catch (_e) {
        // ignore if already finished
      }
    }
  }

  function updateContextBar(ctx) {
    const used = ctx.effective_prompt_tokens || 0;
    const lim = ctx.max_context_limit || 65536;
    const pct = Math.min(100, Math.max(2, (used / lim) * 100));
    $("ctx-usage-bar").style.width = `${pct.toFixed(1)}%`;
    $("ctx-limit-label").textContent = `${used.toLocaleString()} / ${lim.toLocaleString()}${ctx.was_truncated ? " (truncated)" : ""}`;
  }

  // --- Workspace Explorer, Editor, FIM & Unified Diffs ---
  function updateEditorGutter() {
    const lines = ($("code-editor").value || "").split("\n").length;
    const nums = [];
    for (let i = 1; i <= lines; i++) nums.push(i);
    $("editor-gutter").textContent = nums.join("\n");
  }

  async function refreshWorkspaceFiles() {
    try {
      const data = await apiFetch("./api/v1/workspace/files");
      const fileUl = $("workspace-file-list");
      fileUl.innerHTML = "";
      (data.files || []).forEach((fPath) => {
        const li = document.createElement("li");
        li.className = `file-entry ${fPath === state.activeFilePath ? "active" : ""}`;
        li.textContent = fPath;
        li.title = fPath;
        li.addEventListener("click", () => openWorkspaceFile(fPath));
        fileUl.appendChild(li);
      });

      const symUl = $("workspace-symbol-list");
      symUl.innerHTML = "";
      const symMap = data.overview?.symbol_to_files || {};
      Object.entries(symMap).slice(0, 40).forEach(([sym, paths]) => {
        const li = document.createElement("li");
        li.className = "symbol-entry";
        li.textContent = `${sym} (${paths[0]})`;
        li.title = `${sym} -> ${paths.join(", ")}`;
        li.addEventListener("click", () => openWorkspaceFile(paths[0]));
        symUl.appendChild(li);
      });
    } catch (err) {
      showAlert(`Failed to list workspace files: ${err.message}`);
    }
  }

  async function openWorkspaceFile(relPath) {
    try {
      const res = await apiFetch(`./api/v1/workspace/file?path=${encodeURIComponent(relPath)}`);
      state.activeFilePath = res.path;
      state.activeFileContent = res.content;
      $("active-file-badge").textContent = res.path;
      $("code-editor").value = res.content;
      updateEditorGutter();
      refreshWorkspaceFiles();
    } catch (err) {
      showAlert(`Cannot read ${relPath}: ${err.message}`);
    }
  }

  async function saveActiveWorkspaceFile() {
    const content = $("code-editor").value;
    try {
      const res = await apiFetch("./api/v1/workspace/file", {
        method: "POST",
        body: JSON.stringify({ path: state.activeFilePath, content }),
      });
      state.activeFileContent = content;
      if (res.payload?.unified_diff) {
        renderUnifiedDiff(res.payload.unified_diff);
      }
      await refreshWorkspaceFiles();
    } catch (err) {
      showAlert(`Save failed: ${err.message}`);
    }
  }

  function renderUnifiedDiff(diffText) {
    if (!diffText) return;
    const htmlLines = diffText.split("\n").map((line) => {
      const esc = escapeHtml(line);
      if (line.startsWith("+") && !line.startsWith("+++")) {
        return `<span class="diff-line-add">${esc}</span>`;
      }
      if (line.startsWith("-") && !line.startsWith("---")) {
        return `<span class="diff-line-del">${esc}</span>`;
      }
      return esc;
    });
    $("diff-viewer").innerHTML = htmlLines.join("\n");
    $("diff-drawer").classList.remove("hidden");
  }

  async function runFimAtCursor() {
    const editor = $("code-editor");
    const pos = editor.selectionStart || editor.value.length;
    const prefix = editor.value.slice(0, pos);
    const suffix = editor.value.slice(pos);
    $("fim-complete-btn").disabled = true;
    $("fim-complete-btn").textContent = "Completing…";
    try {
      const res = await apiFetch("./api/v1/code/complete", {
        method: "POST",
        body: JSON.stringify({ prefix, suffix, max_tokens: 40, temperature: 0.15 }),
      });
      const insertion = res.completion || "";
      editor.value = prefix + insertion + suffix;
      state.activeFileContent = editor.value;
      updateEditorGutter();
    } catch (err) {
      showAlert(`FIM completion error: ${err.message}`);
    } finally {
      $("fim-complete-btn").disabled = false;
      $("fim-complete-btn").textContent = "FIM Complete";
    }
  }

  // --- Multi-Step Coding Agent & Safety Confirmation Gate ---
  function renderAgentReport(report) {
    const log = $("agent-trace-log");
    log.innerHTML = "";

    if (report.status === "awaiting_confirmation" && report.pending_confirmation_token) {
      state.pendingConfirmation = {
        token: report.pending_confirmation_token,
        action: report.pending_action,
        goal: report.goal,
      };
      $("agent-confirm-reason").textContent = report.stop_reason;
      $("agent-confirm-token-label").textContent = `Token: ${report.pending_confirmation_token}`;
      $("agent-confirm-banner").classList.remove("hidden");
    } else {
      state.pendingConfirmation = null;
      $("agent-confirm-banner").classList.add("hidden");
    }

    (report.traces || []).forEach((t) => {
      const div = document.createElement("div");
      div.className = "trace-item";
      const okBadge = t.tool_result
        ? t.tool_result.ok
          ? `<span style="color:var(--success)">OK</span>`
          : `<span style="color:var(--warning)">${t.tool_result.requires_confirmation ? "CONFIRM_GATE" : "FAILED"}</span>`
        : "";
      const summaryLine = t.tool_result?.summary ? `<div class="muted-small">${escapeHtml(t.tool_result.summary)}</div>` : "";
      div.innerHTML = `
        <div class="trace-top">
          <span><span class="phase-badge">${escapeHtml(t.phase)}</span> <strong>Step ${t.step_index}</strong> ${t.tool_name ? `&bull; <code>${escapeHtml(t.tool_name)}</code>` : ""}</span>
          ${okBadge}
        </div>
        <div>${escapeHtml(t.rationale)}</div>
        ${summaryLine}
      `;
      log.appendChild(div);
    });

    const footer = document.createElement("div");
    footer.className = "trace-item";
    footer.innerHTML = `<strong>Run Status: ${escapeHtml(report.status.toUpperCase())}</strong> &mdash; ${escapeHtml(report.stop_reason)}`;
    log.appendChild(footer);

    if (report.unified_diffs && report.unified_diffs.length > 0) {
      renderUnifiedDiff(report.unified_diffs[report.unified_diffs.length - 1]);
    }
  }

  async function runAgentTask(goal, confirmationToken = null) {
    const btn = $("run-agent-btn");
    btn.disabled = true;
    btn.textContent = "Running…";
    try {
      const report = await apiFetch("./api/v1/agent/run", {
        method: "POST",
        body: JSON.stringify({
          goal,
          max_iterations: Number($("agent-max-iter").value || 6),
          reasoning_mode: $("reasoning-mode-select").value,
          confirmation_token: confirmationToken,
        }),
      });
      renderAgentReport(report);
      await refreshWorkspaceFiles();
    } catch (err) {
      showAlert(`Agent execution error: ${err.message}`);
    } finally {
      btn.disabled = false;
      btn.textContent = "Run Agent";
    }
  }

  // --- Health & Model Spec Poll ---
  async function loadHealthAndSpec() {
    const pill = $("health-pill");
    try {
      const health = await apiFetch("./api/v1/health");
      pill.className = "status-pill status-online";
      pill.innerHTML = `<span class="dot"></span> Online (${escapeHtml(health.runtime.device.toUpperCase())})`;

      const info = await apiFetch("./api/v1/model/info");
      state.modelInfo = info;
      $("runtime-honesty-banner").textContent = info.honest_training_status.status_banner;
      renderArchitectureSpec(info);
    } catch (err) {
      pill.className = "status-pill status-offline";
      pill.innerHTML = `<span class="dot"></span> API Offline`;
      $("runtime-honesty-banner").textContent = `Disconnected: ${err.message}`;
    }
  }

  function renderArchitectureSpec(info) {
    const rt = info.active_runtime;
    const spec = info.production_architecture_spec;
    const bd = spec.parameter_breakdown;
    $("architecture-details").innerHTML = `
      <div class="arch-card">
        <h3>Production Rootism-3B Blueprint</h3>
        <div><strong>Architecture ID:</strong> <code>${escapeHtml(spec.architecture_id)}</code></div>
        <div><strong>Verified Parameters:</strong> <code>${spec.verified_parameter_count.toLocaleString()}</code></div>
        <div><strong>Embeddings:</strong> ${bd.embedding_parameters.toLocaleString()}</div>
        <div><strong>GQA Attention + QK-Norm:</strong> ${bd.attention_parameters.toLocaleString()}</div>
        <div><strong>SwiGLU FFN:</strong> ${bd.feed_forward_parameters.toLocaleString()}</div>
        <div><strong>RMSNorms &amp; Untied Head:</strong> ${(bd.normalisation_parameters + bd.output_projection_parameters).toLocaleString()}</div>
        <div><strong>Context Window:</strong> ${spec.native_context_window.toLocaleString()} native / ${spec.extended_yarn_context_window.toLocaleString()} YaRN</div>
        <div><strong>Token Budget:</strong> ${spec.target_training_tokens.toLocaleString()} (&gt;600B enforced)</div>
      </div>
      <div class="arch-card">
        <h3>Active Loaded Checkpoint</h3>
        <div><strong>Stage Tag:</strong> <code>${escapeHtml(rt.training_stage_tag)}</code></div>
        <div><strong>Checkpoint Step:</strong> ${rt.checkpoint_step}</div>
        <div><strong>Processed Training Tokens:</strong> ${rt.processed_training_tokens.toLocaleString()}</div>
        <div><strong>Loaded Params:</strong> ${rt.analytical_parameters.toLocaleString()}</div>
        <div><strong>Device / Dtype / Quant:</strong> <code>${escapeHtml(rt.device)} / ${escapeHtml(rt.dtype)} / ${escapeHtml(rt.quantization)}</code></div>
        <div><strong>Model Memory Footprint:</strong> ${rt.memory_footprint_mb} MB</div>
        <div><strong>Tokenizer Vocab / Lock:</strong> ${rt.tokenizer_vocab_size} (<code>${escapeHtml(rt.tokenizer_fingerprint.slice(0, 14))}…</code>)</div>
      </div>
      <div class="arch-card">
        <h3>Supported Language Matrix</h3>
        <div><strong>10 Primary Languages:</strong> ${escapeHtml(info.supported_languages.primary.join(", "))}</div>
        <div style="margin-top:8px"><strong>9 Secondary Languages:</strong> ${escapeHtml(info.supported_languages.secondary.join(", "))}</div>
      </div>
    `;
  }

  // --- Export Helpers ---
  function downloadBlob(filename, content, mime) {
    const blob = new Blob([content], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  }

  // --- Wire Events ---
  function bindEvents() {
    $("new-session-btn").addEventListener("click", () => createNewSession());

    $("chat-composer").addEventListener("submit", (ev) => {
      ev.preventDefault();
      const val = $("chat-input").value.trim();
      if (!val || state.activeRequestId) return;
      $("chat-input").value = "";
      streamChatCompletion(val);
    });

    $("chat-input").addEventListener("keydown", (ev) => {
      if ((ev.ctrlKey || ev.metaKey) && ev.key === "Enter") {
        ev.preventDefault();
        $("chat-composer").requestSubmit();
      }
    });

    document.addEventListener("keydown", (ev) => {
      if (ev.key === "Escape" && state.activeRequestId) {
        cancelActiveGeneration();
      }
      if ((ev.ctrlKey || ev.metaKey) && ev.key.toLowerCase() === "s") {
        if (document.activeElement === $("code-editor")) {
          ev.preventDefault();
          saveActiveWorkspaceFile();
        }
      }
    });

    $("stop-stream-btn").addEventListener("click", cancelActiveGeneration);

    $("regenerate-btn").addEventListener("click", () => {
      const sess = activeSession();
      if (!sess || sess.messages.length === 0 || state.activeRequestId) return;
      if (sess.messages[sess.messages.length - 1].role === "assistant") {
        sess.messages.pop();
      }
      const lastUser = sess.messages.pop();
      if (lastUser && lastUser.role === "user") {
        streamChatCompletion(lastUser.content);
      }
    });

    $("clear-chat-btn").addEventListener("click", () => {
      const sess = activeSession();
      if (!sess) return;
      sess.messages = [];
      saveSessions();
      renderMessages();
    });

    $("chat-messages").addEventListener("click", (ev) => {
      const copyCodeBtn = ev.target.closest(".copy-code-btn");
      if (copyCodeBtn) {
        navigator.clipboard?.writeText(decodeURIComponent(copyCodeBtn.dataset.code));
        copyCodeBtn.textContent = "Copied!";
        setTimeout(() => (copyCodeBtn.textContent = "Copy"), 1200);
        return;
      }
      const insertBtn = ev.target.closest(".insert-code-btn");
      if (insertBtn) {
        $("code-editor").value = decodeURIComponent(insertBtn.dataset.code);
        updateEditorGutter();
        return;
      }
      const copyMsgBtn = ev.target.closest(".copy-msg-btn");
      if (copyMsgBtn) {
        const msg = activeSession()?.messages[Number(copyMsgBtn.dataset.idx)];
        if (msg) navigator.clipboard?.writeText(msg.content);
        return;
      }
      const editMsgBtn = ev.target.closest(".edit-msg-btn");
      if (editMsgBtn && !state.activeRequestId) {
        const idx = Number(editMsgBtn.dataset.idx);
        const sess = activeSession();
        const targetMsg = sess.messages[idx];
        if (targetMsg) {
          $("chat-input").value = targetMsg.content;
          sess.messages = sess.messages.slice(0, idx);
          saveSessions();
          renderMessages();
          $("chat-input").focus();
        }
      }
    });

    $("export-json-btn").addEventListener("click", () => {
      const sess = activeSession();
      downloadBlob(`${sess.id}.json`, JSON.stringify(sess, null, 2), "application/json");
    });

    $("export-md-btn").addEventListener("click", () => {
      const sess = activeSession();
      const md = sess.messages
        .map((m) => `### ${m.role.toUpperCase()}\n\n${m.reasoning_summary ? `> Summary: ${m.reasoning_summary}\n\n` : ""}${m.content}\n`)
        .join("\n---\n\n");
      downloadBlob(`${sess.id}.md`, `# ${sess.title}\n\n${md}`, "text/markdown");
    });

    // Editor & Workspace bindings
    $("code-editor").addEventListener("input", updateEditorGutter);
    $("code-editor").addEventListener("scroll", () => {
      $("editor-gutter").scrollTop = $("code-editor").scrollTop;
    });
    $("refresh-files-btn").addEventListener("click", refreshWorkspaceFiles);
    $("save-file-btn").addEventListener("click", saveActiveWorkspaceFile);
    $("fim-complete-btn").addEventListener("click", runFimAtCursor);
    $("close-diff-btn").addEventListener("click", () => $("diff-drawer").classList.add("hidden"));

    $("new-file-btn").addEventListener("click", async () => {
      const rel = window.prompt("New relative workspace path (e.g. src/new_helper.py):", "src/helper.py");
      if (!rel) return;
      state.activeFilePath = rel.trim();
      $("active-file-badge").textContent = state.activeFilePath;
      $("code-editor").value = "# Created in Rootism Studio\n";
      updateEditorGutter();
      await saveActiveWorkspaceFile();
    });

    $("revert-edit-btn").addEventListener("click", async () => {
      try {
        const res = await apiFetch("./api/v1/workspace/revert", { method: "POST" });
        await refreshWorkspaceFiles();
        if (res.payload?.reverted_path) {
          await openWorkspaceFile(state.activeFilePath);
        }
      } catch (err) {
        showAlert(`Rollback note: ${err.message}`);
      }
    });

    // Agent bindings
    $("agent-form").addEventListener("submit", (ev) => {
      ev.preventDefault();
      const goal = $("agent-goal-input").value.trim();
      if (goal) runAgentTask(goal);
    });

    $("quick-inspect-btn").addEventListener("click", async () => {
      const res = await apiFetch("./api/v1/agent/tool", {
        method: "POST",
        body: JSON.stringify({ tool_name: "inspect_project", args: {} }),
      });
      renderAgentReport({
        status: "completed",
        stop_reason: res.summary,
        traces: [{ step_index: 1, phase: "inspect", rationale: "Repository symbol & manifest scan", tool_name: "inspect_project", tool_result: res }],
      });
    });

    $("quick-pytest-btn").addEventListener("click", async () => {
      const res = await apiFetch("./api/v1/agent/tool", {
        method: "POST",
        body: JSON.stringify({ tool_name: "run_tests", args: {} }),
      });
      renderAgentReport({
        status: res.ok ? "completed" : "failed",
        stop_reason: res.summary,
        traces: [{ step_index: 1, phase: "test", rationale: "Run sandboxed pytest suite + FailureDiagnosticsEngine", tool_name: "run_tests", tool_result: res }],
      });
    });

    $("agent-approve-btn").addEventListener("click", async () => {
      if (!state.pendingConfirmation) return;
      const pending = state.pendingConfirmation;
      const decision = await apiFetch("./api/v1/agent/confirm", {
        method: "POST",
        body: JSON.stringify({
          confirmation_token: pending.token,
          approve: true,
          pending_tool_name: pending.action?.tool_name || null,
          pending_tool_args: pending.action?.tool_args || {},
        }),
      });
      $("agent-confirm-banner").classList.add("hidden");
      state.pendingConfirmation = null;
      renderAgentReport({
        status: "completed",
        stop_reason: decision.executed_result?.summary || "Confirmed action executed.",
        traces: [
          {
            step_index: 1,
            phase: "confirmed_execute",
            rationale: `User explicitly approved confirmation token ${pending.token}`,
            tool_name: pending.action?.tool_name,
            tool_result: decision.executed_result,
          },
        ],
      });
      await refreshWorkspaceFiles();
    });

    $("agent-reject-btn").addEventListener("click", async () => {
      if (!state.pendingConfirmation) return;
      const pending = state.pendingConfirmation;
      const decision = await apiFetch("./api/v1/agent/confirm", {
        method: "POST",
        body: JSON.stringify({ confirmation_token: pending.token, approve: false }),
      });
      $("agent-confirm-banner").classList.add("hidden");
      state.pendingConfirmation = null;
      renderAgentReport({
        status: "stopped_guardrail",
        stop_reason: decision.message,
        traces: [],
      });
    });
  }

  // --- Boot ---
  document.addEventListener("DOMContentLoaded", async () => {
    initThemeAndTabs();
    loadSessions();
    bindEvents();
    await loadHealthAndSpec();
    await refreshWorkspaceFiles();
    await openWorkspaceFile("src/metrics.py");
  });
})();
