const S = {
  data: { researchers: [] },
  active: sessionStorage.getItem("autoresearch-active"),
  tabs: JSON.parse(sessionStorage.getItem("autoresearch-tabs") || "{}"),
  mode: sessionStorage.getItem("autoresearch-mode") || "all",
  selected: JSON.parse(sessionStorage.getItem("autoresearch-selected") || "{}"),
  logs: {},
  logText: {},
  logLoaded: {},
  logFetches: {},
  diffFetches: {},
  diffCollapsed: {},
  stream: null,
  raw: "",
  events: null,
};

const UI = {
  researchers: document.querySelector("#researchers"),
  viewBar: document.querySelector("#view-bar"),
  empty: document.querySelector("#empty-state"),
  chart: document.querySelector("#chart-slot"),
  table: document.querySelector("#table-slot"),
  detail: document.querySelector("#detail-slot"),
};
const ansi = new Map([
  [30, "#8f8f8f"], [31, "#ff7b72"], [32, "#7ee787"], [33, "#f2cc60"], [34, "#58a6ff"], [35, "#d2a8ff"], [36, "#56d4dd"], [37, "#d6d6d6"],
  [90, "#6e7681"], [91, "#ffa198"], [92, "#7ee787"], [93, "#f2cc60"], [94, "#79c0ff"], [95, "#d2a8ff"], [96, "#56d4dd"], [97, "#f0f6fc"],
]);

const H = (s = "") => String(s).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");
const esc = (s = "") => String(s).replaceAll("\\u001b", "\x1b").replaceAll("\\x1b", "\x1b");
const fmt = (v) => (v == null ? "" : Number.isFinite(v) ? Number(v.toPrecision(4)).toString() : String(v));
const artifactKey = (tab) => tab?.id || "";
const PIERRE_DIFFS_URL = "./assets/pierre-diffs.min.js";
const PIERRE_DIFF_CSS = `
:host{--diffs-font-family:var(--mono);--diffs-header-font-family:var(--mono);--diffs-font-size:var(--base);--diffs-line-height:1.45}
::slotted([slot=header-prefix]){display:inline-flex;align-items:center;background:#000}
[data-diffs-header=default],[data-separator],[data-expand-button]{font-family:var(--mono);font-size:var(--base);letter-spacing:0}
[data-diffs-header=default]{cursor:pointer}
`;
const pierreDiffs = import(PIERRE_DIFFS_URL).catch(err => {
  console.warn("Pierre diff renderer unavailable; using built-in renderer.", err);
  return null;
});

function saveUiSelection() {
  sessionStorage.setItem("autoresearch-active", S.active || "");
  sessionStorage.setItem("autoresearch-tabs", JSON.stringify(S.tabs));
  sessionStorage.setItem("autoresearch-mode", S.mode);
  sessionStorage.setItem("autoresearch-selected", JSON.stringify(S.selected));
}

function updateUiSelection(patch = {}, options = {}) {
  rememberLogScroll();
  Object.assign(S, patch);
  keepSelectionInPayload();
  saveUiSelection();
  render(options);
}

function keepSelectionInPayload() {
  const researchers = S.data.researchers || [];
  if (!researchers.some(r => r.id === S.active)) {
    S.active = researchers[0]?.id;
  }
  const { row } = selectedPayloadView();
  if (row) {
    S.selected[S.active] = row.id;
  } else if (S.active) {
    delete S.selected[S.active];
  }
}

function selectedPayloadView() {
  const researchers = S.data.researchers || [];
  const researcher = researchers.find(r => r.id === S.active);
  const view = researcher?.views?.[S.mode] || researcher?.views?.all || { table_rows: [], chart_points: [], metric_label: "score", axis_label: "score" };
  const rows = view.table_rows || [];
  let row = rows.find(r => r.id === S.selected[S.active]);
  if (!row) row = rows.find(r => r.kind === "live");
  if (!row) row = rows.at(-1) || null;
  return { researchers, researcher, view, rows, row };
}

function render(options = {}) {
  const st = selectedPayloadView();
  UI.researchers.innerHTML = researcherNav(st.researchers);
  if (!st.researchers.length) {
    UI.empty.hidden = false;
    UI.empty.innerHTML = waiting("Waiting for a researcher", "Runs will appear here as soon as a researcher creates a log directory.");
    UI.viewBar.hidden = true;
    UI.viewBar.innerHTML = UI.chart.innerHTML = UI.table.innerHTML = UI.detail.innerHTML = "";
    return stopStream();
  }
  UI.empty.hidden = true;
  UI.viewBar.hidden = false;
  UI.viewBar.innerHTML = modeToggle();
  UI.chart.innerHTML = chart(st.researcher, st.view);
  UI.table.innerHTML = table(st.view, st.rows, st.row);
  if (st.row) {
    const tabKey = selectedTab(st.row);
    const current = UI.detail.querySelector("#experiment-detail");
    if (current?.dataset.run !== st.row.id || current?.dataset.tab !== tabKey) {
      UI.detail.innerHTML = detail(st.row, tabKey);
    } else {
      current.querySelector(".panel-meta").textContent = st.row.meta || st.row.label || st.row.id;
    }
    renderViewer(st.row);
  } else {
    UI.detail.innerHTML = "";
    stopStream();
  }
  if (options.scrollToDetail) {
    requestAnimationFrame(() => document.querySelector("#experiment-detail")?.scrollIntoView({ behavior: "smooth", block: "start" }));
  }
}

function modeToggle() {
  const label = S.mode === "improvements" ? "All experiments" : "Improvements only";
  return `<button class="view-toggle" data-action="mode-toggle" title="Show ${H(label.toLowerCase())}">${label}</button>`;
}

function researcherNav(researchers) {
  return researchers.map(r => `
    <button class="${r.id === S.active ? "active" : ""}" data-action="active" data-id="${H(r.id)}">
      <span class="research-status ${r.live ? "running" : ""}" aria-hidden="true"></span>
      <span class="nav-copy"><strong>${H(r.label)}</strong><span class="muted">${H(r.meta)}</span></span>
    </button>`).join("");
}

function chart(researcher, view) {
  const pts = view.chart_points || [];
  if (!pts.length) return `<section class="chart-panel waiting"><div class="panel-head"><h2>${H(researcher.label)} waiting</h2><span class="status-pill ${researcher.status === "running" ? "running" : ""}">${H(researcher.status)}</span></div><p class="muted">Waiting for the first metric.</p></section>`;
  const w = 1040, h = 210, m = { l: 92, r: 16, t: 22, b: 54 }, pw = w - m.l - m.r, ph = h - m.t - m.b;
  const attempts = Math.max(view.attempt_count || pts.length, ...pts.map(p => p.number || 1));
  const x = (n) => m.l + pw * (attempts < 2 ? .5 : (n - 1) / (attempts - 1));
  const scale = yScale(pts.map(p => p.score));
  const xy = pts.map(p => [x(p.number || 1), m.t + ph - scale.norm(p.score) * ph, p]);
  const line = (x1, y1, x2, y2, c) => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" class="${c}"/>`;
  const text = (x, y, s, a = "middle", c = "axis-tick", extra = "") => `<text x="${x}" y="${y}" text-anchor="${a}" class="${c}" ${extra}>${H(s)}</text>`;
  return `<section class="chart-panel"><div class="chart"><svg viewBox="0 0 ${w} ${h}">
    ${scale.ticks.map(v => line(m.l, m.t + ph - scale.norm(v) * ph, m.l + pw, m.t + ph - scale.norm(v) * ph, "grid-line")).join("")}
    ${Array.from({ length: attempts }, (_, i) => line(x(i + 1), m.t, x(i + 1), m.t + ph, "grid-line")).join("")}
    ${line(m.l, m.t, m.l, m.t + ph, "axis-line")}${line(m.l, m.t + ph, m.l + pw, m.t + ph, "axis-line")}
    ${scale.ticks.map(v => text(m.l - 16, m.t + ph - scale.norm(v) * ph + 4, fmtAxis(v, scale.step), "end")).join("")}
    <polyline points="${xy.map(([x, y]) => `${x},${y}`).join(" ")}" fill="none" stroke="var(--accent)" stroke-width="2"/>
    ${Array.from({ length: attempts }, (_, i) => text(x(i + 1), m.t + ph + 24, `A${i + 1}`)).join("")}
    ${xy.map(([x, y, p]) => `<circle class="chart-point" cx="${x}" cy="${y}" r="5" data-action="select" data-run="${H(p.id)}"><title>${H(p.title)}</title></circle>`).join("")}
    ${text(12, m.t + ph / 2, view.axis_label || view.metric_label || "score", "middle", "axis-title", `transform="rotate(-90 12 ${m.t + ph / 2})"`)}
    ${text(m.l + pw / 2, h - 8, "attempt", "middle", "axis-title")}
  </svg></div></section>`;
}

function yScale(values) {
  let lo = Math.min(...values), hi = Math.max(...values);
  if (lo === hi) {
    const pad = Math.max(Math.abs(lo) * .1, .01);
    lo -= pad; hi += pad;
  }
  const pad = (hi - lo) * .08;
  lo -= pad; hi += pad;
  const step = niceStep(hi - lo);
  const min = Math.floor(lo / step) * step;
  const max = Math.ceil(hi / step) * step;
  const ticks = [];
  for (let v = min; v <= max + step / 2; v += step) ticks.push(Math.abs(v) < step / 1000 ? 0 : v);
  return { ticks, step, norm: (v) => (v - min) / (max - min || 1) };
}

function niceStep(span) {
  const raw = span / 4;
  const pow = 10 ** Math.floor(Math.log10(raw || 1));
  const unit = raw / pow;
  return (unit <= 1 ? 1 : unit <= 2 ? 2 : unit <= 5 ? 5 : 10) * pow;
}

function fmtAxis(value, step) {
  if (Math.abs(value) < step / 1000) value = 0;
  const decimals = Math.max(0, Math.min(6, Math.ceil(-Math.log10(step)) + 1));
  return Number(value.toFixed(decimals)).toString();
}

function table(view, rows, selected) {
  let body = `<tr><td class="empty-row" colspan="3">No visible experiments. Toggle attempts to show more.</td></tr>`;
  if (rows.length) {
    body = rows.map(row => {
      const selectedClass = selected?.id === row.id ? "selected" : "";
      const number = row.kind === "live" ? `<span class="live-pulse" title="Live run"></span>` : H(row.number);
      const score = row.kind === "live" ? "" : fmt(row.score);
      return `<tr class="${selectedClass}" data-action="select" data-run="${H(row.id)}">
        <td class="experiment-id attempt-number">${number}</td>
        <td>${score}</td>
        <td class="description-cell">${H(row.description || "")}</td>
      </tr>`;
    }).join("");
  }
  return `<section class="experiment-index"><div class="table-wrap"><table><thead><tr>${["#", view.metric_label, "What changed"].map(h => `<th>${H(h)}</th>`).join("")}</tr></thead><tbody>${body}</tbody></table></div></section>`;
}

function detail(row, selected = selectedTab(row)) {
  return `<section class="experiment-panel detail-panel" id="experiment-detail" data-run="${H(row.id)}" data-tab="${H(selected)}">
    <div class="experiment-viewer"><div class="tabs">${row.tab_order.map(key => `<button class="${key === selected ? "active" : ""}" data-action="tab" data-tab="${key}">${H(row.tabs[key].label)}</button>`).join("")}</div><div class="viewer" id="viewer"></div></div>
    <div class="panel-meta">${H(row.meta || row.label || row.id)}</div>
  </section>`;
}

function selectedTab(row) {
  const tabs = row.tab_order;
  const preferred = row.tabs.agent.id ? "agent" : "logs";
  const saved = S.tabs.__global;
  if (tabs.includes(saved)) {
    return saved;
  }
  if (tabs.includes(preferred)) {
    return preferred;
  }
  return tabs[0];
}

function renderViewer(row) {
  const tabKey = selectedTab(row);
  const tab = row.tabs[tabKey];
  const v = document.querySelector("#viewer");
  if (!v) return stopStream();
  if (tabKey === "diff") {
    stopStream();
    cleanUpPierreDiffs(v);
    return v.replaceChildren(diffPanel(row));
  }
  if (tab.format === "markdown") {
    const existing = v.querySelector(".markdown-panel");
    if (existing?.dataset.path === artifactKey(tab)) {
      const status = v.querySelector(".log-status");
      if (status) {
        status.textContent = logStatus(tab);
        status.className = `log-status ${tab.live ? "live" : ""}`;
      }
      hydrateLog(tab, status, existing);
      return;
    }
    cleanUpPierreDiffs(v);
    return v.replaceChildren(markdownPanel(tab));
  }
  const existing = v.querySelector("pre.log");
  if (existing?.dataset.path === artifactKey(tab)) {
    const status = v.querySelector(".log-status");
    if (status) {
      status.textContent = logStatus(tab);
      status.className = `log-status ${tab.live ? "live" : ""}`;
    }
    hydrateLog(tab, status, existing);
    return;
  }
  cleanUpPierreDiffs(v);
  v.replaceChildren(logPanel(tab));
}

function cleanUpPierreDiffs(root) {
  root.querySelectorAll(".diffbox").forEach(box => box.pierreCodeView?.cleanUp());
}

function logStatus(tab) {
  const key = artifactKey(tab);
  if (!key) return `${tab.label.toLowerCase()}: no stream`;
  if (tab.live) return `${tab.label.toLowerCase()}: ${S.logLoaded[key] ? "live" : "tail loading full log"}`;
  return `${tab.label.toLowerCase()}: ${S.logLoaded[key] ? "full" : "tail loading full log"}`;
}

function logPanel(tab) {
  const key = artifactKey(tab);
  const text = key ? S.logText[key] || tab.tail || "" : tab.tail || "";
  const panel = document.createElement("div");
  panel.className = "log-panel";
  panel.innerHTML = `<div class="log-toolbar"><span class="log-status ${tab.live ? "live" : ""}">${H(logStatus(tab))}</span></div>`;
  if (!key && !text) {
    panel.appendChild(fragment(`<div class="waiting-panel">${waiting(tab.label, `Waiting for ${tab.label.toLowerCase()} output.`)}</div>`));
    return panel;
  }
  const node = document.createElement("pre");
  node.className = "log";
  if (key) bindLogScroll(node, key);
  panel.append(node);
  colorize(node, esc(text));
  restoreLogScroll(node, key);
  hydrateLog(tab, panel.querySelector(".log-status"), node);
  return panel;
}

function markdownPanel(tab) {
  const key = artifactKey(tab);
  const text = key ? S.logText[key] || tab.tail || "" : tab.tail || "";
  const panel = document.createElement("div");
  panel.className = "log-panel";
  panel.innerHTML = `<div class="log-toolbar"><span class="log-status ${tab.live ? "live" : ""}">${H(logStatus(tab))}</span></div>`;
  const node = document.createElement("div");
  node.className = "markdown-panel";
  if (key) bindLogScroll(node, key);
  panel.append(node);
  renderText(node, tab, esc(text));
  restoreLogScroll(node, key);
  hydrateLog(tab, panel.querySelector(".log-status"), node);
  return panel;
}

async function hydrateLog(tab, status, node) {
  const key = artifactKey(tab);
  if (!key) return stopStream();
  if (!S.logLoaded[key]) {
    const hasText = Boolean(S.logText[key] || tab.tail || node.textContent);
    if (status && !hasText) status.textContent = `${tab.label.toLowerCase()}: loading full log`;
    S.logFetches[key] ||= fetch(`file?id=${encodeURIComponent(key)}`)
      .then(response => {
        if (!response.ok) throw new Error(`log fetch failed: ${response.status}`);
        return response.text();
      })
      .then(text => {
        const full = esc(text);
        S.logText[key] = full;
        S.logLoaded[key] = true;
        return full;
      })
      .finally(() => { delete S.logFetches[key]; });
    const full = await S.logFetches[key].catch(err => {
      if (status) {
        status.textContent = err.message;
        status.className = "log-status warn";
      }
      return null;
    });
    if (full == null) return;
    if (!node.isConnected) return;
    S.logText[key] = full;
    replaceText(node, tab, full);
    if (status) {
      status.textContent = logStatus(tab);
      status.className = `log-status ${tab.live ? "live" : ""}`;
    }
  }
  if (tab.live) {
    ensureStream(tab, status, node);
  } else {
    stopStream();
  }
}

function ensureStream(tab, status, node) {
  const artifact = artifactKey(tab);
  const key = `${tab.format}:${artifact}`;
  if (S.stream?.key === key) {
    S.stream.status = status;
    S.stream.node = node;
    return;
  }
  stopStream();
  const offset = byteLength(S.logText[artifact] || "");
  const mode = tab.format === "markdown" ? "&replace=1" : "";
  const source = new EventSource(`tail?id=${encodeURIComponent(artifact)}&offset=${offset}${mode}`);
  S.stream = { key, source, status, node };
  source.onmessage = (event) => {
    const data = JSON.parse(event.data);
    const text = esc(data.text || "");
    if (data.replace) {
      S.logText[artifact] = text;
      replaceText(S.stream?.node, tab, text);
    } else {
      S.logText[artifact] = (S.logText[artifact] || "") + text;
      appendText(S.stream?.node, tab, text);
    }
    if (S.stream?.status) {
      S.stream.status.textContent = `${tab.label.toLowerCase()}: live`;
      S.stream.status.className = "log-status live";
    }
  };
  source.onerror = () => {
    if (!S.stream?.status) return;
    S.stream.status.textContent = "reconnecting";
    S.stream.status.className = "log-status warn";
  };
}

function stopStream() {
  S.stream?.source.close();
  S.stream = null;
}

function colorize(node, text) {
  node.innerHTML = text.split("\n").map(logLine).join("\n");
}

function renderText(node, tab, text) {
  if (tab.format === "markdown") node.innerHTML = markdown(text);
  else colorize(node, text);
}

function replaceText(node, tab, text) {
  const scroll = savedOrCurrentScroll(node, artifactKey(tab));
  renderText(node, tab, text);
  restoreLogScroll(node, artifactKey(tab), scroll);
}

function appendText(node, tab, text) {
  if (!node?.isConnected) return;
  const key = artifactKey(tab);
  const scroll = savedOrCurrentScroll(node, key);
  if (tab.format === "markdown") renderText(node, tab, S.logText[key] || text);
  else node.insertAdjacentHTML("beforeend", text.split("\n").map(logLine).join("\n"));
  restoreLogScroll(node, key, scroll);
}

function bindLogScroll(node, path) {
  node.dataset.path = path;
  node.tabIndex = 0;
  node.addEventListener("scroll", () => { S.logs[path] = logScroll(node); }, { passive: true });
  node.addEventListener("wheel", event => {
    if (event.deltaY < 0) S.logs[path] = { ...logScroll(node), pinned: false };
  }, { passive: true });
  node.addEventListener("touchstart", () => { S.logs[path] = { ...logScroll(node), pinned: false }; }, { passive: true });
  node.addEventListener("keydown", event => {
    if (["ArrowUp", "PageUp", "Home"].includes(event.key)) S.logs[path] = { ...logScroll(node), pinned: false };
  });
}

function savedOrCurrentScroll(node, path) {
  const saved = S.logs[path];
  return saved && !saved.pinned ? saved : logScroll(node);
}

function rememberLogScroll() {
  document.querySelectorAll("pre.log[data-path],.markdown-panel[data-path]").forEach(node => { S.logs[node.dataset.path] = logScroll(node); });
}

function logScroll(node) {
  const bottomGap = Math.max(0, node.scrollHeight - node.clientHeight - node.scrollTop);
  return { top: node.scrollTop, pinned: bottomGap <= 8 };
}

function restoreLogScroll(node, path, scroll = S.logs[path]) {
  const apply = () => {
    if (!scroll || scroll.pinned) {
      node.scrollTop = node.scrollHeight;
    } else {
      node.scrollTop = scroll.top;
    }
  };
  apply();
  requestAnimationFrame(apply);
}

function byteLength(text) {
  return new TextEncoder().encode(text).length;
}

function logLine(line) {
  const event = agentEventLine(line);
  if (event) return event;
  if (line.includes("\x1b[")) return ansiLine(line);
  const plain = line.replace(/\x1b\[[0-9;]*m/g, "");
  if (!/^[┏┡└┗┣┠┯┷─━│┃]/.test(plain)) return H(line);
  if (plain.startsWith("│ ")) {
    const parts = plain.split("│");
    if (parts.length >= 4) return `<span class="metric-row">│<span class="metric-key">${H(parts[1])}</span>│<span class="metric-value">${H(parts[2])}</span>│</span>`;
  }
  return `<span class="metric-border">${H(plain)}</span>`;
}

function agentEventLine(line) {
  const text = line.trim();
  if (!text.startsWith("{") || !text.endsWith("}")) return "";
  let event;
  try { event = JSON.parse(text); } catch { return ""; }
  if (!event?.type) return "";
  if (event.type === "init") return eventHtml("init", event.model || "agent", event.session_id ? `session ${event.session_id}` : "");
  if (event.type === "message") return messageEvent(event);
  if (event.type === "tool_use") return eventHtml("tool", event.tool_name || "tool", toolDetail(event.tool_name, event.parameters));
  if (event.type === "tool_result") return eventHtml("result", event.status || "done", compact(event.output || event.result || event.content || ""));
  return eventHtml(event.type.replaceAll("_", " "), event.status || event.role || "event", compact(event.content || event.message || ""));
}

function messageEvent(event) {
  const role = event.role || "agent", content = eventText(event.content);
  if (role === "user" && content.length > 2000) return eventHtml("prompt", "loaded", `${content.length.toLocaleString()} chars`);
  return eventHtml(role, content ? compact(content, 1800) : "message");
}

function toolDetail(tool, params = {}) {
  if (tool === "run_shell_command" && params.command) return `$ ${compact(params.command, 1200)}`;
  if (tool === "read_file" && params.file_path) return params.file_path;
  if (tool === "replace") return [params.file_path, params.instruction].filter(Boolean).join(" · ");
  if (tool === "update_topic") return [params.title, params.summary || params.strategic_intent].filter(Boolean).join("\n");
  return compact(Object.entries(params || {}).map(([k, v]) => `${k}=${eventText(v)}`).join(" "), 1200);
}

function eventHtml(kind, title, detail = "") {
  return `<span class="agent-event"><span class="event-kind">${H(kind)}</span><span class="event-title">${H(title)}</span>${detail ? `<span class="event-detail">${H(detail)}</span>` : ""}</span>`;
}

function eventText(value) {
  if (value == null) return "";
  if (typeof value === "string") return value;
  if (Array.isArray(value)) return value.map(eventText).filter(Boolean).join("\n");
  if (typeof value === "object") return value.text || value.content || value.output || JSON.stringify(value);
  return String(value);
}

function compact(value, limit = 1000) {
  const text = eventText(value).replace(/\r\n/g, "\n").trim();
  return text.length > limit ? `${text.slice(0, limit)} ...` : text;
}

function ansiLine(line) {
  let out = "", index = 0, open = false;
  for (const match of line.matchAll(/\x1b\[([0-9;]*)m/g)) {
    out += H(line.slice(index, match.index));
    if (open) out += "</span>";
    const codes = (match[1] || "0").split(";").filter(Boolean).map(Number);
    const color = [...codes].reverse().find(c => ansi.has(c));
    open = !codes.includes(0) && !!color;
    if (open) out += `<span style="color:${ansi.get(color)};${codes.includes(1) ? "font-weight:700" : ""}">`;
    index = match.index + match[0].length;
  }
  return out + H(line.slice(index)) + (open ? "</span>" : "");
}

function markdown(text) {
  const lines = text.replace(/\r\n/g, "\n").split("\n");
  let out = "", list = false, code = false, buf = [];
  const inline = (s) => H(s).replace(/`([^`]+)`/g, "<code>$1</code>").replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  const closeList = () => { if (list) { out += "</ul>"; list = false; } };
  const flushCode = () => { out += `<pre>${H(buf.join("\n"))}</pre>`; buf = []; };
  for (const line of lines) {
    if (line.startsWith("```")) { code ? flushCode() : closeList(); code = !code; continue; }
    if (code) { buf.push(line); continue; }
    if (!line.trim()) { closeList(); continue; }
    const h = /^(#{2,4})\s+(.+)$/.exec(line);
    if (h) { closeList(); out += `<h${h[1].length}>${inline(h[2])}</h${h[1].length}>`; continue; }
    const li = /^[-*]\s+(.+)$/.exec(line);
    if (li) { if (!list) { out += "<ul>"; list = true; } out += `<li>${inline(li[1])}</li>`; continue; }
    closeList(); out += `<p>${inline(line)}</p>`;
  }
  if (code) flushCode();
  closeList();
  return out || `<p class="muted">No notes yet.</p>`;
}

function diffPanel(row) {
  const diff = row.tabs.diff, compact = diff.compact || "";
  const panel = document.createElement("div");
  panel.className = "diff-panel";
  if (!compact) {
    panel.append(fragment(`<div class="waiting-panel">${waiting("Diff", diff.id ? "Loading captured code diff." : row.live ? "Waiting for the first captured code diff." : "No code diff.")}</div>`));
    hydrateDiff(row, panel);
    return panel;
  }
  const box = document.createElement("div");
  box.className = "diffbox";
  box.dataset.full = "0";
  box.innerHTML = rawDiff(compact);
  renderPierreDiff(row.tabs.diff, box, row.id);
  panel.append(box);
  hydrateDiff(row, panel);
  return panel;
}

async function hydrateDiff(row, panel) {
  const diff = row.tabs.diff;
  if (!diff?.id) return;
  if (diff.files || !diff.files_id && diff.compact) return;
  const key = `${row.id}:${diff.id}:${diff.files_id || ""}`;
  S.diffFetches[key] ||= Promise.all([
    fetch(`file?id=${encodeURIComponent(diff.id)}`).then(response => response.ok ? response.text() : ""),
    diff.files_id ? fetch(`file?id=${encodeURIComponent(diff.files_id)}`).then(response => response.ok ? response.json() : []) : Promise.resolve([]),
  ]).finally(() => { delete S.diffFetches[key]; });
  const [compact, files] = await S.diffFetches[key];
  if (!panel.isConnected || !compact) return;
  row.tabs.diff.compact = compact;
  row.tabs.diff.files = files;
  const rendered = diffPanel(row);
  panel.replaceChildren(...rendered.childNodes);
}

const rawDiff = (text) => `<pre class="raw-diff">${H(text)}</pre>`;

async function renderPierreDiff(diff, box, rowId) {
  const text = diff.compact || "";
  if (!text.includes("diff --git ")) return;
  const key = `${text.length}:${text.slice(0, 80)}`;
  box.dataset.pierreKey = key;
  try {
    const module = await pierreDiffs;
    if (!module) return;
    const { CodeView, parseDiffFromFile } = module;
    const files = Array.isArray(diff.files) ? diff.files : [];
    const items = files.map(file => ({
      id: file.name,
      type: "diff",
      collapsed: !!S.diffCollapsed[`${rowId}:${file.name}`],
      fileDiff: parseDiffFromFile(
        { name: file.name, contents: file.old_text || "", cacheKey: `${key}:${file.name}:old` },
        { name: file.name, contents: file.new_text || "", cacheKey: `${key}:${file.name}:new` },
        { context: 3 },
      ),
      version: text.length,
    }));
    if (!box.isConnected || box.dataset.pierreKey !== key || !items.length) return;
    box.pierreCodeView?.cleanUp();
    box.replaceChildren();
    box.classList.add("pierre-diffbox");
    const viewer = new CodeView({
      theme: "vitesse-black",
      themeType: "dark",
      diffStyle: "split",
      overflow: "wrap",
      lineDiffType: "word",
      diffIndicators: "bars",
      hunkSeparators: "line-info",
      expansionLineCount: 25,
      unsafeCSS: PIERRE_DIFF_CSS,
      stickyHeaders: true,
      renderHeaderPrefix: (_file, context) => {
        const collapsed = !!context.item.collapsed;
        const toggle = document.createElement("button");
        toggle.className = "diff-toggle";
        toggle.type = "button";
        toggle.setAttribute("aria-label", collapsed ? "Expand diff" : "Collapse diff");
        toggle.setAttribute("aria-expanded", collapsed ? "false" : "true");
        toggle.innerHTML = collapsed
          ? `<svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><path fill="currentColor" d="M6.5 3.5 11 8l-4.5 4.5-1-1L9 8 5.5 4.5z"/></svg>`
          : `<svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><path fill="currentColor" d="m3.5 6.5 1-1L8 9l3.5-3.5 1 1L8 11z"/></svg>`;
        return toggle;
      },
      onPostRender: () => bindPierreHeaderClicks(viewer, rowId),
    });
    viewer.setup(box);
    viewer.setItems(items);
    viewer.render(true);
    box.pierreCodeView = viewer;
    bindPierreHeaderClicks(viewer, rowId);
  } catch (err) {
    console.warn("Pierre diff rendering failed; using built-in renderer.", err);
  }
}

function bindPierreHeaderClicks(viewer, rowId) {
  for (const item of viewer.getRenderedItems()) {
    const header = item.element.shadowRoot?.querySelector("[data-diffs-header]");
    if (!header || header.dataset.boundCollapse) continue;
    header.dataset.boundCollapse = "1";
    header.setAttribute("role", "button");
    header.setAttribute("aria-expanded", item.item.collapsed ? "false" : "true");
    header.addEventListener("click", () => {
      const current = viewer.getItem(item.id) || item.item;
      const collapsed = !current.collapsed;
      S.diffCollapsed[`${rowId}:${item.id}`] = collapsed;
      viewer.updateItem({ ...current, collapsed, version: Number(current.version || 0) + 1 });
      viewer.render(true);
      bindPierreHeaderClicks(viewer, rowId);
    });
  }
}

function waiting(title, text) {
  return `<div class="waiting-row"><span class="live-dot"></span><div><strong>${H(title)}</strong><p class="muted">${H(text)}</p></div></div>`;
}

function fragment(html) {
  const t = document.createElement("template");
  t.innerHTML = html;
  return t.content;
}

document.addEventListener("click", (event) => {
  const target = event.target.closest("[data-action]");
  if (!target) return;

  switch (target.dataset.action) {
    case "active":
      updateUiSelection({ active: target.dataset.id });
      break;
    case "mode-toggle":
      updateUiSelection({ mode: S.mode === "improvements" ? "all" : "improvements" });
      break;
    case "select":
      updateUiSelection({ selected: { ...S.selected, [S.active]: target.dataset.run } }, { scrollToDetail: true });
      break;
    case "tab": {
      const row = selectedPayloadView().row;
      if (row) updateUiSelection({ tabs: { ...S.tabs, __global: target.dataset.tab } });
      break;
    }
  }
});

function applyData(raw) {
  const data = JSON.parse(raw), key = JSON.stringify(data);
  if (key === S.raw) return;
  updateUiSelection({ raw: key, data });
}

function connect() {
  S.events?.close();
  S.events = new EventSource("events");
  S.events.onmessage = (event) => applyData(event.data);
  S.events.onerror = () => {
    UI.empty.hidden = false;
    UI.empty.innerHTML = waiting("UI error", "Lost connection to observer.");
  };
}

addEventListener("beforeunload", () => {
  stopStream();
  S.events?.close();
});
connect();
