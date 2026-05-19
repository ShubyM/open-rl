const S = {
  data: { researchers: [] },
  active: sessionStorage.getItem("autoresearch-active"),
  tabs: JSON.parse(sessionStorage.getItem("autoresearch-tabs") || "{}"),
  mode: sessionStorage.getItem("autoresearch-mode") || "all",
  selected: JSON.parse(sessionStorage.getItem("autoresearch-selected") || "{}"),
  logs: {},
  logText: {},
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
    UI.detail.innerHTML = detail(st.row);
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
    ${scale.ticks.map(v => text(m.l - 16, m.t + ph - scale.norm(v) * ph + 4, fmt(v), "end")).join("")}
    <polyline points="${xy.map(([x, y]) => `${x},${y}`).join(" ")}" fill="none" stroke="var(--accent)" stroke-width="2"/>
    ${Array.from({ length: attempts }, (_, i) => text(x(i + 1), m.t + ph + 24, `E${i + 1}`)).join("")}
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
  const span = hi - lo, paddedLo = lo - span * .08, paddedHi = hi + span * .08;
  return { ticks: [0, .25, .5, .75, 1].map(t => paddedLo + (paddedHi - paddedLo) * t), norm: (v) => (v - paddedLo) / (paddedHi - paddedLo) };
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

function detail(row) {
  const selected = selectedTab(row);
  return `<section class="experiment-panel detail-panel" id="experiment-detail">
    <div class="experiment-viewer"><div class="tabs">${row.tab_order.map(key => `<button class="${key === selected ? "active" : ""}" data-action="tab" data-tab="${key}">${H(row.tabs[key].label)}</button>`).join("")}</div><div class="viewer" id="viewer"></div></div>
    <div class="panel-meta">${H(row.meta || row.label || row.id)}</div>
  </section>`;
}

function selectedTab(row) {
  const tabs = row.tab_order;
  const preferred = row.tabs.agent.path ? "agent" : "logs";
  if (tabs.includes(S.tabs[row.id])) {
    return S.tabs[row.id];
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
    return v.replaceChildren(diffPanel(row));
  }
  v.replaceChildren(logPanel(tab));
}

function logPanel(tab) {
  const text = tab.path ? S.logText[tab.path] || tab.tail || "" : tab.tail || "";
  const panel = document.createElement("div");
  const status = tab.path ? `${tab.label.toLowerCase()}: ${tab.live ? "live" : "tail"}` : `${tab.label.toLowerCase()}: no stream`;
  panel.className = "log-panel";
  panel.innerHTML = `<div class="log-toolbar"><span class="log-status ${tab.live ? "live" : ""}">${H(status)}</span></div>`;
  if (!tab.path && !text) {
    panel.appendChild(fragment(`<div class="waiting-panel">${waiting(tab.label, `Waiting for ${tab.label.toLowerCase()} output.`)}</div>`));
    return panel;
  }
  const node = document.createElement("pre");
  node.className = "log";
  if (tab.path) node.dataset.path = tab.path;
  panel.append(node);
  colorize(node, esc(text));
  restoreLogScroll(node, tab.path);
  hydrateLog(tab, panel.querySelector(".log-status"), node);
  return panel;
}

async function hydrateLog(tab, status, node) {
  if (!tab.path) return stopStream();
  if (!S.logText[tab.path]) {
    const full = esc(await (await fetch(`file?path=${encodeURIComponent(tab.path)}`)).text());
    if (!node.isConnected) return;
    S.logText[tab.path] = full;
    replaceLog(node, tab.path, full);
  }
  if (tab.live) {
    ensureStream(tab, status, node);
  } else {
    stopStream();
  }
}

function ensureStream(tab, status, node) {
  const key = `${tab.format}:${tab.path}`;
  if (S.stream?.key === key) {
    S.stream.status = status;
    S.stream.node = node;
    return;
  }
  stopStream();
  const offset = byteLength(S.logText[tab.path] || "");
  const source = new EventSource(`stream?path=${encodeURIComponent(tab.path)}&offset=${offset}`);
  S.stream = { key, source, status, node };
  source.onmessage = (event) => {
    const chunk = esc(JSON.parse(event.data).text || "");
    S.logText[tab.path] = (S.logText[tab.path] || "") + chunk;
    appendLog(S.stream?.node, tab.path, chunk);
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

function replaceLog(node, path, text) {
  const scroll = logScroll(node);
  colorize(node, text);
  restoreLogScroll(node, path, scroll);
}

function appendLog(node, path, text) {
  if (!node?.isConnected) return;
  const scroll = logScroll(node);
  node.insertAdjacentHTML("beforeend", text.split("\n").map(logLine).join("\n"));
  restoreLogScroll(node, path, scroll);
}

function rememberLogScroll() {
  document.querySelectorAll("pre.log[data-path]").forEach(node => { S.logs[node.dataset.path] = logScroll(node); });
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

function diffPanel(row) {
  const diff = row.tabs.diff, compact = diff.compact || "";
  if (!compact) return fragment(`<div class="waiting-panel">${waiting("Diff", row.live ? "Waiting for the first captured code diff." : "No code diff.")}</div>`);
  const panel = document.createElement("div");
  const box = document.createElement("div");
  panel.className = "diff-panel";
  box.className = "diffbox";
  box.dataset.full = "0";
  box.innerHTML = diffHtml(compact, hasFullDiff(row));
  panel.append(box);
  return panel;
}

const hasFullDiff = (row) => !!row.tabs.diff.full && row.tabs.diff.full !== row.tabs.diff.compact;
const rawDiff = (text) => `<pre class="raw-diff">${H(text)}</pre>`;

function diffHtml(text, toggle = false) {
  if (!text.includes("diff --git ")) return rawDiff(text);
  return parseDiff(text).map(file => diffFileHtml(file, toggle)).join("") || rawDiff(text);
}

function parseDiff(text) {
  const files = [];
  let file, hunk, oldLine = 0, newLine = 0;
  for (const line of text.split(/\r?\n/)) {
    if (line.startsWith("diff --git ")) {
      const match = /^diff --git\s+(?:a\/)?(.+?)\s+(?:b\/)?(.+)$/.exec(line);
      file = { old: cleanDiffPath(match?.[1] || "file"), new: cleanDiffPath(match?.[2] || "file"), hunks: [] };
      files.push(file);
      continue;
    }
    if (!file) continue;
    if (line.startsWith("--- ")) file.old = cleanDiffPath(line.slice(4));
    else if (line.startsWith("+++ ")) file.new = cleanDiffPath(line.slice(4));
    else if (line.startsWith("@@")) {
      const match = /@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)?/.exec(line);
      [oldLine, newLine] = match ? [Number(match[1]), Number(match[2])] : [0, 0];
      file.hunks.push(hunk = { content: line, changes: [] });
    } else if (hunk && [" ", "+", "-"].includes(line[0])) {
      const mark = line[0], body = line.slice(1);
      if (mark === " ") hunk.changes.push({ kind: "ctx", old: oldLine++, new: newLine++, text: body });
      if (mark === "+") hunk.changes.push({ kind: "add", new: newLine++, text: body });
      if (mark === "-") hunk.changes.push({ kind: "del", old: oldLine++, text: body });
    }
  }
  return files;
}

function cleanDiffPath(path) {
  const value = path.trim().split("\t")[0];
  return value === "/dev/null" ? value : value.replace(/^[ab]\//, "");
}

function diffFileHtml(file, toggle) {
  const changes = file.hunks.flatMap(h => h.changes);
  const deleted = changes.filter(c => c.kind === "del").length, added = changes.filter(c => c.kind === "add").length;
  const name = file.new !== "/dev/null" ? file.new : file.old;
  const action = toggle ? ` data-action="diff-context" title="Toggle full diff context"` : "";
  return `<div class="diff-file"><div class="diff-file-toolbar">
    <button class="diff-toggle" data-action="collapse" aria-label="Collapse file" title="Collapse file"></button>
    <span class="diff-name"${action}>${H(name)}</span>
    <span class="diff-counts"><span class="del">-${deleted}</span> <span class="add">+${added}</span></span>
  </div><div class="diff-file-body"><div class="split-diff">${file.hunks.map(diffHunkHtml).join("")}</div></div></div>`;
}

function diffHunkHtml(hunk) {
  return `<div class="diff-hunk"><span>${H(hunk.content)}</span><span>${H(hunk.content)}</span></div>` + pairedChanges(hunk.changes).map(diffRowHtml).join("");
}

function pairedChanges(changes) {
  const rows = [];
  for (let i = 0; i < changes.length;) {
    const change = changes[i];
    if (change.kind === "ctx") {
      rows.push([change.old, change.text, "ctx", change.new, change.text, "ctx"]);
      i++;
      continue;
    }
    const dels = [], adds = [];
    while (i < changes.length && changes[i].kind !== "ctx") (changes[i].kind === "add" ? adds : dels).push(changes[i++]);
    for (let j = 0; j < Math.max(dels.length, adds.length); j++) {
      const left = dels[j], right = adds[j];
      rows.push([left?.old || "", left?.text || "", left ? "del" : "empty", right?.new || "", right?.text || "", right ? "add" : "empty"]);
    }
  }
  return rows;
}

function diffRowHtml([leftNo, left, leftKind, rightNo, right, rightKind]) {
  return `<div class="diff-row"><span class="diff-line-no ${leftKind}">${H(leftNo)}</span><span class="diff-code ${leftKind}">${H(left)}</span><span class="diff-line-no ${rightKind}">${H(rightNo)}</span><span class="diff-code ${rightKind}">${H(right)}</span></div>`;
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
      if (row) updateUiSelection({ tabs: { ...S.tabs, [row.id]: target.dataset.tab } });
      break;
    }
    case "collapse": {
      const file = target.closest(".diff-file");
      const collapsed = file.classList.toggle("collapsed");
      target.classList.toggle("collapsed", collapsed);
      target.title = collapsed ? "Expand file" : "Collapse file";
      break;
    }
    case "diff-context": {
      const row = selectedPayloadView().row;
      const diff = row?.tabs?.diff;
      if (!diff) break;
      const box = target.closest(".diffbox");
      const full = box.dataset.full !== "1";
      box.dataset.full = full ? "1" : "0";
      box.innerHTML = diffHtml(full ? (diff.full || diff.compact) : diff.compact, hasFullDiff(row));
      break;
    }
  }
});

function applyData(raw) {
  const data = JSON.parse(raw), key = JSON.stringify(data);
  if (key === S.raw) return;
  updateUiSelection({ raw: key, data });
}

async function refresh() {
  try {
    applyData(await (await fetch(`ui.json?${Date.now()}`)).text());
  } catch (err) {
    UI.empty.hidden = false;
    UI.empty.innerHTML = waiting("UI error", err.message);
  }
}

function connect() {
  S.events?.close();
  S.events = new EventSource("events");
  S.events.onopen = () => refresh();
  S.events.onmessage = (event) => applyData(event.data);
  S.events.onerror = () => setTimeout(refresh, 1000);
}

addEventListener("beforeunload", () => {
  stopStream();
  S.events?.close();
});
refresh();
connect();
setInterval(refresh, 3000);
