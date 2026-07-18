(() => {
  "use strict";

  const API = "/api/v1/control";
  const LOG_COMPONENTS = new Set(["client", "gateway", "scheduler", "trainer", "sampler", "timeslicer"]);
  const VOLATILE_FIELDS = new Set(["age_seconds", "elapsed_seconds", "generated_at", "latency_seconds", "timestamp", "updated_at"]);
  const GOOD = new Set(["pass", "healthy", "ready", "running", "completed", "succeeded"]);
  const WARN = new Set(["warn", "warning", "degraded", "queued", "pending", "starting", "waiting", "not_ready", "not-ready"]);
  const BAD = new Set(["fail", "failed", "error", "crashed", "unhealthy", "unavailable"]);
  const SIMULATION_KEY = "openrl-control-simulation";
  const THEME_KEY = "openrl-control-theme";
  const simulationParam = new URLSearchParams(location.hash.split("?", 2)[1] || "").get("simulate");
  let savedSimulation = null;
  try { savedSimulation = localStorage.getItem(SIMULATION_KEY); } catch { /* Storage can be unavailable. */ }

  const state = {
    liveRuns: [],
    liveCluster: null,
    runs: [],
    cluster: null,
    doctor: null,
    errors: {},
    loading: true,
    refreshing: false,
    updated: null,
    simulationEnabled: simulationParam !== null ? simulationParam !== "0" : savedSimulation === "on",
    stoppingRuns: new Set(),
    stopRequestedRuns: new Set(),
    actionErrors: {},
    entities: new Map(),
    selectedEntity: null,
    log: null,
  };

  const main = document.querySelector("#main");
  const inspector = document.querySelector("#inspector");
  const inspectBody = document.querySelector("#inspect-body");
  const themeButton = document.querySelector("#theme");

  const list = (value) => Array.isArray(value) ? value : [];
  const object = (value) => value && typeof value === "object" && !Array.isArray(value) ? value : {};
  const escapeHtml = (value) => String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
  const label = (value) => String(value || "unknown").replaceAll("_", " ");

  function setTheme(theme, remember = true) {
    const value = theme === "dark" ? "dark" : "light";
    document.documentElement.dataset.theme = value;
    themeButton.textContent = value === "dark" ? "☀" : "☾";
    themeButton.setAttribute("aria-label", `Switch to ${value === "dark" ? "light" : "dark"} theme`);
    if (remember) try { localStorage.setItem(THEME_KEY, value); } catch { /* Storage can be unavailable. */ }
  }

  let initialTheme = null;
  try { initialTheme = localStorage.getItem(THEME_KEY); } catch { /* Storage can be unavailable. */ }
  setTheme(initialTheme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"), false);

  function tone(status) {
    const value = String(status || "unknown").toLowerCase();
    if (GOOD.has(value)) return "good";
    if (WARN.has(value)) return "warn";
    if (BAD.has(value)) return "bad";
    return "unknown";
  }

  function elapsed(value) {
    const seconds = Number(value);
    if (!Number.isFinite(seconds)) return "not reported";
    if (seconds < 60) return `${Math.max(0, Math.floor(seconds))}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h`;
    return `${Math.floor(seconds / 86400)}d`;
  }

  function roleOf(item) {
    const raw = String(item?.role || object(item?.labels).app || item?.id || item?.name || "component").toLowerCase();
    if (raw.includes("gateway")) return "gateway";
    if (raw.includes("timeslicer") || raw.includes("time-slicer")) return "timeslicer";
    if (raw.includes("redis") || raw.includes("store")) return "store";
    if (raw.includes("trainer")) return "trainer";
    if (raw.includes("sampler") || raw.includes("vllm")) return "sampler";
    if (raw.includes("scheduler")) return "scheduler";
    if (raw.includes("client")) return "client";
    return raw;
  }

  function doctorCheck(name) {
    return list(state.doctor?.checks).find((check) => check.name === name);
  }

  function checkStatus(check) {
    return check?.status === "pass" ? "ready" : check?.status === "warn" ? "degraded" : check?.status === "fail" ? "failed" : "unknown";
  }

  function currentView() {
    const hash = location.hash.toLowerCase();
    if (hash.startsWith("#/runs")) return "runs";
    if (hash.startsWith("#/doctor")) return "doctor";
    return "topology";
  }

  function applySimulation() {
    const demo = window.OpenRLDemo;
    state.runs = state.simulationEnabled && demo ? [...demo.runs(), ...state.liveRuns] : state.liveRuns;
    state.cluster = state.simulationEnabled && demo ? demo.augmentCluster(state.liveCluster) : state.liveCluster;
  }

  function dataSignature() {
    return JSON.stringify(
      {
        cluster: state.cluster,
        doctor: state.doctor,
        errors: state.errors,
        runs: state.runs,
        simulationEnabled: state.simulationEnabled,
      },
      (key, value) => VOLATILE_FIELDS.has(key) ? undefined : value,
    );
  }

  function setSimulation(enabled) {
    state.simulationEnabled = enabled;
    try { localStorage.setItem(SIMULATION_KEY, enabled ? "on" : "off"); } catch { /* Storage can be unavailable. */ }
    state.selectedEntity = null;
    state.log = null;
    inspector.hidden = true;
    applySimulation();
    render();
  }

  function contextRunId() {
    return state.runs[0]?.id || null;
  }

  async function api(path, options = {}) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 9000);
    try {
      const response = await fetch(`${API}${path}`, {
        ...options,
        cache: "no-store",
        headers: { Accept: "application/json", ...options.headers },
        signal: controller.signal,
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || payload.error || `${response.status} ${response.statusText}`);
      return payload;
    } finally {
      clearTimeout(timeout);
    }
  }

  function remember(entity) {
    state.entities.set(entity.key, entity);
    return entity;
  }

  function markSelectedCard(key) {
    document.querySelectorAll(".card.selected").forEach((item) => item.classList.remove("selected"));
    document.querySelectorAll(".card[data-entity]").forEach((item) => {
      if (item.dataset.entity === key) item.classList.add("selected");
    });
  }

  function placement(entity) {
    if (entity.node) return entity.node;
    if (entity.kind === "component" && !entity.pod_name) {
      return ["trainer", "sampler"].includes(entity.role) ? "no worker reported" : "logical";
    }
    if (entity.reported === false) return "not reported";
    if (entity.pod_name) return "Unscheduled";
    return "logical";
  }

  function card(entity) {
    remember(entity);
    const detail = entity.reason || entity.message;
    const name = entity.cardName || entity.name;
    return `
      <button class="card${state.selectedEntity?.key === entity.key ? " selected" : ""}" type="button" data-entity="${escapeHtml(entity.key)}" title="${escapeHtml(detail || name)}">
        <span class="card-top"><span class="dot ${tone(entity.status)}"></span><strong>${escapeHtml(name)}</strong>${entity.simulated ? '<span class="sim-badge">SIM</span>' : ""}</span>
        <small>${escapeHtml(placement(entity))}</small>
        ${detail && tone(entity.status) !== "good" ? `<small class="problem">${escapeHtml(detail)}</small>` : ""}
      </button>`;
  }

  function platformEntities() {
    const pods = list(state.cluster?.pods).filter((pod) => !pod.model_id && !["trainer", "sampler"].includes(roleOf(pod)));
    const entities = pods.map((pod) => {
      const role = roleOf(pod);
      const podName = pod.pod_name || pod.name;
      const serviceName = role === "gateway" ? "Gateway" : role === "timeslicer" ? "Time slicer" : role === "store" ? "Redis" : podName;
      return {
        key: `platform:${pod.namespace || "default"}/${podName}`,
        kind: "platform",
        name: serviceName,
        cardName: podName && podName !== serviceName ? `${serviceName} · ${podName}` : serviceName,
        role,
        status: pod.status,
        phase: pod.phase,
        reason: pod.reason,
        message: pod.message,
        node: pod.node,
        pod_name: podName,
        namespace: pod.namespace,
        restarts: pod.restarts,
        image: pod.image,
        run_id: LOG_COMPONENTS.has(role) ? contextRunId() : null,
        reported: true,
        serviceSelected: ["gateway", "timeslicer"].includes(role),
        simulated: Boolean(pod.simulated),
        raw: pod,
      };
    });

    const localNode = state.cluster?.mode === "local" && list(state.cluster?.nodes).length === 1 ? state.cluster.nodes[0].name : null;
    const ensure = (role, name, checkName, message) => {
      if (entities.some((entity) => entity.role === role)) return;
      const check = doctorCheck(checkName);
      entities.push({
        key: `platform:logical/${role}`,
        kind: "platform",
        name,
        role,
        status: check ? checkStatus(check) : "unknown",
        phase: check?.status || "not reported",
        message: check?.message || message,
        node: role === "timeslicer" ? null : localNode,
        run_id: LOG_COMPONENTS.has(role) ? contextRunId() : null,
        reported: false,
        simulated: false,
        raw: check || {},
      });
    };
    ensure("gateway", "Gateway", "gateway", "Gateway pod not reported by this control plane");
    ensure("store", "Store", "store", "Store pod not reported by this control plane");
    ensure("timeslicer", "Time slicer", "timeslicer", "Time-slicer pod not reported by this control plane");
    return entities;
  }

  function runEntity(run) {
    return {
      key: `run:${run.id}`,
      kind: "run",
      name: run.name || run.id,
      role: "run",
      status: run.status,
      phase: run.phase,
      message: run.message,
      run_id: run.id,
      simulated: Boolean(run.simulated),
      raw: run,
    };
  }

  function componentEntities(run) {
    const workerPods = list(state.cluster?.pods).filter((pod) => pod.model_id === run.id);
    const localNode = state.cluster?.mode === "local" && list(state.cluster?.nodes).length === 1 ? state.cluster.nodes[0].name : null;
    const used = new Set();
    const components = list(run.components).map((component) => {
      const role = roleOf(component);
      const podIndex = workerPods.findIndex((pod, index) => !used.has(index) && roleOf(pod) === role);
      const pod = podIndex >= 0 ? workerPods[podIndex] : null;
      if (podIndex >= 0) used.add(podIndex);
      const podName = component.pod_name || component.name || pod?.pod_name || pod?.name || (component.pid ? `process ${component.pid}` : null);
      return {
        key: `component:${run.id}:${component.id || component.role || role}`,
        kind: "component",
        name: label(component.role || component.id || role),
        cardName: `${run.name || run.id} · ${label(component.role || component.id || role)}`,
        role,
        status: component.status || pod?.status,
        phase: component.phase || pod?.phase,
        reason: component.reason || pod?.reason,
        message: component.message || pod?.message,
        node: component.node || pod?.node || ((pod || component.pid) ? localNode : null),
        pod_name: podName,
        namespace: component.namespace || pod?.namespace,
        restarts: component.restarts ?? pod?.restarts,
        image: component.image || pod?.image,
        run_id: run.id,
        reported: Boolean(podName || pod),
        simulated: Boolean(run.simulated || component.simulated || pod?.simulated),
        raw: { ...pod, ...component },
      };
    });
    workerPods.forEach((pod, index) => {
      if (used.has(index)) return;
      const role = roleOf(pod);
      components.push({
        key: `component:${run.id}:${pod.pod_name || pod.name}`,
        kind: "component",
        name: label(role),
        cardName: `${run.name || run.id} · ${label(role)}`,
        role,
        status: pod.status,
        phase: pod.phase,
        reason: pod.reason,
        message: pod.message,
        node: pod.node || localNode,
        pod_name: pod.pod_name || pod.name,
        namespace: pod.namespace,
        restarts: pod.restarts,
        image: pod.image,
        run_id: run.id,
        reported: true,
        simulated: Boolean(run.simulated || pod.simulated),
        raw: pod,
      });
    });
    return components;
  }

  function nodeGpu(node) {
    const capacity = object(node.capacity);
    const allocatable = object(node.allocatable);
    const key = [...new Set([...Object.keys(capacity), ...Object.keys(allocatable)])].find((name) => name.toLowerCase().includes("gpu"));
    return key ? `${key} ${allocatable[key] ?? capacity[key] ?? "not reported"}` : "GPU not advertised";
  }

  function nodeEntity(node) {
    return {
      key: `node:${node.name}`,
      kind: "node",
      name: node.name || "Unnamed node",
      role: "node",
      status: node.status || (node.ready ? "ready" : "not ready"),
      phase: list(node.roles).join(", ") || "worker",
      message: list(node.conditions).find((condition) => condition.type === "Ready" && condition.status !== "True")?.message,
      node: node.name,
      simulated: Boolean(node.simulated),
      raw: node,
    };
  }

  function renderRunsPage() {
    const rows = state.runs.map((run) => {
      const queues = object(run.queue);
      const stopping = state.stoppingRuns.has(run.id);
      const stopRequested = state.stopRequestedRuns.has(run.id);
      const actionError = state.actionErrors[run.id];
      const tracker = safeTrackerHref(run);
      const runStates = [...new Set([run.status, run.phase].filter(Boolean))];
      return `
      <article class="run-item">
        <div class="run-identity">
          <span class="dot ${tone(run.status)}"></span>
          <div><strong>${run.simulated ? '<span class="sim-badge">SIM</span> ' : ""}${escapeHtml(run.name || run.id)}</strong><small>${escapeHtml(run.id)} · ${escapeHtml(run.base_model || "model not reported")}</small></div>
        </div>
        <div class="run-facts">${runStates.map((value) => `<span>${escapeHtml(label(value))}</span>`).join("")}<span>train ${Math.max(0, Number(queues.training) || 0)}</span><span>sample ${Math.max(0, Number(queues.sampling) || 0)}</span></div>
        <div class="run-actions">
          ${tracker ? `<a class="tracker-link" href="${escapeHtml(tracker)}" target="_blank" rel="noopener noreferrer">W&amp;B ↗</a>` : ""}
        ${(run.can_stop === true && !stopRequested) || stopping ? `<button class="stop-run" data-action="stop-run" data-run-id="${escapeHtml(run.id)}" type="button"${stopping ? " disabled aria-busy=\"true\"" : ""} aria-label="Stop ${escapeHtml(run.name || run.id)}">${stopping ? "Stopping…" : run.simulated ? "Stop demo" : "Stop"}</button>` : ""}
        </div>
        ${actionError ? `<small class="run-action-error" role="alert">${escapeHtml(actionError)}</small>` : ""}
      </article>`;
    }).join("");
    main.innerHTML = `
      <section class="page-head runs-head"><div><h1>Runs</h1><p>Launch state and worker queues; experiment tracking stays in W&amp;B</p></div><span class="summary">${state.runs.length} total</span></section>
      ${banner()}
      <section class="runs-panel">${rows || '<div class="empty">No runs reported</div>'}</section>`;
  }

  function banner() {
    const errors = Object.entries(state.errors);
    const notices = [];
    if (state.simulationEnabled) notices.push('<div class="banner simulation"><strong>SIMULATION</strong> · fictional nodes and jobs overlaid on live cluster data</div>');
    if (errors.length) notices.push(`<div class="banner"><strong>Partial data</strong> · ${errors.map(([name, message]) => `${escapeHtml(name)}: ${escapeHtml(message)}`).join(" · ")}</div>`);
    return notices.join("");
  }

  function renderTopology() {
    const runs = state.runs;
    const platform = platformEntities();
    const components = [];
    const emptyRuns = [];
    runs.forEach((run) => {
      const items = componentEntities(run).filter((item) => item.reported || ["client", "trainer", "sampler"].includes(item.role));
      if (items.length) components.push(...items);
      else emptyRuns.push({ ...runEntity(run), cardName: `${run.name || run.id} · no components` });
    });

    const nodeSections = list(state.cluster?.nodes).map((node) => {
      const entity = remember(nodeEntity(node));
      const platformHere = platform.filter((item) => item.node === node.name);
      const componentsHere = components.filter((item) => item.node === node.name);
      return `
        <section class="node-group">
          <div class="node-head">
            <button type="button" data-entity="${escapeHtml(entity.key)}"><span class="dot ${tone(entity.status)}"></span><strong>${escapeHtml(entity.name)}</strong>${entity.simulated ? '<span class="sim-badge">SIM</span>' : ""}</button>
            <span>${escapeHtml(entity.phase)} · ${escapeHtml(nodeGpu(node))} · ${node.pod_count ?? 0} pods</span>
          </div>
          <div class="node-columns">
            <div class="node-lane"><h3>Platform</h3><div class="row">${platformHere.length ? platformHere.map(card).join("") : '<span class="muted">No platform pods reported</span>'}</div></div>
            <div class="node-lane"><h3>Run workloads</h3><div class="row">${componentsHere.length ? componentsHere.map(card).join("") : '<span class="muted">No run workers reported</span>'}</div></div>
          </div>
        </section>`;
    }).join("");

    const unplaced = [...platform.filter((item) => !item.node), ...components.filter((item) => !item.node), ...emptyRuns];
    const unplacedSection = unplaced.length ? `
      <section class="node-group unplaced">
        <div class="node-head"><strong>Unplaced / logical</strong><span>${unplaced.length} object${unplaced.length === 1 ? "" : "s"}</span></div>
        <div class="row">${unplaced.map(card).join("")}</div>
      </section>` : "";

    const summary = object(state.cluster?.summary);
    main.innerHTML = `
      <section class="page-head topology-head"><div><h1>Physical topology</h1><p>Platform services and run workers grouped by reported node</p></div><span class="summary">${summary.ready_nodes ?? 0}/${summary.nodes ?? 0} nodes ready</span></section>
      ${banner()}
      <section class="topology">
        <div class="node-grid">${nodeSections || '<div class="empty">No physical nodes reported</div>'}${unplacedSection}</div>
      </section>`;
  }

  function safeTrackerHref(run) {
    if (typeof run?.tracker_url !== "string" || run.tracker_url.length > 2048) return null;
    try {
      const url = new URL(run.tracker_url);
      return ["http:", "https:"].includes(url.protocol) && !url.username && !url.password ? url.href : null;
    } catch {
      return null;
    }
  }

  function renderDoctor() {
    const checks = list(state.doctor?.checks);
    const clusterErrors = list(state.cluster?.errors).map((message) => ({ name: "cluster", status: "fail", message }));
    main.innerHTML = `
      <section class="page-head"><div><h1>Doctor</h1><p>Gateway, store, and cluster checks</p></div><span class="summary">${escapeHtml(label(state.doctor?.status))}</span></section>
      ${banner()}
      <section class="panel">${[...checks, ...clusterErrors].length ? [...checks, ...clusterErrors].map((check) => `<div class="item"><div class="item-head"><span class="dot ${tone(check.status)}"></span><strong>${escapeHtml(check.name)}</strong><span class="state">${escapeHtml(label(check.status))}</span></div><p>${escapeHtml(check.message || "No message reported")}</p></div>`).join("") : '<div class="empty">No diagnostics reported</div>'}</section>`;
  }

  function facts(entity) {
    const raw = object(entity.raw);
    const source = entity.simulated ? [["Source", "simulation"]] : [];
    if (entity.kind === "run") {
      const queues = object(raw.queue);
      return [
        ...source,
        ["Model", raw.base_model || "not reported"],
        ["Phase", label(raw.phase)],
        ["Elapsed", elapsed(raw.elapsed_seconds)],
        ["Training queue", queues.training ?? 0],
        ["Sampling queue", queues.sampling ?? 0],
      ];
    }
    if (entity.kind === "node") {
      return [
        ...source,
        ["Roles", list(raw.roles).join(", ") || "worker"],
        ["Pods", raw.pod_count ?? "not reported"],
        ["GPU", nodeGpu(raw)],
        ["Capacity", JSON.stringify(object(raw.capacity))],
        ["Allocatable", JSON.stringify(object(raw.allocatable))],
        ["Taints", list(raw.taints).length ? JSON.stringify(raw.taints) : "none"],
      ];
    }
    return [
      ...source,
      ["Role", label(entity.role)],
      ["Phase", label(entity.phase)],
      ["Pod", entity.pod_name || "not reported"],
      ["Namespace", entity.namespace || "not reported"],
      ["Node", entity.node || (entity.reported === false ? "not reported" : "Unscheduled/logical")],
      ["Restarts", entity.restarts ?? "not reported"],
      ["Image", entity.image || "not reported"],
    ];
  }

  function canLoadLogs(entity) {
    return Boolean(logRequest(entity));
  }

  function logRequest(entity) {
    if (!entity || entity.simulated) return null;
    if (entity.kind === "platform" && entity.pod_name) {
      return {
        target: `pod:${entity.namespace || "default"}/${entity.pod_name}`,
        path: `/pods/${encodeURIComponent(entity.pod_name)}/logs?tail=500`,
      };
    }
    if (LOG_COMPONENTS.has(entity.role) && entity.run_id) {
      return {
        target: `run:${entity.run_id}/${entity.role}`,
        path: `/runs/${encodeURIComponent(entity.run_id)}/logs?component=${encodeURIComponent(entity.role)}&tail=500`,
      };
    }
    return null;
  }

  function renderLog(entity) {
    if (entity.simulated) return '<p class="muted">Logs are unavailable for simulated components.</p>';
    const request = logRequest(entity);
    if (!request && !LOG_COMPONENTS.has(entity.role)) return '<p class="muted">Logs are unavailable because no pod was reported for this component.</p>';
    if (!request) return '<p class="muted">Logs need either a pod or run context.</p>';
    if (!state.log || state.log.key !== entity.key || state.log.target !== request.target) return '<p class="muted">Loading logs…</p>';
    if (state.log.loading && !state.log.data) return '<p class="muted">Loading logs…</p>';
    if (state.log.error) return `<p class="error">${escapeHtml(state.log.error)}</p><button type="button" data-action="refresh-log">Retry</button>`;
    const payload = object(state.log.data);
    let lines = list(payload.lines);
    if (!lines.length && payload.logs) lines = String(payload.logs).split("\n").map((message) => ({ message }));
    const logRows = lines.map((line) => {
      if (typeof line === "string") return `<div class="log-row"><time>—</time><code>${escapeHtml(line)}</code></div>`;
      const timestamp = line.timestamp ? String(line.timestamp).replace("T", " ").replace(/(\.\d{3})\d+/, "$1").replace(/Z$/, "") : "—";
      const message = line.message ?? line.text ?? JSON.stringify(line);
      return `<div class="log-row"><time>${escapeHtml(timestamp)}</time><code>${escapeHtml(message)}</code></div>`;
    }).join("");
    return `
      <div class="log-toolbar">
        <span>${escapeHtml(payload.source || "unknown source")}${payload.pod_name ? ` · ${escapeHtml(payload.pod_name)}` : ""}${payload.error ? ` · ${escapeHtml(payload.error)}` : ""}</span>
        <button type="button" data-action="refresh-log">Refresh</button>
      </div>
      <div class="log-table-head"><span>Time</span><span>Message</span></div>
      ${logRows ? `<div class="logs log-scroll">${logRows}</div>` : '<div class="log-empty"><strong>No logs yet</strong><span>This pod has not written anything in the last 500 lines.</span></div>'}`;
  }

  function renderInspector() {
    const liveEntity = state.entities.get(state.selectedEntity?.key);
    if (state.selectedEntity && currentView() === "topology" && !liveEntity) {
      state.selectedEntity = null;
      state.log = null;
    }
    const entity = liveEntity || state.selectedEntity;
    if (!entity) {
      inspector.hidden = true;
      return;
    }
    state.selectedEntity = entity;
    const hasLogView = canLoadLogs(entity);
    const tab = hasLogView ? "logs" : "details";
    const sameView = inspectBody.dataset.entityKey === entity.key && inspectBody.dataset.tab === tab;
    const previousLog = sameView ? inspectBody.querySelector(".log-scroll") : null;
    const previousView = sameView ? inspectBody.querySelector(".inspector-view") : null;
    const previousLogTop = previousLog?.scrollTop || 0;
    const followLogTail = !previousLog || previousLog.scrollHeight - previousLog.clientHeight - previousLog.scrollTop < 24;
    const previousViewTop = previousView?.scrollTop || 0;
    inspector.hidden = false;
    const rows = facts(entity).map(([name, value]) => `<dt>${escapeHtml(name)}</dt><dd>${escapeHtml(value)}</dd>`).join("");
    const run = entity.kind === "run" ? entity.raw : null;
    const tracker = safeTrackerHref(run);
    const detailsView = `
      <div class="inspector-view details-view">
        <section class="detail-section"><h3>Runtime</h3><dl class="facts">${rows}</dl></section>
        ${entity.reason || entity.message ? `<section class="detail-section"><h3>Message</h3><div class="message">${escapeHtml(entity.reason || entity.message)}</div>${entity.reason && entity.message && entity.reason !== entity.message ? `<p class="message-note">${escapeHtml(entity.message)}</p>` : ""}</section>` : ""}
        ${tracker ? `<a class="tracker-link" href="${escapeHtml(tracker)}" target="_blank" rel="noopener noreferrer">Open tracker ↗</a>` : ""}
      </div>`;
    const logView = `
      <div class="inspector-view logs-view">
        ${entity.serviceSelected && !entity.pod_name ? '<p class="muted log-note">Logs are selected by service; the returned pod is shown below.</p>' : ""}
        ${renderLog(entity)}
      </div>`;
    inspectBody.innerHTML = `
      <div class="inspector-head">
        <span class="inspector-kicker">${escapeHtml(label(entity.kind))} · ${escapeHtml(label(entity.role))}${entity.simulated ? ' · <span class="sim-badge">SIM</span>' : ""}</span>
        <div class="inspector-title"><h2>${escapeHtml(entity.name)}</h2><span class="status-pill ${tone(entity.status)}"><span class="dot ${tone(entity.status)}"></span>${escapeHtml(label(entity.status))}</span></div>
      </div>
      ${tab === "logs" ? logView : detailsView}`;
    inspectBody.dataset.entityKey = entity.key;
    inspectBody.dataset.tab = tab;
    const nextView = inspectBody.querySelector(".inspector-view");
    if (nextView) nextView.scrollTop = previousViewTop;
    const nextLog = inspectBody.querySelector(".log-scroll");
    if (nextLog) nextLog.scrollTop = followLogTail ? nextLog.scrollHeight : previousLogTop;
  }

  async function loadLogs() {
    const entity = state.selectedEntity;
    const request = logRequest(entity);
    if (!entity || !request || state.log?.loading) return;
    const key = entity.key;
    const target = request.target;
    const previousData = state.log?.key === key && state.log?.target === target ? state.log.data : null;
    state.log = { key, target, loading: true, data: previousData, error: null };
    if (!previousData) renderInspector();
    try {
      const data = await api(request.path);
      if (state.selectedEntity?.key !== key || logRequest(state.selectedEntity)?.target !== target) return;
      state.log = { key, target, loading: false, data, error: null };
      if (!previousData || JSON.stringify(previousData) !== JSON.stringify(data)) renderInspector();
    } catch (error) {
      if (state.selectedEntity?.key !== key || logRequest(state.selectedEntity)?.target !== target) return;
      state.log = { key, target, loading: false, data: null, error: error.message };
      renderInspector();
    }
  }

  function selectEntity(entity) {
    state.selectedEntity = entity;
    markSelectedCard(entity.key);
    state.log = null;
    inspectBody.dataset.entityKey = "";
    renderInspector();
    if (canLoadLogs(entity)) loadLogs();
  }

  function closeInspector() {
    inspector.hidden = true;
    state.selectedEntity = null;
    state.log = null;
    markSelectedCard(null);
  }

  async function stopRun(runId) {
    const run = state.runs.find((item) => item.id === runId);
    if (!run || run.can_stop !== true || state.stoppingRuns.has(runId)) return;
    const name = run.name || run.id;
    if (run.simulated) {
      if (!window.confirm(`Stop ${name}? This changes only the fictional simulation overlay.`)) return;
      window.OpenRLDemo?.stop(runId);
      applySimulation();
      render();
      return;
    }
    if (!window.confirm(`Stop ${name}? Trainer and sampler workers will be asked to exit.`)) return;

    delete state.actionErrors[runId];
    state.stoppingRuns.add(runId);
    run.can_stop = false;
    render();
    try {
      await api(`/runs/${encodeURIComponent(runId)}/stop`, { method: "POST" });
      state.stopRequestedRuns.add(runId);
      const current = state.runs.find((item) => item.id === runId);
      if (current) current.can_stop = false;
      await refresh();
    } catch (error) {
      const current = state.runs.find((item) => item.id === runId);
      if (current) current.can_stop = true;
      state.actionErrors[runId] = error.name === "AbortError" ? "Stop request timed out" : error.message;
    } finally {
      state.stoppingRuns.delete(runId);
      render();
    }
  }

  function renderChrome() {
    const failures = Object.keys(state.errors).length;
    const hasData = Boolean(state.cluster || state.doctor || state.runs.length);
    const connection = failures ? (hasData ? "partial" : "offline") : "live";
    document.querySelector("#connection").textContent = `${connection}${state.updated ? ` · ${new Date(state.updated).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}` : ""}`;
    document.querySelector("#refresh").disabled = state.refreshing;
    const simulation = document.querySelector("#simulation");
    simulation.textContent = `Simulation ${state.simulationEnabled ? "on" : "off"}`;
    simulation.setAttribute("aria-pressed", String(state.simulationEnabled));
    simulation.classList.toggle("active", state.simulationEnabled);
  }

  function render() {
    state.entities = new Map();
    document.querySelectorAll("nav a").forEach((link) => link.classList.toggle("active", link.dataset.view === currentView()));
    if (state.loading) {
      main.innerHTML = '<div class="loading">Connecting…</div>';
    } else if (currentView() === "doctor") {
      renderDoctor();
    } else if (currentView() === "runs") {
      renderRunsPage();
    } else {
      renderTopology();
    }
    renderChrome();
    if (state.selectedEntity) renderInspector();
  }

  async function refresh() {
    if (state.refreshing) return;
    const before = dataSignature();
    const wasLoading = state.loading;
    state.refreshing = true;
    document.querySelector("#refresh").disabled = true;
    const endpoints = [
      ["runs", "/runs", (payload) => {
        const runs = list(payload.runs);
        state.stopRequestedRuns.forEach((runId) => {
          const run = runs.find((item) => item.id === runId);
          if (!run || run.can_stop !== true) state.stopRequestedRuns.delete(runId);
          else run.can_stop = false;
        });
        state.liveRuns = runs;
        applySimulation();
      }],
      ["cluster", "/cluster", (payload) => {
        state.liveCluster = payload;
        applySimulation();
      }],
      ["doctor", "/doctor", (payload) => { state.doctor = payload; }],
    ];
    try {
      await Promise.all(endpoints.map(async ([name, path, apply]) => {
        try {
          apply(await api(path));
          delete state.errors[name];
        } catch (error) {
          state.errors[name] = error.name === "AbortError" ? "request timed out" : error.message;
        }
      }));
    } finally {
      state.loading = false;
      state.updated = Date.now();
      state.refreshing = false;
      if (wasLoading || before !== dataSignature()) render();
      else renderChrome();
    }
  }

  document.addEventListener("click", (event) => {
    const stopButton = event.target.closest('[data-action="stop-run"]');
    if (stopButton) {
      stopRun(stopButton.dataset.runId);
      return;
    }
    const target = event.target.closest("[data-entity]");
    if (target) {
      const entity = state.entities.get(target.dataset.entity);
      if (entity) selectEntity(entity);
      return;
    }
    if (event.target.closest(".close")) {
      closeInspector();
      return;
    }
    if (event.target.closest('[data-action="refresh-log"]')) {
      loadLogs();
      return;
    }
    if (!inspector.hidden && !event.target.closest("#inspector")) closeInspector();
  });

  document.querySelector("#refresh").addEventListener("click", refresh);
  themeButton.addEventListener("click", () => setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"));
  document.querySelector("#simulation").addEventListener("click", () => setSimulation(!state.simulationEnabled));
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !inspector.hidden) document.querySelector(".close").click();
  });
  window.addEventListener("hashchange", () => {
    if (location.hash.toLowerCase().startsWith("#/metrics")) history.replaceState(null, "", "#/topology");
    state.selectedEntity = null;
    state.log = null;
    inspector.hidden = true;
    markSelectedCard(null);
    render();
  });

  if (!location.hash || location.hash.toLowerCase().startsWith("#/metrics")) history.replaceState(null, "", "#/topology");
  render();
  refresh();
  setInterval(() => {
    if (!document.hidden) {
      refresh();
      if (canLoadLogs(state.selectedEntity || {})) loadLogs();
    }
  }, 5000);
})();
