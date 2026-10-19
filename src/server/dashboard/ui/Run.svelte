<script>
  import { createQuery, inspectUrl, now, viewLink, saved, MAX_WINDOW } from "./state.svelte.js";
  import { nodeTime } from "./node-time.js";
  import TimeControl from "./TimeControl.svelte";
  import Status from "./Status.svelte";
  import Chart from "./Chart.svelte";
  import Activity from "./Activity.svelte";
  import Logs from "./Logs.svelte";
  import CopyLink from "./CopyLink.svelte";
  import Configuration from "./Configuration.svelte";
  import { duration, shortNodeName, KINDS, runDescription, isFinished, finishedAt } from "./format.js";
  let { inventory, view } = $props();
  // The same window and gestures as the Nodes page; the query window holds still while a gesture runs.
  let frozen = $state(null);
  const observed = $derived(now(inventory));
  const range = $derived.by(() => {
    const duration = Math.max(60, Math.min(MAX_WINDOW, Math.round(view.duration || 1800)));
    const end = Math.max(observed - MAX_WINDOW + duration, Math.min(observed, view.end ?? observed));
    return { start: end - duration, now: end, live: view.end === null };
  });
  const queryRange = $derived(frozen || range);
  const validEvent = $derived(view.event === null || (view.event >= range.start && view.event <= range.now));
  $effect(() => { if (!validEvent) view.event = null; });
  const url = $derived(inspectUrl("activity", { since: new Date(queryRange.start * 1000).toISOString(), until: new Date(queryRange.now * 1000).toISOString(), run_id: view.id }));
  const detail = createQuery(() => url, { poll: 15000, pollWhen: () => view.end === null && !frozen, key: () => `${view.id} ${view.duration} ${view.end} ${frozen?.start ?? ""}` });
  const run = $derived((detail.status !== 404 ? detail.data?.run : null) || inventory.runs.find((r) => r.run_id === view.id));
  const error = $derived(inventory.store_error || inventory.processes_error || detail.error);
  const title = $derived(run ? [(run.model || "Run").split("/").at(-1), run.run_id.slice(0, 8), KINDS[run.fine_tuning_type]].filter(Boolean).join(" · ") : "Run");
  const description = $derived(run ? runDescription(run, title) : "");
  const samples = $derived((detail.data?.samples || []).filter((s) => s.at >= range.start && s.at <= range.now));
  const roles = $derived([...new Set(samples.map((s) => s.role || "process"))]);
  const names = $derived([...new Set(samples.flatMap((s) => Object.entries(s.metrics || {}).filter(([, v]) => Number.isFinite(v)).map(([k]) => k)))].slice(0, 12));
  const metricStatus = $derived(detail.error ? `${detail.error}${samples.length ? " · Showing previously fetched metrics" : ""}` : detail.pending ? "Updating…" : "");
  const metricsOpen = () => view.tab === "metrics" && `${view.id} ${view.duration} ${view.end}`;
  // A shared pool's load is every run's, so it is charted on Nodes rather than here.
  const vllm = createQuery(() => run?.shared_runtime ? null : inspectUrl("series", { run_id: view.id, since: new Date(range.start * 1000).toISOString(), until: new Date(range.now * 1000).toISOString() }), { poll: 15000, pollWhen: () => view.end === null, key: metricsOpen });
  // The vLLM series the inspection API returns for a run's samplers, as vLLM's and llm-d's dashboards chart them.
  const SAMPLER_CHARTS = [["generation_tokens_per_second", "Generated tokens", "tok/s"], ["prompt_tokens_per_second", "Prompt tokens", "tok/s"], ["requests_running", "Running requests", ""], ["requests_waiting", "Waiting requests", ""],
    ["kv_cache_usage", "KV cache use", "%", 100], ["preemptions_per_second", "Preemptions", "/s"], ["time_to_first_token_p99_seconds", "Time to first token p99", "s"], ["request_latency_p99_seconds", "Request latency p99", "s"]];
  // What the run's training client reported per step, such as reward and eval pass rate, charted by step.
  const reported = createQuery(() => inspectUrl("run_metrics", { run_id: view.id }), { poll: 30000, pollWhen: () => view.end === null, key: () => view.tab === "metrics" && view.id });
  const reportedRows = $derived(reported.data?.rows || []);
  const reportedNames = $derived([...new Set(reportedRows.flatMap((row) => Object.keys(row.metrics)))]);
  const REPORTED_CHARTS = [["env/all/reward/total", "Reward"], ["test/env/all/reward/total", "Held-out reward"], ["env/all/lab/criteria_pass_fraction", "Criteria pass rate"],
    ["test/env/all/lab/criteria_pass_fraction", "Held-out criteria pass rate"], ["env/all/lab/no_output", "Episodes without output"], ["test/env/all/lab/no_output", "Held-out episodes without output"],
    ["optim/entropy", "Entropy"], ["optim/kl_sample_train_v2", "Sampler-trainer KL"], ["loss_per_token", "Loss per token"], ["train_mean_nll", "Train NLL"], ["time/total", "Step time", "s"]];
  let reportedFilter = $state("");
  const reportedShown = $derived(reportedFilter ? reportedNames.filter((name) => name.includes(reportedFilter)).slice(0, 12).map((name) => [name, name]) : REPORTED_CHARTS.filter(([name]) => reportedNames.includes(name)));
  // A finished run opens on its own lifetime rather than the last 30 minutes, unless the link names a window.
  const finished = $derived(isFinished(run));
  let opened = null;
  $effect(() => {
    if (!run || !finished || opened === run.run_id) return;
    opened = run.run_id;
    const params = new URLSearchParams(location.hash.split("?")[1] || "");
    const stop = finishedAt(run);
    if (params.has("end") || params.has("duration") || !Number.isFinite(stop)) return;
    const span = Math.max(0, stop - (run.created_at || stop));
    view.duration = Math.max(1800, Math.min(MAX_WINDOW, Math.round(span * 1.05)));
    view.end = Math.min(observed, Math.round(stop + Math.max(60, span * 0.025)));
  });
  const link = (tab, changes = {}) => viewLink(`run/${encodeURIComponent(view.id)}/${tab}`, { duration: view.duration, end: view.end, back: view.back, event: view.event, q: view.q, pod: view.pod, ...changes });
</script>

<p class="overview-back"><a href={view.back || saved.overview}>← {view.back ? "Nodes" : "Overview"}</a></p>
{#if !run}
  <h1 class="heading">Run</h1>
  {#if error}<p class="source-error" role="status">{error}</p>{:else}<p class="empty-state">{view.id ? "Loading run…" : "Choose a run from Overview"}</p>{/if}
  {#if detail.error}<button type="button" class="chip" onclick={detail.retry}>Retry</button>{/if}
{:else}
  <div class="run-heading"><h1 class="heading" title={run.run_id}>{title}</h1><Status value={run.display_status || run.status} /></div>
  {#if error}<p class="source-error" role="status">{error}</p>{/if}
  {#if description}<p class="run-description">{description}</p>{/if}
  <section class="run-state" aria-label="Run state">
    <!-- The summary is the agent-facing one-liner; when the workers are listed below, only pod problems add to it. -->
    {#if run.workers?.length}
      {#each (run.pods || []).filter((p) => p.problem) as pod}<p class="source-error" role="status">{pod.name}: {pod.problem}</p>{/each}
    {:else}<p>{run.summary || "Run state unavailable"}</p>{/if}
    {#if run.progress && !finished}<p class={run.progress.state === "OK" ? "muted" : "source-error"} role="status">Progress: {run.progress.state} · {run.progress.reason}</p>{/if}
    <p class="muted">Model session: {run.session_live == null ? "Unknown" : run.session_live ? "Live" : "No live session"}{run.shared_runtime ? " · Shared runtime" : ""}</p>
    {#if run.workers_error}<p class="source-error" role="status">Workers: {run.workers_error}</p>{/if}
    {#if run.workers?.length}
      <div class="table-scroll"><table class="run-table worker-table" aria-label="Workers"><thead><tr><th>Worker</th><th>State</th><th>Node</th><th>Restarts</th><th>Age</th></tr></thead><tbody>
        {#each run.workers as worker}
          {@const pod = (run.pods || []).find((p) => p.worker === worker.workload)}
          <tr><td>{worker.role} <span class="muted micro">{worker.workload || ""}</span></td><td title={worker.detail}><Status value={worker.state} /></td>
            <td title={worker.node}>{worker.node ? shortNodeName(worker.node, inventory.cluster.nodes) : "—"}</td><td>{pod?.restarts ?? "—"}</td><td>{worker.age_seconds == null ? "—" : duration(worker.age_seconds)}</td></tr>
        {/each}
      </tbody></table></div>
    {:else if !run.workers_error}<p class="muted">No workers reported</p>{/if}
    <Configuration settings={run.settings} />
  </section>
  <div use:nodeTime={{ view, getRange: () => range, getNow: () => observed, onGesture: (value) => frozen = value }}>
  <div class="run-toolbar">
    <nav class="workspace-tabs" aria-label="Run views">
      <a href={link("activity")} aria-current={view.tab === "activity" ? "page" : undefined}>Activity</a>
      <a href={link("metrics")} aria-current={view.tab === "metrics" ? "page" : undefined}>Metrics</a>
      <a href={link("logs")} aria-current={view.tab === "logs" ? "page" : undefined}>Logs</a>
    </nav>
    <div class="time-control run-time-controls" aria-label="Run time range"><TimeControl {view} {range} {observed} /><CopyLink /><a href={url}>Agent JSON ↗</a></div>
  </div>
  <div id="run-panel">
    {#if view.tab === "logs"}
      {#if validEvent}<Logs {run} {view} {range} allLogs={link("logs", { event: null, end: null })} />{/if}
    {:else if view.tab !== "metrics"}
      <div class="run-activity"><Activity placements={detail.data?.placements || []} {range} {inventory} acrossNodes={true} error={detail.error} status={detail.status} pending={detail.pending} onpick={(at) => location.hash = link("logs", { event: at, end: range.now })} /></div>
    {:else}
      <div class="run-metrics" data-run-metrics={view.id}>
          <div class="run-metric-summary"><span>Completed steps <strong>{run.steps ?? "—"}</strong></span>{#if metricStatus}<span class={detail.error ? "source-error" : "muted"} role="status">{metricStatus}</span>{/if}</div>
          <div class="chart-grid">
            {#each roles as role (role)}<Chart title={`${{ trainer: "Trainer", sampler: "Sampler", process: "Process" }[role] || role} operation duration`} unit="s" points={samples.filter((s) => (s.role || "process") === role).map((s) => [s.at, s.elapsed_seconds])} start={range.start} end={range.now} min={0} tone="accent" />{/each}
            {#each names as name (name)}<Chart title={name} points={samples.filter((s) => Number.isFinite(s.metrics?.[name])).map((s) => [s.at, s.metrics[name]])} start={range.start} end={range.now} tone="accent" />{/each}
            {#if !samples.length && detail.data && !detail.error && !detail.pending}<p class="empty-state">No operations recorded in this time range</p>{/if}
          </div>
          <h2 class="scheduler-title">Training</h2>
          {#if reported.error}<p class="source-error" role="status">{reported.error}</p>
          {:else if !reported.data}<p class="muted">Loading…</p>
          {:else if !reportedRows.length}<p class="muted">The run's training client has not sent metrics. <code>python -m server.scripts.forward_metrics LOG_PATH</code> sends a cookbook run's.</p>
          {:else}
            <div class="log-toolbar"><input type="search" bind:value={reportedFilter} placeholder={`Filter ${reportedNames.length} metrics, e.g. reward`} aria-label="Filter training metrics" /></div>
            <div class="chart-grid">
              {#each reportedShown as [key, title, unit = ""] (key)}<Chart {title} {unit} steps points={reportedRows.filter((row) => Number.isFinite(row.metrics[key])).map((row) => [row.step, row.metrics[key]])} start={reportedRows[0].step} end={reportedRows.at(-1).step} empty="No values" />{/each}
              {#if !reportedShown.length}<p class="empty-state">No metric matches</p>{/if}
            </div>
          {/if}
          <h2 class="scheduler-title">Samplers</h2>
          {#if run.shared_runtime}<p class="muted">The pool's samplers serve every run on this base model; see their load on <a href={viewLink("nodes")}>Nodes</a>.</p>
          {:else if vllm.error}<p class="source-error" role="status">{vllm.error}</p>
          {:else if !vllm.data}<p class="muted">Loading…</p>
          {:else if !Object.values(vllm.data.samplers || {}).some((points) => points.length)}<p class="muted">{vllm.data.samplers_reason || "No sampler samples in this time range"}</p>
          {:else}<div class="chart-grid">
            {#each SAMPLER_CHARTS as [key, title, unit, scale = 1] (key)}<Chart {title} {unit} points={(vllm.data.samplers[key] || []).map(([at, value]) => [at, value * scale])} start={range.start} end={range.now} min={0} />{/each}
          </div>{/if}
      </div>
    {/if}
  </div>
  </div>
  {#if detail.data?.incidents?.length}
    <section class="run-incidents" aria-label="Significant events">
      {#each detail.data.incidents.slice(0, 3) as event}
        {@const at = new Date(event.at * 1000).toISOString()}
        {@const label = event.reason === "WorkerRestarted" ? "Worker restarted" : event.reason}
        <div class={`run-incident incident-${event.tone}`}>
          <time class="incident-time" datetime={at} title={at}>{at.slice(11, 19)}</time><span class="incident-label">{label}</span>
          <span class="incident-context" title={event.message}>{event.pod}{event.container ? ` / ${event.container}` : ""}{event.restart_count ? ` · ${event.restart_count} total restarts` : event.count > 1 ? ` · ${event.count} reported occurrences` : ""}</span>
          <a href={link("logs", { event: event.at, end: range.now })} aria-label={`View logs near ${label} on ${event.pod}`}>View logs</a>
        </div>
      {/each}
    </section>
  {/if}
{/if}
