<script>
  import { createQuery, inspectUrl, rangeOf, viewLink, saved } from "./state.svelte.js";
  import Status from "./Status.svelte";
  import Chart from "./Chart.svelte";
  import Activity from "./Activity.svelte";
  import Logs from "./Logs.svelte";
  import CopyLink from "./CopyLink.svelte";
  import Configuration from "./Configuration.svelte";
  import { duration, shortNodeName, KINDS, runDescription } from "./format.js";
  let { inventory, view } = $props();
  const range = $derived(rangeOf(inventory, view));
  const validEvent = $derived(view.event === null || (view.event >= range.start && view.event <= range.now));
  $effect(() => { if (!validEvent) view.event = null; });
  const url = $derived(inspectUrl("activity", { since: new Date(range.start * 1000).toISOString(), until: new Date(range.now * 1000).toISOString(), run_id: view.id }));
  const detail = createQuery(() => url, { poll: 15000, pollWhen: () => view.end === null, key: () => `${view.id} ${view.duration} ${view.end}` });
  const run = $derived((detail.status !== 404 ? detail.data?.run : null) || inventory.runs.find((r) => r.run_id === view.id));
  const error = $derived(inventory.store_error || inventory.processes_error || detail.error);
  const title = $derived(run ? [(run.model || "Run").split("/").at(-1), run.run_id.slice(0, 8), KINDS[run.fine_tuning_type]].filter(Boolean).join(" · ") : "Run");
  const description = $derived(run ? runDescription(run, title) : "");
  const samples = $derived((detail.data?.samples || []).filter((s) => s.at >= range.start && s.at <= range.now));
  const roles = $derived([...new Set(samples.map((s) => s.role || "process"))]);
  const names = $derived([...new Set(samples.flatMap((s) => Object.entries(s.metrics || {}).filter(([, v]) => Number.isFinite(v)).map(([k]) => k)))].slice(0, 12));
  const metricStatus = $derived(detail.error ? `${detail.error}${samples.length ? " · Showing previously fetched metrics" : ""}` : detail.pending ? "Updating…" : "");
  const custom = $derived(view.end !== null || ![600, 1800, 3600].includes(view.duration));
  const metricsOpen = () => view.tab === "metrics" && `${view.id} ${view.duration} ${view.end}`;
  // A shared pool's load is every run's, so it is charted on Nodes rather than here.
  const vllm = createQuery(() => run?.shared_runtime ? null : inspectUrl("series", { run_id: view.id, since: new Date(range.start * 1000).toISOString(), until: new Date(range.now * 1000).toISOString() }), { poll: 15000, pollWhen: () => view.end === null, key: metricsOpen });
  // The vLLM series the inspection API returns for a run's samplers, as vLLM's and llm-d's dashboards chart them.
  const SAMPLER_CHARTS = [["generation_tokens_per_second", "Generated tokens", "tok/s"], ["prompt_tokens_per_second", "Prompt tokens", "tok/s"], ["requests_running", "Running requests", ""], ["requests_waiting", "Waiting requests", ""],
    ["kv_cache_usage", "KV cache use", "%", 100], ["preemptions_per_second", "Preemptions", "/s"], ["time_to_first_token_p99_seconds", "Time to first token p99", "s"], ["request_latency_p99_seconds", "Request latency p99", "s"]];
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
    {#if run.progress}<p class={run.progress.state === "OK" ? "muted" : "source-error"} role="status">Progress: {run.progress.state} · {run.progress.reason}</p>{/if}
    <p class="muted">Model session: {run.session_live == null ? "Unknown" : run.session_live ? "Live" : "No live session"}{run.shared_runtime ? " · Shared runtime" : ""}</p>
    {#if run.workers_error}<p class="source-error" role="status">Workers: {run.workers_error}</p>{/if}
    {#if run.workers?.length}
      <div class="table-scroll"><table class="run-table" aria-label="Workers"><thead><tr><th>Worker</th><th>State</th><th>Node</th><th>Age</th></tr></thead><tbody>
        {#each run.workers as worker}
          <tr><td>{worker.role}<div class="muted micro">{worker.workload || ""}</div></td><td title={worker.detail}><Status value={worker.state} /></td>
            <td title={worker.node}>{worker.node ? shortNodeName(worker.node, inventory.cluster.nodes) : "—"}</td><td>{worker.age_seconds == null ? "—" : duration(worker.age_seconds)}</td></tr>
        {/each}
      </tbody></table></div>
    {:else if !run.workers_error}<p class="muted">No workers reported</p>{/if}
    <Configuration settings={run.settings} />
  </section>
  <div class="run-toolbar">
    <nav class="workspace-tabs" aria-label="Run views">
      <a href={link(view.tab === "metrics" ? "metrics" : "activity")} aria-current={view.tab !== "logs" ? "page" : undefined}>Activity</a>
      <a href={link("logs")} aria-current={view.tab === "logs" ? "page" : undefined}>Logs</a>
    </nav>
    <div class="run-time-controls"><label>Time range
      <select id="event-range" aria-label="Time range" title={`${new Date(range.start * 1000).toISOString()} – ${new Date(range.now * 1000).toISOString()}`}
        value={view.duration / 60} onchange={(e) => { view.duration = Number(e.currentTarget.value) * 60; view.end = view.event = null; }}>
        {#if custom}<option value={view.duration / 60}>{view.back ? "Selected node window" : "Selected time window"}</option>{/if}
        {#each [10, 30, 60] as minutes}<option value={minutes}>Last {minutes} minutes</option>{/each}
      </select></label><CopyLink /><a href={url}>Agent JSON ↗</a>
    </div>
  </div>
  <div id="run-panel">
    {#if view.tab === "logs"}
      {#if validEvent}<Logs {run} {view} {range} allLogs={link("logs", { event: null, end: null })} />{/if}
    {:else}
      <div class="run-activity"><Activity placements={detail.data?.placements || []} {range} {inventory} acrossNodes={true} error={detail.error} status={detail.status} pending={detail.pending} onpick={(at) => location.hash = link("logs", { event: at, end: range.now })} /></div>
      <details class="run-metrics" data-run-metrics={view.id} bind:open={() => view.tab === "metrics", (open) => view.tab = open ? "metrics" : "activity"}>
        <summary>Metrics</summary>
        {#if view.tab === "metrics"}
          <div class="run-metric-summary"><span>Completed steps <strong>{run.steps ?? "—"}</strong></span>{#if metricStatus}<span class={detail.error ? "source-error" : "muted"} role="status">{metricStatus}</span>{/if}</div>
          <div class="chart-grid">
            {#each roles as role (role)}<Chart title={`${{ trainer: "Trainer", sampler: "Sampler", process: "Process" }[role] || role} operation duration`} unit="s" points={samples.filter((s) => (s.role || "process") === role).map((s) => [s.at, s.elapsed_seconds])} start={range.start} end={range.now} min={0} tone="accent" />{/each}
            {#each names as name (name)}<Chart title={name} points={samples.filter((s) => Number.isFinite(s.metrics?.[name])).map((s) => [s.at, s.metrics[name]])} start={range.start} end={range.now} tone="accent" />{/each}
            {#if !samples.length && detail.data && !detail.error && !detail.pending}<p class="empty-state">No operations recorded in this time range</p>{/if}
          </div>
          <h2 class="scheduler-title">Samplers</h2>
          {#if run.shared_runtime}<p class="muted">The pool's samplers serve every run on this base model; see their load on <a href={viewLink("nodes")}>Nodes</a>.</p>
          {:else if vllm.error}<p class="source-error" role="status">{vllm.error}</p>
          {:else if !vllm.data}<p class="muted">Loading…</p>
          {:else if !Object.values(vllm.data.samplers || {}).some((points) => points.length)}<p class="muted">{vllm.data.samplers_reason || "No sampler samples in this time range"}</p>
          {:else}<div class="chart-grid">
            {#each SAMPLER_CHARTS as [key, title, unit, scale = 1] (key)}<Chart {title} {unit} points={(vllm.data.samplers[key] || []).map(([at, value]) => [at, value * scale])} start={range.start} end={range.now} min={0} />{/each}
          </div>{/if}
          <h2 class="scheduler-title">Processes</h2>
          {#if run.shared_runtime}<p class="muted">Shared LoRA runtime</p>{/if}
          <div class="table-scroll"><table class="run-table"><thead><tr><th>Process</th><th>Kind</th><th>Node</th><th>State</th><th>Restarts</th></tr></thead>
            <tbody>{#each run.pods || [] as p (p.uid || p.name)}<tr><td>{p.name}</td><td>{p.role ? p.role[0].toUpperCase() + p.role.slice(1) : "Unknown"}</td><td>{p.node ?? "—"}</td><td title={p.problem}><Status value={p.problem?.split(":")[0] || p.phase} /></td><td>{p.restarts ?? "—"}</td></tr>{/each}</tbody>
          </table></div>
          {#if !run.pods?.length}<p class="empty-state">{inventory.cluster.available ? "No current pods" : inventory.cluster.error || "Process information unavailable"}</p>{/if}
        {/if}
      </details>
    {/if}
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
