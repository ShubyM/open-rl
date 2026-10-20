<script>
  import Chart from "./Chart.svelte";
  import { createQuery, inspectUrl } from "./state.svelte.js";
  let { params, view, range, navigating = false, search = $bindable("") } = $props();
  const CHART_LIMIT = 12;
  const filter = $derived(search.trim().toLowerCase());
  let requestedFilter = $state("");
  let page = $state(0);
  $effect(() => {
    const value = filter, timer = setTimeout(() => requestedFilter = value, 250);
    page = 0;
    return () => clearTimeout(timer);
  });
  const collector = createQuery(() => inspectUrl("telemetry", { ...params, metric: requestedFilter }), {
    scope: () => params.run_id || params.node_id, poll: 15000, pollWhen: () => view.end === null, remember: () => view.end !== null,
    key: () => !navigating && filter === requestedFilter && `${params.run_id || params.node_id} ${view.duration} ${view.end} ${requestedFilter}`,
  });
  const series = $derived((collector.data?.series || []).filter((item) => item.points.some(([, value]) => Number.isFinite(value))));
  const names = $derived([...new Set(series.map((item) => item.name))]);
  const matches = $derived(series.filter((item) => item.name.toLowerCase().includes(filter)).sort((a, b) => Number(b.name.toLowerCase() === filter) - Number(a.name.toLowerCase() === filter)));
  const pages = $derived(Math.max(1, Math.ceil(matches.length / CHART_LIMIT)));
  const currentPage = $derived(Math.min(page, pages - 1));
  const shown = $derived(matches.slice(currentPage * CHART_LIMIT, (currentPage + 1) * CHART_LIMIT));
  const searching = $derived(filter !== requestedFilter || collector.pending);
  const scopeLabel = (item) => `${{ session: "Session-wide", runtime: "Shared runtime", model: "Model", pod: "Pod", node: "Node" }[item.scope] || item.scope}: ${item.scope_id} · ${item.source}-reported`;
</script>

<section aria-label="Collector metrics">
  <h2 class="scheduler-title">Collector metrics</h2>
  <p class="muted">Sampled by Prometheus; intermediate updates between scrapes are not retained.</p>
  {#if collector.error}<p class="source-error" role="status">{collector.error}{series.length ? " · Showing previously fetched metrics" : ""}</p>{/if}
  {#if collector.data?.available || series.length}
    <div class="log-toolbar"><input type="search" bind:value={search} maxlength="200" list="collector-metric-names" placeholder="Search collector metric names" aria-label="Filter collector metrics" /></div>
    <datalist id="collector-metric-names">{#each names as name (name)}<option value={name}></option>{/each}</datalist>
    <p class="muted">Showing {shown.length} of {series.length} available series{filter ? ` · ${matches.length} matching` : ""} · {CHART_LIMIT} charts per page. Search to choose metrics.</p>
    {#if collector.data.truncated}<p class="muted">First {collector.data.limit} series; filter a metric name to narrow results.</p>{/if}
    {#if pages > 1}<div class="log-toolbar" aria-label="Collector metric pages"><button type="button" class="chip" disabled={currentPage === 0} onclick={() => page = currentPage - 1}>Previous series</button><span class="muted">Page {currentPage + 1} of {pages}</span><button type="button" class="chip" disabled={currentPage === pages - 1} onclick={() => page = currentPage + 1}>Next series</button></div>{/if}
    <div class="chart-grid">
      {#each shown as item (item.id)}
        {@const producer = [item.labels?.k8s_pod_name, item.labels?.service_instance_id].filter(Boolean).join(" / ")}
        <div><Chart title={[item.name, item.phase].filter(Boolean).join(" · ")} points={item.points} start={range.start} end={range.now} {navigating} /><p class="muted micro">{scopeLabel(item)}{producer ? ` · Producer: ${producer}` : ""}</p></div>
      {/each}
      {#if !shown.length}<p class="empty-state">{searching ? "Searching…" : filter ? "No metric matches" : collector.data.reason || "No collector samples in this time range"}</p>{/if}
    </div>
  {:else if collector.data}<p class="muted">{collector.data.reason || "No collector samples in this time range"}</p>
  {:else if !collector.error}<p class="muted">Loading…</p>{/if}
</section>
