<script>
  import Chart from "./Chart.svelte";
  import { createQuery, inspectUrl } from "./state.svelte.js";
  let { params, view, range, navigating = false } = $props();
  const collector = createQuery(() => inspectUrl("telemetry", params), {
    scope: () => params.run_id || params.node_id, poll: 15000, pollWhen: () => view.end === null, remember: () => view.end !== null,
    key: () => !navigating && `${params.run_id || params.node_id} ${view.duration} ${view.end}`,
  });
  const series = $derived((collector.data?.series || []).filter((item) => item.points.some(([, value]) => Number.isFinite(value))));
  // Runs from before OpenTelemetry keep their client metrics as step history; charted by step, important ones first.
  const stored = createQuery(() => params.run_id ? inspectUrl("history", { run_id: params.run_id }) : null, { key: () => params.run_id });
  const rows = $derived(stored.data?.rows || []);
  const FIRST = [["env/all/reward/total", "Reward"], ["test/env/all/reward/total", "Held-out reward"], ["env/all/lab/criteria_pass_fraction", "Criteria pass rate"],
    ["test/env/all/lab/criteria_pass_fraction", "Held-out criteria pass rate"], ["optim/kl_sample_train_v2", "Sampler-trainer KL"], ["optim/entropy", "Entropy"], ["time/total", "Step time", "s"]];
  const stepCharts = $derived.by(() => {
    const names = new Set(rows.flatMap((row) => Object.keys(row.metrics)));
    return [...FIRST.filter(([name]) => names.has(name)), ...[...names].filter((name) => !FIRST.some(([first]) => first === name)).sort().map((name) => [name, name])];
  });
  const scopeLabel = (item) => `${{ session: "Session-wide", runtime: "Shared runtime", model: "Model", pod: "Pod", node: "Node" }[item.scope] || item.scope}: ${item.scope_id} · ${item.source}-reported`;
</script>

<section aria-label="Collector metrics">
  <h2 class="scheduler-title">Collector metrics</h2>
  {#if collector.error}<p class="source-error" role="status">{collector.error}{series.length ? " · Showing previously fetched metrics" : ""}</p>{/if}
  {#if series.length}
    {#if collector.data?.truncated}<p class="muted">First {collector.data.limit} series.</p>{/if}
    <div class="chart-grid">
      {#each series as item (item.id)}
        {@const producer = [item.labels?.k8s_pod_name, item.labels?.service_instance_id].filter(Boolean).join(" / ")}
        <div><Chart title={[item.name, item.phase].filter(Boolean).join(" · ")} points={item.points} start={range.start} end={range.now} {navigating} /><p class="muted micro">{scopeLabel(item)}{producer ? ` · Producer: ${producer}` : ""}</p></div>
      {/each}
    </div>
  {:else if rows.length}
    <p class="muted">Stored by step before OpenTelemetry.</p>
    <div class="chart-grid">
      {#each stepCharts as [key, title, unit = ""] (key)}<Chart {title} {unit} steps points={rows.filter((row) => Number.isFinite(row.metrics[key])).map((row) => [row.step, row.metrics[key]])} start={rows[0].step} end={rows.at(-1).step} empty="No values" />{/each}
    </div>
  {:else if collector.data}<p class="muted">{collector.data.reason || "0 metrics"}</p>
  {:else if !collector.error}<p class="muted">Loading…</p>{/if}
</section>
