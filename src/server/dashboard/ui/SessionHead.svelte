<script>
  import Status from "./Status.svelte";
  import { createQuery, inspectUrl, now } from "./state.svelte.js";
  import { duration } from "./format.js";
  let { id, inventory } = $props();
  const summary = createQuery(() => inspectUrl(`sessions/${encodeURIComponent(id)}/summary`), { poll: 15000, pollWhen: () => summary.data?.live !== false });
  const data = $derived(summary.data);
</script>

<div class="session-head" data-session={id}>
  <span class="session-id">Session <span class="mono">{id}</span></span>
  {#if data}
    <Status value={data.live ? "live" : "ended"} />
    <span class="muted">started {duration(Math.max(0, now(inventory) - data.created_at))} ago</span>
    {#each data.models as model (model.model_id)}<span class="session-model"><span class="mono">{model.model_id.slice(0, 8)}</span> {model.state || "Unknown"} · {model.steps} step{model.steps === 1 ? "" : "s"}{model.median_step_s == null ? "" : ` · ${duration(model.median_step_s)} median step`}</span>{/each}
    {#each data.failing as check}<span class="check-badge" data-verdict="fail">{check}</span>{/each}
    {#each data.warning as check}<span class="check-badge" data-verdict="warn">{check}</span>{/each}
  {:else if summary.error}<span class="muted">{summary.status === 404 ? "No session record" : summary.error}</span>
  {:else}<span class="muted">Loading…</span>{/if}
</div>
