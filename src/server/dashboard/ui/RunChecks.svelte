<script>
  import { createQuery, inspectUrl } from "./state.svelte.js";
  // One summary per session serves every run row in it; ended sessions load once.
  let { session, model } = $props();
  const summary = createQuery(() => inspectUrl(`sessions/${encodeURIComponent(session)}/summary`), { poll: 15000, pollWhen: () => summary.data?.live !== false });
  const mine = $derived(summary.data?.models.find((m) => m.model_id === model));
</script>

{#if mine?.failing.length || mine?.warning.length}<span class="run-checks">{#each mine.failing as check}<span class="check-badge" data-verdict="fail">{check}</span>{/each}{#each mine.warning as check}<span class="check-badge" data-verdict="warn">{check}</span>{/each}</span>{/if}
