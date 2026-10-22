<script>
  import { createQuery, inspectUrl } from "./state.svelte.js";
  import { flaggedChecks } from "./format.js";
  // A run's failing and warning checks, each drawn by the Health page's row snippet; flagged stays undefined until they load.
  let { run, row, flagged = $bindable() } = $props();
  const checks = createQuery(() => inspectUrl(`runs/${encodeURIComponent(run.run_id)}/checks`), { poll: 15000 });
  const shown = $derived(flaggedChecks(checks.data?.checks));
  $effect(() => { flagged = checks.data || checks.error ? shown.length + (checks.error ? 1 : 0) : undefined; });
</script>

{#each shown as check (check.check)}{@render row(run, check)}{/each}
{#if checks.error}{@render row(run, { check: "checks", verdict: "unknown", reason: checks.error })}{/if}
