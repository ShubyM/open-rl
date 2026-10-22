<script>
  import { createQuery, inspectUrl } from "./state.svelte.js";
  let { id, finished } = $props();
  const checks = createQuery(() => inspectUrl(`runs/${encodeURIComponent(id)}/checks`), { poll: 15000, pollWhen: () => !finished });
</script>

<section aria-label="Checks">
  {#if checks.error}<p class="source-error" role="status">{checks.error}{checks.data ? " · Showing previously fetched checks" : ""}</p>{/if}
  {#if checks.data}
    <div class="table-scroll"><table class="run-table check-table" aria-label="Checks"><thead><tr><th>Verdict</th><th>Check</th><th>Reason</th><th>Evidence</th></tr></thead><tbody>
      {#each checks.data.checks as check (check.check)}<tr><td><span class="check-badge" data-verdict={check.verdict}>{check.verdict}</span></td><td class="mono">{check.check}</td><td>{check.reason}</td><td class="muted">{check.evidence || "—"}</td></tr>{/each}
    </tbody></table></div>
  {:else if !checks.error}<p class="muted">Loading…</p>{/if}
</section>
