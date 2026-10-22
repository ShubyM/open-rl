<script>
  import { inspectUrl } from "./state.svelte.js";
  // The run page's query, which also labels the run.
  let { checks } = $props();
  // Symptoms first, then the causes that explain them.
  const ordered = $derived([...(checks.data?.checks || [])].sort((a, b) => (a.kind === "cause") - (b.kind === "cause")));
  const link = (e) => inspectUrl(e.read, e.args);
</script>

<section aria-label="Checks">
  {#if checks.error}<p class="source-error" role="status">{checks.error}{checks.data ? " · Showing previously fetched checks" : ""}</p>{/if}
  {#if checks.data}
    <p class="check-verdict">Verdict <span class="check-badge" data-verdict={checks.data.verdict}>{checks.data.verdict}</span> <span class="muted">from the symptom checks; causes say where to look when one fails</span></p>
    <div class="table-scroll"><table class="run-table check-table" aria-label="Checks"><thead><tr><th>Verdict</th><th>Check</th><th>Reason</th><th>Evidence</th></tr></thead><tbody>
      {#each ordered as check (check.check)}<tr data-kind={check.kind}><td><span class="check-badge" data-verdict={check.verdict}>{check.verdict}</span></td><td><span class="mono">{check.check}</span><span class="muted micro">{check.kind === "cause" ? `cause of ${check.explains}` : "symptom"}</span></td><td>{check.reason}</td><td class="muted">{#each check.evidence || [] as e}<a class="check-evidence" href={link(e)} title={e.head || undefined}>{e.read}{Object.keys(e.args).length ? ` · ${Object.entries(e.args).filter(([k]) => k !== "run_id").map(([k, v]) => `${k}=${v}`).join(" ")}` : ""}</a>{:else}—{/each}</td></tr>{/each}
    </tbody></table></div>
  {:else if !checks.error}<p class="muted">Loading…</p>{/if}
</section>
