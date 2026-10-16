<script>
  import Status from "./Status.svelte";
  import { inspectUrl, now, viewLink } from "./state.svelte.js";
  import { duration, elapsedTime } from "./format.js";

  let { inventory, view = $bindable() } = $props();
  const GROUPS = { unassigned: ["Unassigned"], finished: ["Completed", "Ended"] };
  const FILTERS = [
    ["all", "All", "Every recorded run"],
    ["active", "Active", "Runs with workers, or whose workers cannot be checked"],
    ["unassigned", "Unassigned", "Recorded as active, but no workers are assigned; usually a client that exited without finishing"],
    ["finished", "Finished", "Completed or ended"],
  ];
  const KINDS = { lora: "LoRA", full: "FFT", fft: "FFT" };
  const group = (run) => Object.keys(GROUPS).find((key) => GROUPS[key].includes(run.display_status)) || "active";
  const runLink = (run, tab = "activity") => viewLink(`run/${encodeURIComponent(run.run_id)}/${tab}`);
  const shown = $derived.by(() => {
    const terms = view.q.toLowerCase().split(/\s+/).filter(Boolean);
    return inventory.runs.filter((run) => {
      const text = [run.run_id, run.name, run.model, run.recipe_name, run.display_status, run.summary, KINDS[run.fine_tuning_type], ...run.nodes, ...run.pods.map((pod) => pod.name)].filter(Boolean).join(" ").toLowerCase();
      return (view.status === "all" || group(run) === view.status) && terms.every((term) => text.includes(term));
    }).sort((a, b) => Number(group(b) === "active") - Number(group(a) === "active"));
  });
  function age(run) {
    const created = typeof run.created_at === "number" ? run.created_at : Date.parse(run.created_at) / 1000;
    return Number.isFinite(created) && created > 0 ? `${duration(Math.max(0, now(inventory) - created))} ago` : "—";
  }

  const FAILED_WAITING = new Set(["CrashLoopBackOff", "ImagePullBackOff", "ErrImagePull", "CreateContainerConfigError", "CreateContainerError", "RunContainerError", "InvalidImageName", "ContainerCannotRun", "StartError"]);
  function podTone(pod) {
    const failing = (pod.containers || []).some((c) => (c.state === "waiting" && FAILED_WAITING.has(c.reason)) || (c.state === "terminated" && ((c.exit_code != null && c.exit_code !== 0) || (c.reason && c.reason !== "Completed"))));
    const reason = String(pod.problem || "").split(":", 1)[0];
    return pod.phase === "Failed" || failing || reason === "Failed" || FAILED_WAITING.has(reason) ? "error" : "warning";
  }
  const errors = $derived([...new Set([inventory.store_error, inventory.history_error, inventory.cluster.error, inventory.cluster.nodes_error, inventory.cluster.events_error, inventory.cluster.devices?.error, inventory.cluster.scheduler?.error].filter(Boolean))]);
  const issues = $derived.by(() => {
    const pods = (inventory.cluster.pods || []).filter((pod) => pod.problem).map((pod) => {
      const run = inventory.runs.find((r) => r.pods.some((p) => p.uid === pod.uid));
      // A problem reads "Reason: detail". The reason is the status; the detail and restarts are the evidence.
      const [reason, detail = ""] = pod.problem.split(/:\s*(.*)/s);
      const evidence = [detail, pod.restarts ? `${pod.restarts} restarts` : ""].filter(Boolean).join(" · ") || "—";
      return { issue: reason, resource: pod.name, evidence, tone: podTone(pod), link: run ? runLink(run, "logs") : inspectUrl(`pods/${encodeURIComponent(pod.name)}/logs`), label: "Logs" };
    });
    return [...pods, ...(inventory.cluster.nodes || []).filter((node) => node.ready !== true).map((node) => ({ issue: "Node not ready", resource: node.name, evidence: "Ready condition is false or unknown", tone: "error", link: viewLink("nodes", { node: node.name }), label: "Nodes" }))];
  });
</script>

{#snippet healthStatus(label, tone)}
  <span class={`health-status health-${tone}`}><span class="health-dot" aria-hidden="true"></span>{label}</span>
{/snippet}

{#if view.page === "overview"}
  <h1 class="heading">Overview</h1>
  {#if inventory.store_error}<p class="source-error" role="status">{inventory.store_error}</p>{/if}
  {#if !inventory.store_error || inventory.runs.length}
    <div class="overview-toolbar">
      <input id="run-search" type="search" placeholder="Search by run ID, model, name, recipe, node or pod" aria-label="Search runs" bind:value={view.q} autocomplete="off">
      <div class="run-filters" role="group" aria-label="Filter by status">
        {#each FILTERS as [key, label, title]}
          <button type="button" class="chip" data-run-filter={key} {title} aria-pressed={view.status === key} onclick={() => view.status = key}>{label} <span class="run-filter-count">{key === "all" ? inventory.runs.length : inventory.runs.filter((run) => group(run) === key).length}</span></button>
        {/each}
      </div>
    </div>
    <div class="job-list run-list">
      {#if shown.length}
        <div class="job-list-head run-row">
          <span>Job</span><span>Status</span><span>Training kind</span><span>Completed steps</span><span>Created</span><span>Elapsed</span>
        </div>
        {#each shown as run (run.run_id)}
          {@const label = [run.display_name, run.recipe_name].filter((value, i, values) => value && values.indexOf(value) === i).join(" · ")}
          {@const model = (run.model || "Run").split("/").at(-1)}
          <div class="job-list-row run-row" data-key={run.run_id}>
            <span class="job-identity"><a href={runLink(run)} title={run.run_id}>{model} · <span class="mono">{run.run_id.slice(0, 8)}</span></a>{#if label}<span class="muted micro">{label}</span>{/if}</span>
            <span title={run.summary || undefined}><Status value={run.display_status || run.status} /></span><span>{KINDS[run.fine_tuning_type] || run.fine_tuning_type || "—"}</span><span>{run.steps ?? "—"}</span><span>{age(run)}</span><span>{group(run) === "unassigned" ? "—" : elapsedTime(run, inventory.observed_at)}</span>
          </div>
        {/each}
      {/if}
    </div>
    {#if !shown.length}<div class="empty-state">{inventory.runs.length ? "No runs match the search and status filter" : "No runs recorded"}</div>{/if}
  {/if}
{:else if view.page === "health"}
  <h1 class="heading">Health</h1>
  {#each errors as error}<p class="health-message" role="status">{@render healthStatus("Source unavailable", "warning")}<span>{error}</span></p>{/each}
  {#if issues.length}
    <div class="scheduler-table-wrap"><table class="scheduler-table"><thead><tr><th>Issue</th><th>Resource</th><th>Evidence</th><th></th></tr></thead><tbody>
      {#each issues as issue}<tr><td>{@render healthStatus(issue.issue, issue.tone)}</td><td>{issue.resource}</td><td>{issue.evidence}</td><td><a href={issue.link}>{issue.label} ↗</a></td></tr>{/each}
    </tbody></table></div>
  {:else if inventory.cluster.available === true && !errors.length}<p class="health-message">{@render healthStatus("Healthy", "success")}<span>No pod problems, node problems or source errors.</span></p>{/if}
  <p class="run-json-link"><a href={inspectUrl("inventory")}>Diagnostic JSON ↗</a> · <a href="/docs">API reference ↗</a></p>
{/if}
