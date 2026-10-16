<script>
  import { untrack, tick } from "svelte";
  import { get, inspectUrl } from "./state.svelte.js";
  let { run, view, range, allLogs } = $props();
  let records = $state([]), lifetimes = $state([]), cursor = $state(null), pending = $state(true), error = $state(null), source = $state(null);
  let controller, request = 0, loaded = "", filters = "", successfulParams, successfulEnd, failedPage, lines, fromBottom = 0;
  const logRange = $derived({
    since: new Date(Math.max(range.start, view.event === null ? range.start : view.event - 120) * 1000).toISOString(),
    until: new Date(Math.min(range.now, view.event === null ? range.now : view.event + 120) * 1000).toISOString(),
  });
  const params = $derived({ run_id: view.id, q: view.q, limit: "200", ...logRange, ...(view.pod ? { pod: view.pod } : {}) });
  const queryKey = $derived(JSON.stringify(params));
  const filterKey = $derived(JSON.stringify([view.id, view.q, view.pod, view.event, view.duration]));
  const sources = $derived.by(() => {
    const pods = new Map((run.pods || []).map((p) => [p.name, p]));
    for (const record of [...lifetimes, ...records]) if (record.pod && !pods.has(record.pod)) pods.set(record.pod, { ...record, name: record.pod });
    if (view.pod && !pods.has(view.pod)) pods.set(view.pod, { name: view.pod });
    return [...pods.values()].sort((a, b) => a.name.localeCompare(b.name));
  });
  // Dozzle-style line tags: the part of the pod name that differs between the run's pods, e.g. "0-trainer".
  const cut = $derived.by(() => {
    const [first = "", ...rest] = sources.map((p) => p.name);
    let n = 0;
    while (rest.length && n < first.length && rest.every((name) => name[n] === first[n])) n++;
    return rest.length ? first.lastIndexOf("-", n - 1) + 1 : first.length;
  });
  const hues = $derived(new Map(sources.map((p, i) => [p.name, `hue-${i % 8}`])));
  const tagWidth = $derived(Math.min(20, Math.max(0, ...sources.map((p) => p.name.length - cut))));
  const match = $derived(view.q ? new RegExp(`(${RegExp.escape(view.q)})`, "i") : null);
  // The API pages newest first; a tail reads down to the newest line.
  const shown = $derived(records.toReversed());
  const paging = $derived(pending && view.end !== null);
  const status = $derived([
    { gke: "Cloud Logging" }[source] || source || "Logs",
    records.length || !error ? `${records.length.toLocaleString()} ${records.length === 1 ? "record" : "records"}` : "",
    pending && !records.length ? "Loading…" : error ? "" : view.end === null ? "Live" : "Paused",
  ].filter(Boolean).join(" · "));
  const stop = () => { request++; controller?.abort(); controller = null; pending = false; };
  async function load(query, page = null) {
    stop();
    const version = request, windowEnd = range.now;
    controller = new AbortController();
    pending = true;
    error = null;
    try {
      const data = await get(inspectUrl("logs", { ...query, ...(page ? { cursor: page } : {}) }), controller.signal);
      if (version !== request) return;
      if (data.error) throw new Error(data.error);
      fromBottom = page ? lines.scrollHeight - lines.clientHeight - lines.scrollTop : 0;
      records = [...new Map([...(page ? records : []), ...(data.records || [])].map((r) => [r.id, r])).values()].slice(0, 2000);
      lifetimes = data.sources || [];
      cursor = records.length < 2000 ? data.next_cursor : null;
      source = data.source;
      successfulParams = query;
      successfulEnd = windowEnd;
      loaded = JSON.stringify(query);
      failedPage = null;
    } catch (failure) {
      if (version === request) { error = failure.message; failedPage = { query, page }; }
    } finally { if (version === request) { pending = false; controller = null; } }
  }
  $effect(() => {
    const key = queryKey, query = JSON.parse(key), follow = view.end === null, scope = filterKey;
    return untrack(() => {
      const changed = scope !== filters, searched = changed && filters && JSON.parse(scope)[1] !== JSON.parse(filters)[1];
      if (changed) { records = []; cursor = source = error = null; loaded = ""; successfulParams = successfulEnd = undefined; filters = scope; }
      const start = key !== loaded ? setTimeout(() => load(query), searched ? 250 : 0) : null;
      const timer = follow ? setInterval(() => { if (!document.hidden && !pending) load(query); }, 5000) : null;
      return () => { clearTimeout(start); clearInterval(timer); stop(); };
    });
  });
  // Pin the tail to the newest line; an older page keeps the lines in view where they were.
  $effect(() => { shown; lines.scrollTop = lines.scrollHeight - lines.clientHeight - fromBottom; fromBottom = 0; });
  function pause() {
    stop();
    // Event-clipped `until` can be earlier than the inspected window end.
    view.end = successfulEnd ?? range.now;
  }
  function follow() {
    view.event = view.end = null;
    cursor = null;
    loaded = "";
  }
  async function older() {
    if (paging || !cursor) return;
    pause();
    // Wait for the paused query's effect cleanup before starting its next page.
    await tick();
    load(successfulParams, cursor);
  }
  const timestamp = (value) => Number.isFinite(Date.parse(value)) ? new Date(value).toISOString() : "";
  const severity = (record) => ["ERROR", "CRITICAL", "ALERT", "EMERGENCY"].includes(record.severity) ? "error" : record.severity === "WARNING" ? "warning" : undefined;
</script>

{#if run.shared_runtime}<p class="muted">These pods serve a shared LoRA runtime. Their logs can include other runs.</p>{/if}
{#if view.event !== null}
  <div class="log-time-scope"><span>{logRange.since.slice(0, 10)} · {logRange.since.slice(11, 19)}–{logRange.until.slice(11, 19)}</span><a href={allLogs}>All logs</a></div>
{/if}
<div class="log-toolbar">
  <input id="log-search" type="search" bind:value={view.q} placeholder="Search logs" aria-label="Search logs" />
  <select id="log-source" aria-label="Pod" bind:value={view.pod}><option value="">All pods</option>
    {#each sources as pod (pod.name)}<option value={pod.name}>{pod.name.slice(cut) || pod.name}</option>{/each}
  </select>
  <button type="button" class="chip" onclick={() => view.end === null ? pause() : follow()}>{view.end === null ? "Pause" : "Jump to latest"}</button>
</div>
<div id="log-status" class="log-status" role="status"><span>{status}</span>
  {#if error}<span class="log-error">{error}{records.length ? " · Showing previously fetched records" : ""}</span><span><button type="button" class="chip" disabled={pending} onclick={() => load(failedPage.query, failedPage.page)}>Retry logs</button></span>{/if}
</div>
<!-- svelte-ignore a11y_no_noninteractive_tabindex (The log region supports native keyboard scrolling.) -->
<div id="log-lines" bind:this={lines} tabindex="0" role="region" aria-label="Run logs" style:--tag-width={`${tagWidth}ch`} onscroll={(event) => { const box = event.currentTarget; if (view.end === null && box.scrollHeight - box.clientHeight - box.scrollTop > 24) pause(); }}>
  {#if cursor}<button type="button" class="chip log-older" disabled={paging} onclick={older}>{paging ? "Loading…" : "Older logs"}</button>
  {:else if records.length === 2000}<p class="muted log-older">2,000 records shown. Narrow the time range or search to inspect more.</p>{/if}
  {#each shown as record (record.id)}
    {@const at = timestamp(record.timestamp)}
    <div class="workspace-logrow" data-key={record.id} data-severity={severity(record)}>
      <time class="log-time" datetime={at} title={at || "No timestamp"}>{at ? at.slice(11, 23) : "—"}</time><span class={`log-pod ${hues.get(record.pod) || ""}`} title={record.pod}>{record.pod?.slice(cut)}</span><pre class="log-message">{#each match ? record.message.split(match) : [record.message] as part, i}{#if i % 2}<mark>{part}</mark>{:else}{part}{/if}{/each}{#if record.message_truncated}<span class="muted">{"\n[Message truncated by source]"}</span>{/if}</pre>
    </div>
  {:else}{#if !pending && !error}<p class="empty-state">No logs match this time range and filter</p>{/if}{/each}
</div>
