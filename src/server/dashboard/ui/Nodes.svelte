<script>
  import { tick, onDestroy } from "svelte";
  import Chart from "./Chart.svelte";
  import Activity from "./Activity.svelte";
  import CopyLink from "./CopyLink.svelte";
  import { createQuery, inspectUrl, now, viewLink, crosshair, crosshairOffset, timeAt } from "./state.svelte.js";
  import { WINDOWS, nodeTime } from "./node-time.js";
  import { axisTime, chartTime, deviceLabel, acceleratorLabel, shortNodeName, placementColor, clipPlacements, duty, claimLabel, capacityBars, activitySource, deviceSeries, sumSeries, gapFor, lastValue } from "./visualization.js";

  let { inventory, view } = $props();
  let element, picker, frozen = $state(null);
  const observed = $derived(now(inventory));
  const range = $derived.by(() => {
    const duration = Math.max(60, Math.min(86400, Math.round(view.duration || 1800)));
    const end = Math.max(observed - 86400 + duration, Math.min(observed, view.end ?? observed));
    return { start: end - duration, now: end, live: view.end === null };
  });
  const queryRange = $derived(frozen || range);
  const bounds = $derived({ since: new Date(queryRange.start * 1000).toISOString(), until: new Date(queryRange.now * 1000).toISOString() });
  const history = createQuery(() => inspectUrl("activity", bounds), { poll: 15000, pollWhen: () => !frozen });
  // Lanes wait for the first history response so they render once, complete, instead of filling in and shifting.
  const loading = $derived(!history.data && !history.error);
  const all = $derived(clipPlacements(history.data?.placements || [], range, history.error, history.status));
  const node = $derived(inventory.cluster.nodes.find((n) => n.name === view.node));
  const series = createQuery(() => node ? inspectUrl("series", { node_id: node.name, ...bounds }) : null, { scope: () => node?.name, poll: 15000, pollWhen: () => !frozen });
  const metrics = $derived(series.data || { devices: [], workers: [], reason: series.error || "Loading…" });
  const selectedDevices = $derived([...new Map((metrics.devices || []).filter((d) => view.gpu === "all" || view.gpu === d.id).map((d) => [d.uuid || d.id, d])).values()]);
  const gpuPoints = $derived(deviceSeries(selectedDevices, "utilization"));
  const cpuPoints = $derived(sumSeries(metrics.workers || [], "cpu_cores"));
  const gpuMemory = $derived(lastValue(deviceSeries(selectedDevices, "memory_mib"), range));
  const workerMemory = $derived(lastValue(sumSeries(metrics.workers || [], "memory_bytes"), range));
  const failures = $derived([...new Set([series.error, ...selectedDevices.map((d) => d.reason)].filter(Boolean))]);
  const visible = $derived(all.filter((p) => p.node === node?.name && (view.gpu === "all" || p.devices.includes(view.gpu)) && p.start <= range.now && p.end >= range.start));
  const errors = $derived([...new Set([inventory.cluster.error, inventory.cluster.nodes_error, inventory.cluster.devices?.error, inventory.cluster.scheduler?.error, inventory.history_error, history.error].filter(Boolean))]);
  const nodes = $derived([...inventory.cluster.nodes].sort((a, b) => Number(inventory.placements.some((p) => p.node === b.name)) - Number(inventory.placements.some((p) => p.node === a.name)) || Number(b.gpu_capacity > 0) - Number(a.gpu_capacity > 0)));
  const unavailable = $derived(!node && (view.node || (view.placement && history.data)));
  const duration = $derived(range.now - range.start);
  const custom = $derived([[Math.floor(duration / 3600), "h"], [Math.floor(duration / 60) % 60, "m"], [duration % 60, "s"]].filter(([n]) => n).map(([n, unit]) => `${n}${unit}`).join(" "));
  const options = $derived(WINDOWS.some(([n]) => n === duration) ? WINDOWS : [[duration, `${custom} (custom)`], ...WINDOWS]);
  const local = (at) => new Date(at * 1000).toISOString().slice(0, duration <= 120 ? 19 : 16);
  const timeLabel = $derived(range.live ? `Last ${WINDOWS.find(([n]) => n === duration)?.[1] || custom}` : `${local(range.start).slice(5, 10)} · ${chartTime(range.start, duration <= 120)} – ${local(range.start).slice(0, 10) === local(range.now).slice(0, 10) ? "" : `${local(range.now).slice(5, 10)} · `}${chartTime(range.now, duration <= 120)}`);
  const back = $derived(viewLink("nodes", { duration, end: range.now, node: view.node, placement: view.placement, gpu: view.node && view.gpu !== "all" ? view.gpu : null }));
  const sourceNote = $derived({ gke: "GPU: DCGM via Cloud Monitoring · CPU/memory: GKE container metrics", prometheus: "GPU: DCGM via Prometheus · CPU/memory: GKE container metrics" }[metrics.source] || "");
  const cursor = $derived(crosshairOffset(range));
  $effect(() => {
    if (view.duration !== duration) view.duration = duration;
    if (view.end !== null && view.end !== range.now) view.end = range.now;
    if (view.placement && !all.some((p) => p.id === view.placement)) {
      const alias = all.find((p) => p.allocation_id === view.placement);
      if (alias) view.placement = alias.id;
    }
    const selected = all.find((p) => p.id === view.placement);
    if (!view.node && selected) view.node = selected.node;
    if (node && selected && selected.node !== node.name) view.placement = null;
    if (node && (node.devices.length === 1 || (view.gpu !== "all" && !node.devices.some((d) => d.id === view.gpu)))) view.gpu = "all";
  });
  let scrollSelection = $state(true);
  $effect(() => {
    if (!view.node && !view.placement) scrollSelection = false;
    if (node && !loading && scrollSelection) { scrollSelection = false; tick().then(() => element?.querySelector("#placement-detail")?.scrollIntoView({ block: "start" })); }
  });
  function keepViewport() { element.closest("#app").style.minHeight = `${window.scrollY + innerHeight}px`; }
  function select(p, reset = false) {
    if (view.node !== p.node) element.closest("#app").style.removeProperty("min-height");
    if (reset) view.gpu = "all";
    view.node = p.node;
    view.placement = p.id;
  }
  async function close() {
    const previous = view.node;
    view.node = view.placement = null;
    element.closest("#app").style.removeProperty("min-height");
    await tick();
    element.querySelector(`[data-node-details="${CSS.escape(previous || "")}"]`)?.focus({ preventScroll: true });
  }
  function openNode(name) {
    if (view.node === name) return close();
    element.closest("#app").style.removeProperty("min-height");
    view.node = name; view.placement = null; view.gpu = "all";
  }
  function shift(sign) { const end = range.now + sign * duration; view.end = end >= observed ? null : end; }
  function until(event) {
    if (!event.currentTarget.validity.valid) { event.currentTarget.reportValidity(); return; }
    const end = event.currentTarget.value ? Date.parse(`${event.currentTarget.value}Z`) / 1000 : null;
    view.end = Number.isFinite(end) ? Math.min(end, observed) : null;
  }
  function escapeKey(event) {
    if (event.key !== "Escape" || event.defaultPrevented) return;
    if (picker?.open) { picker.open = false; picker.querySelector("summary").focus(); event.preventDefault(); }
    else if (view.node || view.placement) close();
  }
  onDestroy(() => element?.closest("#app")?.style.removeProperty("min-height"));
</script>
<svelte:window onhashchange={() => scrollSelection = true} onkeydown={escapeKey} onclick={(event) => { if (picker?.open && !picker.contains(event.target)) picker.open = false; }} />
<div bind:this={element} use:nodeTime={{ view, getRange: () => range, getNow: () => observed, onGesture: (value) => frozen = value }} data-crosshair={cursor !== null ? "" : undefined} style:--crosshair={`${cursor ?? 0}%`}>
  <div class="nodes-heading"><h1 class="heading">Kubernetes nodes</h1>
    <div class="time-control" aria-label="Node time range">
      <button type="button" class="time-shift" data-time-shift="-1" aria-label="Previous time window" disabled={range.start <= observed - 86400} onclick={() => shift(-1)}><svg class="time-chevron" viewBox="0 0 16 16" aria-hidden="true"><path d="m3 5.5 5 5 5-5" transform="rotate(90 8 8)" /></svg></button>
      <details class="time-picker" data-key="node-time-picker" bind:this={picker}><summary class="time-summary" title={timeLabel}><span>{timeLabel}</span><svg class="time-chevron" viewBox="0 0 16 16" aria-hidden="true"><path d="m3 5.5 5 5 5-5" /></svg></summary>
        <div class="time-popover"><label>Window <select data-time-duration value={duration} onchange={(event) => { view.duration = Number(event.currentTarget.value); if (view.end !== null) view.end = range.now; }}>{#each options as [n, text] (n)}<option value={n}>{text}</option>{/each}</select></label>
          <label>Until (UTC)<input type="datetime-local" data-time-end value={range.live ? "" : local(range.now)} min={local(observed - 86400 + duration)} max={local(observed)} step="any" onchange={until}></label>
          {#if range.live}<span class="muted">Following current time</span>{:else}<button type="button" class="chip" data-time-live="true" onclick={() => view.end = null}>Return to live</button>{/if}
          <span class="muted">W/S zoom · A/D pan</span>
        </div>
      </details>
      <button type="button" class="time-shift" data-time-shift="1" aria-label="Next time window" disabled={range.now >= observed} onclick={() => shift(1)}><svg class="time-chevron" viewBox="0 0 16 16" aria-hidden="true"><path d="m3 5.5 5 5 5-5" transform="rotate(-90 8 8)" /></svg></button><CopyLink />
    </div>
  </div>
  {#each errors as error}<p class="source-error" role="status">{error}</p>{/each}
  {#if history.error}<button type="button" class="chip" data-retry-data="activity:nodes" onclick={history.retry}>Retry activity</button>{/if}
  {#if !inventory.cluster.available && !errors.length}<p class="empty-state">Kubernetes unavailable</p>{/if}
  {#if unavailable}<div class="unavailable-selection" role="status"><span>{view.node ? "The selected node is no longer reported." : "The selected placement is not in retained history."}</span><button type="button" class="detail-close" data-close-details aria-label="Clear unavailable selection" title="Clear selection" onclick={close}>×</button></div>{/if}
  <div class="node-time-header"><span>Node</span><div class="node-axis">{#each [0, 1, 2, 3] as tick}<span>{axisTime(range.start + (range.now - range.start) * tick / 3, range)}</span>{/each}</div><span class="node-duty" title="GPU claimed time over this window, not measured GPU utilization">Claimed</span></div>
  {#each loading ? [] : nodes as item (item.name)}
    {@const devices = item.devices || []}
    {@const height = devices.length === 1 ? 30 : 22}
    {@const placements = all.filter((p) => p.node === item.name && p.start <= range.now && p.end >= range.start)}
    {@const current = range.live ? inventory.placements.filter((p) => p.node === item.name) : placements.filter((p) => p.start <= range.now && p.end >= range.now)}
    {@const unmapped = placements.filter((p) => !p.devices.length || p.devices.some((id) => !devices.some((d) => d.id === id)))}
    <div class="node-placement-group" data-key={item.name}>
      <div class="node-lane"><button type="button" class="node-lane-label" data-node-details={item.name} aria-expanded={view.node === item.name} title={item.name} onclick={() => openNode(item.name)}><span class="node-accelerator">{acceleratorLabel(item)} <span class="muted">· {shortNodeName(item.name, inventory.cluster.nodes)}</span></span>{#if item.gpu_capacity}<span class="claim-label">{claimLabel(item, current, range.live)}</span>{/if}</button>
        <div>{#if devices.length}<div class="gpu-capacity"><div class="gpu-lane-ids" style:grid-auto-rows={`${height}px`}>{#each devices as device (device.id)}<span title={device.id}>{deviceLabel(device.name)}</span>{/each}</div><!-- svelte-ignore a11y_no_static_element_interactions (Pointer location follows the shared time axis.) --><div class="capacity-track" style:height={`${devices.length * height}px`} style:--gpu-lane-height={`${height}px`} onpointermove={(event) => crosshair.at = event.buttons ? null : timeAt(event, range)} onpointerleave={() => crosshair.at = null}>
          {#each capacityBars(devices, placements, range, view.placement) as bar (bar.key)}
            <button type="button" class={`capacity-allocation ${bar.shared ? "shared" : placementColor(bar.placement)}`} class:selected={bar.selected} class:ended={!bar.shared && bar.placement.ended} data-key={bar.key} data-placement={bar.placement.id} aria-expanded={bar.selected} aria-label={bar.title} title={bar.title} style={bar.style} onclick={(event) => select(placements.find((p) => p.id === event.target.closest("[data-placement]")?.dataset.placement) || bar.placement, true)}>
              {#if bar.shared}{#each bar.holds as hold}<span class={`hold ${placementColor(hold.placement)}`} class:selected={hold.placement.id === view.placement} data-placement={hold.placement.id} style={hold.style} title={`${hold.placement.label} · ${hold.placement.role || ""}`}></span>{/each}
              {:else}<span class="allocation-name">{bar.placement.label}</span><span class="allocation-count">{bar.count} GPU{bar.count === 1 ? "" : "s"}</span>{/if}
            </button>
          {/each}
        </div></div>{/if}
        {#each unmapped as p (p.id)}<button type="button" class={`capacity-allocation unknown-mapping ${placementColor(p)}`} data-key={`unmapped:${p.id}`} data-placement={p.id} aria-expanded={p.id === view.placement} onclick={() => select(p, true)}><span class="allocation-name">{p.label}</span><span class="allocation-count">{p.device_count} GPU{p.device_count === 1 ? "" : "s"}</span></button>{/each}
        {#if unmapped.length}<span class="muted micro">Device mapping unavailable</span>{:else if !devices.length && item.gpu_capacity}<p class="empty-state">GPUs without DRA devices</p>{/if}</div>
        <span class="node-duty" title="Recorded GPU allocation time over the selected window">{duty(item, placements, range)}</span>
      </div>
      {#if item.name === view.node}
        {@const label = `${acceleratorLabel(item)} · ${shortNodeName(item.name, inventory.cluster.nodes)}`}
        <section class="allocation-expansion" id="placement-detail"><div class="allocation-detail-head inspector-heading"><h2 title={item.name}>{label}</h2><button type="button" class="detail-close" data-close-details aria-label="Close node details" title="Close (Esc)" onclick={close}>×</button></div>
          {#if devices.length > 1}<div class="allocation-device-picker" aria-label="GPU selection">
            <button type="button" class="chip" data-device="all" aria-pressed={view.gpu === "all"} onclick={() => { keepViewport(); view.gpu = "all"; }}>All GPUs</button>
            <!-- GPU labels wait for the first metrics response, so the centered numbers don't slide when utilization arrives. -->
            {#each devices as device (device.id)}{@const value = lastValue(deviceSeries((metrics.devices || []).filter((d) => d.id === device.id), "utilization"), range)}<button type="button" class="chip" data-device={device.id} title={`GPU ${deviceLabel(device.name)}`} aria-pressed={view.gpu === device.id} onclick={() => { keepViewport(); view.gpu = device.id; }}>{#if series.data || series.error}{deviceLabel(device.name)}{Number.isFinite(value) ? ` · ${Math.round(value)}%` : ""}{/if}</button>{/each}
          </div>{/if}
          <div class="node-activity" data-key="node-timeline"><Activity placements={visible} {range} {inventory} selected={view.placement} onselect={select} {back} activeOnly showSource={false} navigating={!!frozen} /></div>
          {#if item.gpu_capacity}<div class="node-gpu-detail" data-key="node-gpu-detail"><div><Chart title={`GPU utilization · ${label}`} unit="%" points={gpuPoints} start={range.start} end={range.now} min={0} max={100} gapSeconds={gapFor(gpuPoints, 60)} empty={series.error || metrics.reason || selectedDevices.find((d) => d.reason)?.reason || "No samples for this GPU"} note={failures.length ? `${failures.join(" · ")}${series.error ? " · Showing previously fetched samples" : ""}` : ""} navigating={!!frozen} />
            {#if series.error}<button type="button" class="chip" onclick={series.retry}>Retry GPU metrics</button>{/if}
          </div><div class="allocation-detail-footer"><span>{activitySource(visible)}</span><span class="allocation-memory">GPU memory <strong>{Number.isFinite(gpuMemory) ? `${(gpuMemory / 1024).toFixed(1)} GiB` : "—"}</strong></span></div></div>{/if}
          {#if placements.length || metrics.workers?.length}<div class="node-gpu-detail" data-key="node-worker-detail"><Chart title={`Worker CPU${metrics.workers.length ? ` · ${metrics.workers.length} process${metrics.workers.length === 1 ? "" : "es"}` : ""}`} unit="cores" points={cpuPoints} start={range.start} end={range.now} min={0} gapSeconds={gapFor(cpuPoints, 180)} empty={metrics.reason || metrics.workers.find((w) => w.reason)?.reason || "No CPU samples for these workers"} navigating={!!frozen} /><div class="allocation-detail-footer"><span>{sourceNote}</span><span class="allocation-memory">Worker memory <strong>{Number.isFinite(workerMemory) ? `${(workerMemory / 2 ** 30).toFixed(1)} GiB` : "—"}</strong></span></div></div>{/if}
        </section>
      {/if}
    </div>
  {:else}{#if loading}<p class="empty-state">Loading activity…</p>{:else if !errors.length && inventory.cluster.available}<p class="empty-state">No nodes reported by Kubernetes</p>{/if}{/each}
</div>
