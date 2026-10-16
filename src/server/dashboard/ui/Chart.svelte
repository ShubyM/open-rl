<script>
  import { chartNumber, chartTime, valueText } from "./visualization.js";
  import { crosshair, crosshairOffset } from "./state.svelte.js";
  let { title, unit = "", points = [], start, end, min, max, tone = "neutral", gapSeconds = Infinity, empty = "No samples in this time range", note = "", navigating = false } = $props();
  let plot = $state(null), svg = $state(null), tip = $state(null);
  let hover = $state(null);
  const samples = $derived(points.filter(([at]) => Number.isFinite(at) && at >= start && at <= end).sort((a, b) => a[0] - b[0]));
  const data = $derived(samples.filter(([, value]) => Number.isFinite(value)));
  const values = $derived(data.map(([, value]) => value));
  const scale = $derived.by(() => {
    let low = Number.isFinite(min) ? min : Math.min(...values), high = Number.isFinite(max) ? max : Math.max(...values);
    if (!values.length) return { low: 0, high: 1, ticks: [] };
    if (high <= low) { const pad = Math.max(Math.abs(low) * 0.1, 0.1); if (!Number.isFinite(min)) low -= pad; high += pad; }
    let ticks = [high, (high + low) / 2, low];
    if (!Number.isFinite(min) && !Number.isFinite(max)) {
      const raw = (high - low) / 3, magnitude = 10 ** Math.floor(Math.log10(raw));
      const step = [1, 2, 5, 10].find((n) => n >= raw / magnitude) * magnitude;
      low = Math.floor(low / step) * step; high = Math.ceil(high / step) * step;
      ticks = Array.from({ length: Math.round((high - low) / step) + 1 }, (_, i) => Number((high - i * step).toPrecision(12)));
    }
    return { low, high, ticks };
  });
  const segments = $derived.by(() => {
    const result = [];
    let current;
    for (const sample of samples) {
      if (!Number.isFinite(sample[1])) { current = null; continue; }
      if (!current || sample[0] - current.at(-1)[0] > gapSeconds) result.push(current = []);
      current.push(sample);
    }
    return result;
  });
  const spans = $derived(segments.map((segment) => [segment[0][0], segment.at(-1)[0]]));
  const tickCount = $derived(end === start ? 0 : 4);
  const x = (at) => ((at - start) / (end - start || 1) * 1000).toFixed(1);
  const y = (value) => (1000 - (Math.max(scale.low, Math.min(scale.high, value)) - scale.low) / (scale.high - scale.low) * 1000).toFixed(1);
  const path = (segment) => segment.map(([at, value], i) => `${i ? "L" : "M"}${x(at)},${y(value)}`).join(" ");
  const label = (at) => `${end - start >= 86400 ? new Date(at * 1000).toISOString().slice(5, 10) + " " : ""}${chartTime(at, end - start <= 120)}`;
  const crosshairX = $derived(crosshairOffset({ start, now: end }));
  $effect(() => { points; start; end; min; max; navigating; hover = null; });
  function inspect(index) {
    if (!data.length || navigating) return;
    index = Math.max(0, Math.min(data.length - 1, index));
    const [at, value] = data[index], box = svg.getBoundingClientRect();
    const left = (at - start) / (end - start || 1) * box.width;
    hover = { index, left: box.left - plot.getBoundingClientRect().left + left, top: Math.max(0, Math.min(100, (scale.high - value) / (scale.high - scale.low) * 100)), tipLeft: Math.max(-left, Math.min(8, box.width - left - (tip?.offsetWidth || 160))), text: `${chartTime(at, true)} · ${valueText(value, unit)}` };
  }
  function pointer(event) {
    const box = svg.getBoundingClientRect(), at = start + (event.clientX - box.left) / (box.width || 1) * (end - start);
    const outside = event.clientX < box.left || event.clientX > box.right;
    crosshair.at = navigating || event.buttons || outside ? null : at;
    if (!data.length || navigating || event.buttons) { hover = null; return; }
    let left = 0, right = data.length - 1;
    while (left < right) { const middle = Math.floor((left + right) / 2); if (data[middle][0] < at) left = middle + 1; else right = middle; }
    const index = left > 0 && Math.abs(data[left - 1][0] - at) < Math.abs(data[left][0] - at) ? left - 1 : left;
    const near = Math.abs(data[index][0] - at) / (end - start || 1) * box.width < 6;
    if (outside || (!near && !spans.some(([a, b]) => at >= a && at <= b))) hover = null;
    else inspect(index);
  }
  function keyboard(event) {
    if (!["ArrowLeft", "ArrowRight", "Home", "End", "Escape"].includes(event.key)) return;
    event.preventDefault();
    event.stopPropagation();
    if (event.key === "Escape") { hover = null; plot.blur(); }
    else inspect(event.key === "Home" ? 0 : event.key === "End" ? data.length - 1 : (hover?.index ?? data.length - 1) + (event.key === "ArrowLeft" ? -1 : 1));
  }
</script>
<svelte:window onresize={() => hover = null} />
<figure class="chart" data-key={title} data-tone={tone}>
  <figcaption class="chart-head"><h2>{title}</h2>{#if data.length}<span class="chart-latest" title={new Date(data.at(-1)[0] * 1000).toISOString()}><span>Latest</span> <strong>{valueText(data.at(-1)[1], unit)}</strong></span>{/if}</figcaption>
  {#if !data.length}<p class="chart-empty">{empty}</p>
  {:else}
    <!-- svelte-ignore a11y_no_noninteractive_tabindex, a11y_no_noninteractive_element_interactions (Arrow keys inspect the chart samples.) -->
    <div class="chart-plot" bind:this={plot} tabindex="0" role="group" aria-label={`${title}. Use left and right arrow keys to inspect samples.`} onpointermove={pointer} onpointerleave={() => hover = crosshair.at = null} onfocusout={() => hover = null} onkeydown={keyboard}>
      <div class="chart-y">{#each scale.ticks as tick}<span>{chartNumber(tick)}</span>{/each}</div>
      <svg bind:this={svg} viewBox="0 0 1000 1000" preserveAspectRatio="none" role="img" aria-label={`${title}. Latest ${valueText(data.at(-1)[1], unit)}.`}>
        {#each scale.ticks as tick}<line class="chart-grid" x1="0" x2="1000" y1={y(tick)} y2={y(tick)} vector-effect="non-scaling-stroke" />{/each}
        {#each segments as segment}{#if segment.length > 1}<path class="chart-area" d={`${path(segment)} L${x(segment.at(-1)[0])},1000 L${x(segment[0][0])},1000 Z`} />{/if}{/each}
        {#each segments as segment}<path class={segment.length === 1 ? "chart-dot" : "chart-line"} d={segment.length === 1 ? `M${x(segment[0][0])},${y(segment[0][1])} h0.01` : path(segment)} vector-effect="non-scaling-stroke" />{/each}
        {#if !hover && crosshairX !== null}<line class="chart-crosshair" x1={crosshairX * 10} x2={crosshairX * 10} y1="0" y2="1000" vector-effect="non-scaling-stroke" />{/if}
      </svg>
      <div class="chart-x">{#each Array.from({ length: tickCount + 1 }, (_, i) => start + (end - start) * i / (tickCount || 1)) as at}<span>{label(at)}</span>{/each}</div>
      <div class="chart-hover" hidden={!hover} style:left={`${hover?.left || 0}px`}><div class="chart-cursor"></div><div class="chart-point" style:top={`${hover?.top || 0}%`}></div><div class="chart-tip" bind:this={tip} style:left={`${hover?.tipLeft || 0}px`}>{hover?.text || ""}</div></div>
    </div>
    <div class="chart-summary"><span title="Arithmetic mean of the reported samples">Average <strong>{valueText(values.reduce((sum, value) => sum + value, 0) / values.length, unit)}</strong></span><span>Peak <strong>{valueText(Math.max(...values), unit)}</strong></span>{#if note}<span class="chart-note" role="status">{note}</span>{/if}</div>
  {/if}
</figure>
