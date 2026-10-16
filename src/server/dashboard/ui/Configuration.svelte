<script>
  let { settings = {} } = $props();
  const labels = { fine_tuning_type: "Training", trainer_backend: "Trainer", trainer_gpus: "Trainer GPUs", trainer_cp: "Context parallelism", sampler_replicas: "Sampler replicas", weight_sync: "Weight sync", exclusive: "Exclusive allocation" };
  const names = { lora: "LoRA", full: "Full", delta: "Delta", pytorch: "PyTorch", automodel: "Automodel" };
  const value = (v) => v == null ? "Not recorded" : typeof v === "boolean" ? v ? "Yes" : "No" : typeof v === "object" ? JSON.stringify(v) : Object.hasOwn(names, v) ? names[v] : String(v);
  const rows = $derived(Object.entries(settings || {}));
  const summary = $derived([
    settings?.["openrl.trainer_backend"] == null ? null : value(settings["openrl.trainer_backend"]),
    settings?.["openrl.trainer_gpus"] == null ? null : `${settings["openrl.trainer_gpus"]} trainer GPU${settings["openrl.trainer_gpus"] === 1 ? "" : "s"}`,
    settings?.["openrl.sampler_replicas"] == null ? null : `${settings["openrl.sampler_replicas"]} sampler${settings["openrl.sampler_replicas"] === 1 ? "" : "s"}`,
  ].filter(Boolean).join(" · "));
</script>

<details class="run-settings">
  <summary>Configuration <span class="muted">{summary || "Not recorded"}</span></summary>
  {#if rows.length}
    <p class="muted micro">Saved model settings. Worker environment and per-step parameters are separate.</p>
    <dl>{#each rows as [name, v]}{@const key = name.replace(/^openrl\./, "")}<dt title={name}>{Object.hasOwn(labels, key) ? labels[key] : name}</dt><dd>{value(v)}</dd>{/each}</dl>
    <details><summary>Raw JSON</summary><pre>{JSON.stringify(settings, null, 2)}</pre></details>
  {:else}<p class="muted">No configuration recorded for this run.</p>{/if}
</details>
