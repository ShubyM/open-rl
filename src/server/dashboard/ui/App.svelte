<script>
  import { onMount } from "svelte";
  import { app, view, saved, parseView, currentView, createQuery, inspectUrl } from "./state.svelte.js";
  import Run from "./Run.svelte";
  import Nodes from "./Nodes.svelte";
  import Views from "./Views.svelte";

  const inventory = createQuery(() => inspectUrl("inventory"), { poll: 10000 });
  $effect(() => { if (inventory.data) app.inventory = inventory.data; });
  $effect(() => {
    const hash = currentView();
    if (location.hash !== hash) history.replaceState(history.state, "", hash);
    if (view.page === "overview" || view.page === "nodes") saved[view.page] = hash;
    if (view.page === "run" && view.back) saved.nodes = view.back;
  });
  const section = $derived(view.page === "run" ? view.back ? "nodes" : "overview" : view.page);
  const connection = $derived(inventory.error ? app.inventory ? "Stale snapshot" : "Unavailable" : !app.inventory ? "Connecting…" : app.inventory.cluster.available ? "Connected" : "Cluster unavailable");
  onMount(() => {
    const navigate = () => {
      const next = parseView();
      if (view.page !== next.page || view.id !== next.id) window.scrollTo(0, 0);
      Object.assign(view, next);
    };
    const visible = () => { if (!document.hidden) inventory.retry(); };
    window.addEventListener("hashchange", navigate);
    document.addEventListener("visibilitychange", visible);
    return () => { window.removeEventListener("hashchange", navigate); document.removeEventListener("visibilitychange", visible); };
  });
</script>

<div class="product">
  <header class="appbar">
    <a href={saved.overview} class="brand">OpenRL</a>
    <nav aria-label="Main">
      {#each [["overview", "Overview"], ["nodes", "Nodes"], ["health", "Health"]] as [page, label]}
        <a href={saved[page] || `#${page}`} aria-current={section === page ? "page" : undefined}>{label}</a>
      {/each}
    </nav>
    <span id="connection" role="status" title={inventory.error}>{connection}</span>
  </header>
  <div id="snapshot-error" class="snapshot-error" role="status" hidden={!inventory.error}>
    <span>{inventory.error}{app.inventory ? ` Showing the snapshot from ${app.inventory.observed_at}.` : ""}</span>
    <button type="button" class="chip" data-retry-snapshot onclick={inventory.retry}>Retry</button>
  </div>
  <main id="content" class="pad">
    {#if app.inventory}
      {#if view.page === "run"}
        {#key view.id}<Run inventory={app.inventory} {view} />{/key}
      {:else if view.page === "nodes"}<Nodes inventory={app.inventory} {view} />
      {:else}<Views inventory={app.inventory} {view} />{/if}
    {:else if !inventory.error}Loading cluster…{/if}
  </main>
</div>
