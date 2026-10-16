<script>
  import { onDestroy } from "svelte";
  import { copyLink, currentView } from "./state.svelte.js";
  let { href } = $props(), label = $state("Copy link"), timer;
  async function copy() {
    label = await copyLink(new URL(href || currentView(true), location.href).href);
    clearTimeout(timer);
    timer = setTimeout(() => label = "Copy link", 2000);
  }
  onDestroy(() => clearTimeout(timer));
</script>
<button type="button" class="chip" data-copy-view onclick={copy}>{label}</button>
