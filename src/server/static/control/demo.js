(() => {
  "use strict";

  const NS = "simulation";
  const IMAGE = "ghcr.io/open-rl/server:demo";
  const stoppedRuns = new Set();
  const ago = (seconds) => new Date(Date.now() - seconds * 1000).toISOString();
  const list = (value) => Array.isArray(value) ? value : [];
  const object = (value) => value && typeof value === "object" && !Array.isArray(value) ? value : {};

  function component(runId, role, status, phase, node, extra = {}) {
    return {
      id: `sim-${role}`,
      role,
      model_id: runId,
      status,
      phase,
      node,
      pod_name: `sim-${role}-${runId.slice(4)}`,
      namespace: NS,
      ready: ["ready", "running", "waiting"].includes(status),
      restarts: 0,
      image: IMAGE,
      updated_at: ago(4),
      simulated: true,
      ...extra,
    };
  }

  function run(id, name, status, phase, seconds, components, options = {}) {
    return {
      id,
      name: `Demo · ${name}`,
      base_model: options.base_model || "Qwen/Qwen2.5-0.5B",
      training_kind: options.training_kind || "full",
      status,
      phase,
      message: options.message || `SIMULATION: ${phase.replaceAll("_", " ")}`,
      created_at: ago(seconds),
      updated_at: ago(options.updatedAgo ?? 4),
      elapsed_seconds: seconds,
      queue: options.queue || { training: 0, sampling: 0 },
      components,
      can_stop: !["completed", "failed", "stopped"].includes(status),
      simulated: true,
    };
  }

  function applyStopped(item) {
    if (!stoppedRuns.has(item.id)) return item;
    return {
      ...item,
      status: "stopped",
      phase: "stopped",
      message: "SIMULATION: stopped from the control UI",
      can_stop: false,
      components: item.components.map((part) => ({
        ...part,
        status: "completed",
        phase: "stopped",
        ready: false,
        message: "SIMULATION: worker stopped from the control UI",
      })),
    };
  }

  function runs() {
    const math = "sim-run-math-rl";
    const code = "sim-run-code-rl";
    const warmup = "sim-run-sampler-warmup";
    const queued = "sim-run-queued";
    const oom = "sim-run-oom";
    const complete = "sim-run-complete";
    return [
      run(
        math,
        "math policy RL",
        "running",
        "forward_backward",
        1128,
        [
          component(math, "trainer", "running", "forward_backward", "sim-a100-01", { message: "SIMULATION: training step 18/100" }),
          component(math, "sampler", "ready", "sample_complete", "sim-h100-01", { message: "SIMULATION: sampler serving policy revision 18" }),
        ],
        { message: "SIMULATION: training step 18/100", queue: { training: 2, sampling: 0 } },
      ),
      run(
        code,
        "code reasoning RL",
        "waiting",
        "waiting_for_gpu",
        684,
        [
          component(code, "trainer", "waiting", "waiting_for_gpu", "sim-h100-01", { message: "SIMULATION: waiting behind math policy sampler" }),
          component(code, "sampler", "ready", "ready", "sim-a100-01", { message: "SIMULATION: sampler ready" }),
        ],
        { message: "SIMULATION: waiting 41s for an accelerator slice", queue: { training: 4, sampling: 1 } },
      ),
      run(
        warmup,
        "sampler warmup",
        "starting",
        "initializing_engine",
        96,
        [
          component(warmup, "trainer", "ready", "ready", "sim-a100-01", { message: "SIMULATION: trainer registered" }),
          component(warmup, "sampler", "starting", "initializing_engine", "sim-h100-01", { message: "SIMULATION: loading model shards 7/10" }),
        ],
        { message: "SIMULATION: loading sampler model shards 7/10", queue: { training: 0, sampling: 3 } },
      ),
      run(
        queued,
        "queued experiment",
        "queued",
        "scheduling_trainer",
        213,
        [
          component(queued, "scheduler", "queued", "scheduling_trainer", null, { pod_name: null, image: null, message: "SIMULATION: looking for 8 free GPUs" }),
          component(queued, "trainer", "pending", "pending", null, {
            ready: false,
            reason: "Unschedulable",
            message: "SIMULATION: 0/4 nodes have 8 free GPUs",
          }),
        ],
        { message: "SIMULATION: queued; no node currently has 8 free GPUs", queue: { training: 7, sampling: 0 } },
      ),
      run(
        oom,
        "failed long-context run",
        "failed",
        "forward_backward_failed",
        917,
        [
          component(oom, "trainer", "failed", "failed", "sim-spot-01", {
            ready: false,
            restarts: 2,
            reason: "OOMKilled",
            message: "SIMULATION: trainer exceeded its 76 GiB memory limit",
          }),
          component(oom, "sampler", "completed", "succeeded", "sim-spot-01", { ready: false, message: "SIMULATION: sampler exited after trainer failure" }),
        ],
        { message: "SIMULATION: trainer was OOMKilled", updatedAgo: 128 },
      ),
      run(
        complete,
        "completed reward run",
        "completed",
        "completed",
        1542,
        [
          component(complete, "trainer", "completed", "succeeded", "sim-a100-01", { ready: false, message: "SIMULATION: trainer completed 100 steps" }),
          component(complete, "sampler", "completed", "succeeded", "sim-a100-01", { ready: false, message: "SIMULATION: sampler shut down cleanly" }),
        ],
        { message: "SIMULATION: completed 100/100 steps", updatedAgo: 420 },
      ),
    ].map(applyStopped);
  }

  function stop(runId) {
    const item = runs().find((candidate) => candidate.id === runId);
    if (!item?.can_stop) return false;
    stoppedRuns.add(runId);
    return true;
  }

  function nodes() {
    const node = (name, ready, roles, gpu, cpu) => ({
      name,
      status: ready ? "ready" : "not_ready",
      ready,
      roles,
      capacity: { cpu: String(cpu), memory: ready ? "128Gi" : "80Gi", pods: "110", ...(gpu ? { "nvidia.com/gpu": String(gpu) } : {}) },
      allocatable: { cpu: String(cpu - 1), memory: ready ? "120Gi" : "76Gi", pods: "100", ...(gpu ? { "nvidia.com/gpu": String(gpu) } : {}) },
      conditions: [{ type: "Ready", status: ready ? "True" : "False", reason: ready ? "SimulatedReady" : "SimulatedSpotPreemption", message: ready ? "SIMULATION: node is ready" : "SIMULATION: spot node was preempted", simulated: true }],
      taints: name.includes("spot") ? [{ key: "demo.open-rl.dev/spot", value: "true", effect: "NoSchedule", simulated: true }] : [],
      labels: { "open-rl.dev/simulated": "true" },
      pod_count: 0,
      simulated: true,
    });
    return [
      node("sim-control-01", true, ["control-plane"], 0, 16),
      node("sim-a100-01", true, ["worker", "gpu"], 8, 64),
      node("sim-h100-01", true, ["worker", "gpu"], 8, 96),
      node("sim-spot-01", false, ["worker", "gpu", "spot"], 4, 32),
    ];
  }

  function platformPod(name, role, node, status = "ready", extra = {}) {
    return {
      name,
      pod_name: name,
      namespace: NS,
      node,
      status,
      phase: status === "ready" ? "running" : status,
      message: extra.message || `SIMULATION: ${role} ${status}`,
      reason: extra.reason || null,
      ready: status === "ready",
      restarts: extra.restarts || 0,
      image: role === "store" ? "redis:7-demo" : IMAGE,
      role,
      model_id: null,
      labels: { app: `sim-open-rl-${role}`, "open-rl.dev/simulated": "true" },
      started_at: Date.now() / 1000 - 3600,
      updated_at: Date.now() / 1000,
      simulated: true,
    };
  }

  function workloadPods() {
    return runs().flatMap((item) => item.components
      .filter((part) => ["trainer", "sampler"].includes(part.role))
      .map((part) => ({
        ...part,
        name: part.pod_name,
        labels: {
          app: `open-rl-${part.role}-worker`,
          "open-rl.dev/simulated": "true",
          "timeslice.io/group": `${part.role}s`,
        },
        status: part.status === "starting" || part.status === "waiting" ? "running" : part.status === "queued" ? "pending" : part.status,
        started_at: Date.now() / 1000 - 600,
        updated_at: Date.now() / 1000,
        simulated: true,
      })));
  }

  function summarize(nodesList, pods) {
    const pending = pods.filter((pod) => pod.status === "pending");
    return {
      nodes: nodesList.length,
      ready_nodes: nodesList.filter((node) => node.ready).length,
      pods: pods.length,
      running_pods: pods.filter((pod) => ["running", "ready"].includes(pod.status)).length,
      pending_pods: pending.length,
      actionable_pending_pods: pending.filter((pod) => Boolean(pod.reason)).length,
      failed_pods: pods.filter((pod) => pod.status === "failed").length,
    };
  }

  function augmentCluster(liveCluster) {
    const live = object(liveCluster);
    const demoNodes = nodes();
    const demoPods = [
      platformPod("sim-gateway-0", "gateway", "sim-control-01"),
      platformPod("sim-redis-0", "store", "sim-control-01"),
      platformPod("sim-scheduler-0", "scheduler", "sim-control-01"),
      platformPod("sim-timeslicer-a100", "timeslicer", "sim-a100-01"),
      platformPod("sim-timeslicer-h100", "timeslicer", "sim-h100-01"),
      platformPod("sim-timeslicer-spot", "timeslicer", "sim-spot-01", "failed", { reason: "NodeLost", message: "SIMULATION: time slicer lost its spot node" }),
      ...workloadPods(),
    ];
    const allNodes = [...list(live.nodes), ...demoNodes];
    const allPods = [...list(live.pods), ...demoPods];
    const podCounts = allPods.reduce((counts, pod) => ({ ...counts, ...(pod.node ? { [pod.node]: (counts[pod.node] || 0) + 1 } : {}) }), {});
    const mergedNodes = allNodes.map((node) => ({ ...node, pod_count: podCounts[node.name] || 0 }));
    const summary = summarize(mergedNodes, allPods);
    return {
      ...live,
      status: summary.ready_nodes < summary.nodes || summary.failed_pods || summary.actionable_pending_pods ? "degraded" : (live.status || "healthy"),
      summary,
      nodes: mergedNodes,
      pods: allPods,
      errors: list(live.errors),
      generated_at: new Date().toISOString(),
      simulated: true,
    };
  }

  window.OpenRLDemo = { runs, augmentCluster, stop };
})();
