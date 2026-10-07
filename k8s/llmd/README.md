# LoRA sampling through llm-d

The gateway can send a LoRA model's sampling to an llm-d router instead of the
Redis queue. The router sends each request to the vLLM replica that already
caches the longest prefix of its prompt, so an agent's turns keep landing
where their history is cached. The pool is shared by every LoRA job on its
base model.

```
gateway --/v1/completions--> router (Envoy + endpoint picker) --> vLLM replica
   |                                                                  |
   +-- writes each sampler version to /mnt/shared/open-rl/llmd-adapters <-- loads it by name
```

## Install

Both pieces go in `openrl-system`, next to the gateway and its shared volume.

1. The vLLM pool. Edit the model, its flags and `replicas` in `pool.yaml` first.

   ```sh
   kubectl apply -f k8s/llmd/pool.yaml
   ```

2. The router.

   ```sh
   helm template openrl-llmd oci://ghcr.io/llm-d/charts/llm-d-router-standalone \
     --version v0.11.0 -n openrl-system -f k8s/llmd/router.values.yaml \
     | kubectl apply -n openrl-system -f -
   ```

   The router Service is `openrl-llmd-epp`, port 80.

3. Point the gateway at it, one entry per base model the pool serves:

   ```sh
   kubectl -n openrl-system set env deploy/open-rl-gateway \
     OPEN_RL_LLMD_ROUTERS='{"Qwen/Qwen3.5-9B":"http://openrl-llmd-epp.openrl-system.svc:80"}'
   ```

LoRA jobs on a mapped base model then sample through the router and launch no
queue samplers. Full fine-tuning keeps the queue path.

## Adapters

When the client retrieves a `save_weights_for_sampler` result, the gateway
copies the adapter into `llmd-adapters/<name>`, a name unique to that version,
before the client can sample it. Replicas load it on first use with vLLM's
filesystem LoRA resolver. A version is deleted once a newer one exists and no
request uses it, and all of a job's versions go when the job is deleted. The
bookkeeping lives in the gateway's memory, so run one gateway replica.
