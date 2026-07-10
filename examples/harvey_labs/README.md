# Harvey LAB RL

Live-rollout RL scaffold for Harvey's Legal Agent Benchmark.

The training path reuses tinker-cookbook's GRPO loop and multi-turn tool environment. This module only adapts LAB tasks, sandbox tools, rubric reward, and Gemma 4 tool-call rendering.

## Layout

- `train.py`: small LAB-specific config mapped into `tinker_cookbook.rl.train`.
- `env.py`: LAB task loading, sandbox env construction, and dataset builders.
- `tools.py`: thin LAB `ToolExecutor` adapter for tinker-cookbook.
- `reward.py`: terminal rubric reward wrapper around LAB's judge.
- `renderer.py`: Gemma 4 chat renderer with XML-style tool calls.
- `score_lab_run.py`: helper executed inside the LAB checkout/venv for rubric scoring.

## Commands

Phase 0 backend smoke, from repo root on `box` with gateway at `:9003`:

```bash
export PATH=$PATH:$HOME/.local/bin
TINKER_API_KEY=tml-dummy-key TINKER_BASE_URL=http://127.0.0.1:9003 \
uv --project examples run python examples/autoresearch/recipes/math_rl/train_gemma.py \
  model_name=google/gemma-4-E4B-it renderer_name=gemma4 env=gsm8k \
  group_size=2 groups_per_batch=1 max_steps=1 max_tokens=128 \
  base_url=http://127.0.0.1:9003 save_every=0 eval_every=0 \
  behavior_if_log_dir_exists=delete log_path=artifacts/harvey-labs/phase0-math
```

Tiny LAB run:

```bash
export PATH=$PATH:$HOME/.local/bin
TINKER_API_KEY=tml-dummy-key TINKER_BASE_URL=http://127.0.0.1:9003 \
uv --project examples run python examples/harvey_labs/train.py \
  base_url=http://127.0.0.1:9003 \
  lab_root=experiments/lab-traces/harvey-labs \
  train_limit=1 eval_limit=0 \
  max_reward_criteria=3 log_path=artifacts/harvey-labs/lab-tiny
```

By default, training uses a fixed bootstrap curriculum of 40 distinct LAB tasks
with small source-document sets, including the task completed in the baseline
smoke test. Exact duplicate document sets are excluded and no practice area
contributes more than four tasks. Pass `task=<category/task>` to run a specific
LAB task instead. Each rollout is trained as a separate microbatch and is
capped at 32K trajectory tokens, with at most 3K generated tokens per tool turn.
The 32K cap completed a full Gemma 4 E4B FFT update on one 80GB H100; a 64K
trajectory did not. The 3K generation budget lets Gemma finish its reasoning and
tool call in one stock cookbook environment step rather than introducing a
custom continuation policy.
The renderer also recovers schema-valid calls when Gemma places final prose
before the call or ends a brace-complete call to a declared tool with EOS instead
of its terminator. Ambiguous or incomplete argument objects remain parse errors.

LAB uses the instruction-tuned `google/gemma-4-E4B-it` checkpoint and its native
function-calling template. Until the upstream fix is merged, the renderer pins
the canonical template from [Gemma 4 discussion #36](https://huggingface.co/google/gemma-4-E4B-it/discussions/36)
at commit `4e34fcbc4c9a95b92d6a8a97c2faed16dd783f91`; this fixes null arguments,
multi-turn tool-call closure, and preservation of reasoning in tool chains. The
renderer enables Gemma's thinking channel for live rollouts and reconstructs
the template's pre-opened thought channel when parsing post-tool continuations.
The environment reuses LAB's system prompt, default
`docx`/`pptx`/`xlsx` skill manuals and scripts, six sandbox tools, and rubric
judge without redefining their behavior. The tools keep their canonical names
(`bash`, `read`, `write`, `edit`, `glob`, and `grep`) so their schemas match
LAB's system prompt and executor. As in stock LAB, a response with no tool call
finishes the episode; Open-RL does not add a separate terminal tool or duplicate
LAB's workspace instructions.

Individual tool observations are capped at 8K tokens. Line-based oversized
`read` results include the exact next offset, so the model can continue with
LAB's native `offset`/`limit` arguments or use `grep`; a single large document
can no longer consume the entire 32K trajectory by itself. The underlying read
still executes normally and counts toward LAB's document-coverage metrics.

For tasks that name output files, Open-RL appends a short task-specific path
contract: `write` receives bare relative names, while shell and skill commands
use absolute `/workspace/output/...` paths. It also repeats the exact requested
filenames and requires binary validation before stopping; it does not add task
content or alter LAB's tools.

Reward is 80% LAB rubric score and 20% bounded process progress. Process credit
comes only from unique known source documents read, a non-empty output, and a
structurally valid output of a requested file type. Repeated tool calls, empty
files, and markdown renamed to `.docx` do not earn additional credit. This gives
GRPO signal before the base policy can reliably complete a perfect deliverable.

Run the repository unit tests with:

```bash
make test unit
```
