import asyncio
from difflib import SequenceMatcher
import logging
import os
import random
import statistics
from pathlib import Path
from typing import Any, cast

import chz
import tinker
from datasets import load_dataset
from tinker import types
from tinker_cookbook.utils import ml_log

from texttosql_sft import (
    BASE_URL,
    DATASET,
    build_examples,
    clean_sql_for_execution,
    evaluate,
    normalize_sql,
    require_server,
    sql_results_match,
)

LOG_DIR = Path(__file__).resolve().parent / "artifacts" / "texttosql_sft_grpo_{preset}"

os.environ.setdefault("TINKER_API_KEY", "tml-dummy-key")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


@chz.chz
class Config:
    base_model: str
    tokenizer_name: str
    rank: int
    prompt_format: str = "plain_sql_completion"
    phase: str = "full"
    base_url: str = os.getenv("TINKER_BASE_URL") or os.getenv("OPEN_RL_BASE_URL") or BASE_URL
    seed: int = 30
    grad_clip_norm: float = 0.3
    dataset_limit: int = 12_500
    train_limit: int = 100
    rl_train_limit: int = 64
    eval_limit: int = 25
    eval_max_tokens: int = 64
    log_dir: str = str(LOG_DIR)
    sft_steps: int = 100
    sft_batch_size: int = 1
    sft_learning_rate: float = 5e-5
    sft_eval_every: int = 100
    rl_steps: int = 40
    rl_prompts_per_step: int = 4
    rl_samples_per_prompt: int = 4
    rl_learning_rate: float = 1e-5
    rl_temperature: float = 0.8
    rl_max_tokens: int = 64
    rl_eval_every: int = 10
    rl_loss_fn: str = "ppo"
    rl_clip_range: float = 0.2
    compile_reward: float = 0.25
    match_reward: float = 1.0
    compile_error_penalty: float = -0.5
    resume_state_path: str = ""
    resume_with_optimizer: bool = False
    save_sft_state_name: str = ""
    save_final_state_name: str = ""


PRESETS = {
    "gemma4_e2b": chz.Blueprint(Config).apply(
        {
            "base_model": "google/gemma-4-e2b",
            "tokenizer_name": "google/gemma-4-e2b",
            "rank": 32,
            "train_limit": 100,
            "rl_train_limit": 64,
            "eval_limit": 25,
            "sft_steps": 100,
            "sft_batch_size": 1,
            "sft_learning_rate": 5e-5,
            "sft_eval_every": 100,
            "rl_steps": 40,
            "rl_prompts_per_step": 4,
            "rl_samples_per_prompt": 4,
            "rl_learning_rate": 1e-5,
            "rl_temperature": 0.8,
            "rl_eval_every": 10,
            "rl_loss_fn": "ppo",
        },
        layer_name="gemma4_e2b",
    ),
    "gemma4_e2b_smoke": chz.Blueprint(Config).apply(
        {
            "base_model": "google/gemma-4-e2b",
            "tokenizer_name": "google/gemma-4-e2b",
            "rank": 16,
            "train_limit": 16,
            "rl_train_limit": 8,
            "eval_limit": 4,
            "sft_steps": 4,
            "sft_batch_size": 1,
            "sft_learning_rate": 5e-5,
            "sft_eval_every": 2,
            "rl_steps": 2,
            "rl_prompts_per_step": 2,
            "rl_samples_per_prompt": 3,
            "rl_learning_rate": 1e-5,
            "rl_temperature": 0.7,
            "rl_eval_every": 1,
            "rl_loss_fn": "ppo",
        },
        layer_name="gemma4_e2b_smoke",
    ),
}


def group_relative_advantages(rewards: list[float]) -> list[float]:
    if len(rewards) < 2:
        return [0.0] * len(rewards)

    reward_mean = statistics.fmean(rewards)
    reward_std = statistics.pstdev(rewards)
    if reward_std < 1e-8:
        return [0.0] * len(rewards)
    return [(reward - reward_mean) / reward_std for reward in rewards]


def make_rl_datum(
    prompt_tokens: list[int],
    completion_tokens: list[int],
    completion_logprobs: list[float],
    advantage: float,
) -> types.Datum:
    full_tokens = prompt_tokens + list(completion_tokens)
    target_tokens = full_tokens[1:]
    prompt_weight_count = max(0, len(prompt_tokens) - 1)
    completion_weight_count = len(completion_tokens)

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=full_tokens[:-1]),
        loss_fn_inputs=cast(
            Any,
            {
                "target_tokens": target_tokens,
                "weights": types.TensorData.from_list(
                    [0.0] * prompt_weight_count + [1.0] * completion_weight_count,
                    dtype="float32",
                ),
                "logprobs": types.TensorData.from_list(
                    [0.0] * prompt_weight_count + list(completion_logprobs),
                    dtype="float32",
                ),
                "advantages": types.TensorData.from_list(
                    [0.0] * prompt_weight_count + [advantage] * completion_weight_count,
                    dtype="float32",
                ),
            },
        ),
    )


def compute_sql_reward(
    example: dict[str, Any],
    predicted_sql: str,
    *,
    compile_reward: float,
    match_reward: float,
    compile_error_penalty: float,
) -> dict[str, Any]:
    execution_match, execution_error = sql_results_match(
        example["context"],
        predicted_sql,
        example["target"],
        target_rows=example["target_rows"],
    )
    compiles = execution_error is None
    total_reward = compile_error_penalty
    if compiles:
        total_reward = compile_reward
        if execution_match:
            total_reward += match_reward

    normalized_prediction = normalize_sql(predicted_sql)
    normalized_target = normalize_sql(example["target"])
    similarity = SequenceMatcher(None, normalized_prediction, normalized_target).ratio()

    return {
        "total": total_reward,
        "compile": float(compiles),
        "execution_match": float(execution_match),
        "similarity": similarity,
        "sqlite_error": execution_error or "",
    }


def next_batch(
    examples: list[dict[str, Any]],
    order: list[int],
    batch_size: int,
    position: int,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], int]:
    if position + batch_size > len(order):
        rng.shuffle(order)
        position = 0
    batch = [examples[order[idx]] for idx in range(position, position + batch_size)]
    return batch, position + batch_size


async def snapshot_eval(
    trainer,
    client: tinker.ServiceClient,
    tokenizer,
    alias: str,
    eval_examples: list[dict[str, Any]],
    config: Config,
) -> tuple[float, float]:
    sampler_path = trainer.save_weights_for_sampler(name=alias).result().path
    sampler = client.create_sampling_client(sampler_path)
    return await evaluate(sampler, tokenizer, alias, eval_examples, config)


async def create_or_resume_trainer(client: tinker.ServiceClient, config: Config):
    if not config.resume_state_path:
        return await client.create_lora_training_client_async(
            base_model=config.base_model,
            rank=config.rank,
            seed=config.seed,
            train_mlp=True,
            train_attn=True,
            train_unembed=False,
        )

    try:
        if config.resume_with_optimizer:
            return await client.create_training_client_from_state_with_optimizer_async(config.resume_state_path)
        return await client.create_training_client_from_state_async(config.resume_state_path)
    except Exception as exc:
        raise RuntimeError(
            "Failed to resume from resume_state_path. The hosted Tinker API supports this, "
            "but this local Open-RL backend may not have checkpoint restore wired yet."
        ) from exc


async def maybe_save_state(trainer, name: str, label: str) -> str:
    if not name:
        return ""
    save_result = trainer.save_state(name).result()
    save_path = getattr(save_result, "path", "") or ""
    logging.info("%s state saved to %s", label, save_path or "<empty path>")
    return save_path


async def run_sft_phase(
    *,
    trainer,
    client: tinker.ServiceClient,
    tokenizer,
    train_examples: list[dict[str, Any]],
    eval_examples: list[dict[str, Any]],
    config: Config,
    ml_logger,
    step_offset: int,
) -> tuple[float, float]:
    if config.sft_steps <= 0:
        return await snapshot_eval(trainer, client, tokenizer, "texttosql_sft_skip", eval_examples, config)

    batch_size = min(config.sft_batch_size, len(train_examples))
    rng = random.Random(config.seed)
    order = list(range(len(train_examples)))
    rng.shuffle(order)
    position = 0
    losses: list[float] = []
    latest_exec = 0.0
    latest_sim = 0.0

    logging.info(
        "Starting SFT: steps=%s batch=%s lr=%g train_examples=%s",
        config.sft_steps,
        batch_size,
        config.sft_learning_rate,
        len(train_examples),
    )

    for local_step in range(1, config.sft_steps + 1):
        batch, position = next_batch(train_examples, order, batch_size, position, rng)
        datums = [example["datum"] for example in batch]
        active_tokens = sum(example["active_tokens"] for example in batch)

        fwdbwd_future = await trainer.forward_backward_async(datums, "cross_entropy")
        optim_future = await trainer.optim_step_async(
            types.AdamParams(
                learning_rate=config.sft_learning_rate,
                grad_clip_norm=config.grad_clip_norm,
            )
        )
        fwdbwd = await fwdbwd_future
        await optim_future

        loss = float(fwdbwd.metrics.get("loss:sum", 0.0)) / max(1, active_tokens)
        global_step = step_offset + local_step
        losses.append(loss)
        ml_logger.log_metrics({"phase": "sft_train", "loss": loss}, step=global_step)

        if local_step % config.sft_eval_every == 0 or local_step == config.sft_steps:
            alias = f"texttosql_sft_s{local_step}"
            latest_exec, latest_sim = await snapshot_eval(trainer, client, tokenizer, alias, eval_examples, config)
            ml_logger.log_metrics(
                {"phase": "sft_eval", "execution_match": latest_exec, "similarity": latest_sim},
                step=global_step,
            )
            logging.info(
                "[sft step %03d] loss=%.4f eval_exec=%.1f%% eval_similarity=%.1f%%",
                local_step,
                loss,
                latest_exec * 100,
                latest_sim * 100,
            )

    loss_drop = 0.0
    if len(losses) >= 2:
        loss_drop = (losses[0] - losses[-1]) / (abs(losses[0]) or 1.0)
    logging.info("Completed SFT: loss_drop=%.1f%%", loss_drop * 100)
    return latest_exec, latest_sim


async def run_rl_step(
    *,
    trainer,
    client: tinker.ServiceClient,
    tokenizer,
    examples: list[dict[str, Any]],
    config: Config,
    alias: str,
) -> dict[str, float | int]:
    sampler_path = trainer.save_weights_for_sampler(name=alias).result().path
    sampler = client.create_sampling_client(sampler_path)
    futures = [
        sampler.sample_async(
            prompt=types.ModelInput.from_ints(tokens=example["prompt_tokens"]),
            num_samples=config.rl_samples_per_prompt,
            sampling_params=types.SamplingParams(
                max_tokens=config.rl_max_tokens,
                temperature=config.rl_temperature,
            ),
        )
        for example in examples
    ]
    responses = await asyncio.gather(*futures)

    datums: list[types.Datum] = []
    rollout_rows: list[dict[str, Any]] = []

    for example, response in zip(examples, responses):
        group_rollouts: list[dict[str, Any]] = []
        for sequence in response.sequences:
            predicted_sql = clean_sql_for_execution(
                tokenizer.decode(sequence.tokens, skip_special_tokens=True)
            )
            reward = compute_sql_reward(
                example,
                predicted_sql,
                compile_reward=config.compile_reward,
                match_reward=config.match_reward,
                compile_error_penalty=config.compile_error_penalty,
            )
            group_rollouts.append(
                {
                    "question": example["question"],
                    "target": example["target"],
                    "predicted_sql": predicted_sql,
                    "prompt_tokens": example["prompt_tokens"],
                    "completion_tokens": list(sequence.tokens),
                    "completion_logprobs": [float(value) for value in (sequence.logprobs or [])],
                    "reward": reward["total"],
                    "compile": reward["compile"],
                    "execution_match": reward["execution_match"],
                    "similarity": reward["similarity"],
                    "sqlite_error": reward["sqlite_error"],
                }
            )

        advantages = group_relative_advantages([float(item["reward"]) for item in group_rollouts])
        for rollout, advantage in zip(group_rollouts, advantages):
            completion_tokens = rollout["completion_tokens"]
            completion_logprobs = rollout["completion_logprobs"]
            if not completion_tokens or len(completion_tokens) != len(completion_logprobs):
                continue
            datums.append(
                make_rl_datum(
                    rollout["prompt_tokens"],
                    completion_tokens,
                    completion_logprobs,
                    advantage,
                )
            )
            rollout["advantage"] = advantage
            rollout_rows.append(rollout)

    if not datums:
        return {
            "loss": 0.0,
            "reward": 0.0,
            "compile_rate": 0.0,
            "execution_match": 0.0,
            "similarity": 0.0,
            "num_rollouts": 0,
        }

    loss_fn_config = {"clip_range": config.rl_clip_range} if config.rl_loss_fn == "ppo" else None
    fwdbwd_future = await trainer.forward_backward_async(datums, config.rl_loss_fn, loss_fn_config=loss_fn_config)
    optim_future = await trainer.optim_step_async(
        types.AdamParams(
            learning_rate=config.rl_learning_rate,
            grad_clip_norm=config.grad_clip_norm,
        )
    )
    fwdbwd = await fwdbwd_future
    await optim_future

    best_rollout = max(rollout_rows, key=lambda row: (row["reward"], row["execution_match"], row["compile"]))
    logging.info(
        "\n--- [RL Rollout Sample] ---\nQuestion: %s\nPredicted: %s\nTarget:    %s\nReward:    %.2f\nCompile:   %s\nExecution: %s%s\n",
        best_rollout["question"],
        normalize_sql(best_rollout["predicted_sql"]),
        normalize_sql(best_rollout["target"]),
        best_rollout["reward"],
        "YES" if best_rollout["compile"] else "NO",
        "MATCH" if best_rollout["execution_match"] else "NO MATCH",
        f"\nSQLite:    {best_rollout['sqlite_error']}" if best_rollout["sqlite_error"] else "",
    )

    loss = float(fwdbwd.metrics.get("loss:mean", 0.0))
    return {
        "loss": loss,
        "reward": statistics.fmean(float(row["reward"]) for row in rollout_rows),
        "compile_rate": statistics.fmean(float(row["compile"]) for row in rollout_rows),
        "execution_match": statistics.fmean(float(row["execution_match"]) for row in rollout_rows),
        "similarity": statistics.fmean(float(row["similarity"]) for row in rollout_rows),
        "num_rollouts": len(rollout_rows),
    }


async def run_rl_phase(
    *,
    trainer,
    client: tinker.ServiceClient,
    tokenizer,
    rl_examples: list[dict[str, Any]],
    eval_examples: list[dict[str, Any]],
    config: Config,
    ml_logger,
    step_offset: int,
) -> tuple[float, float]:
    if config.rl_steps <= 0:
        return await snapshot_eval(trainer, client, tokenizer, "texttosql_rl_skip", eval_examples, config)

    batch_size = min(config.rl_prompts_per_step, len(rl_examples))
    rng = random.Random(config.seed + 1)
    order = list(range(len(rl_examples)))
    rng.shuffle(order)
    position = 0
    latest_exec = 0.0
    latest_sim = 0.0

    logging.info(
        "Starting GRPO-style RL: steps=%s prompts_per_step=%s samples_per_prompt=%s lr=%g loss_fn=%s",
        config.rl_steps,
        batch_size,
        config.rl_samples_per_prompt,
        config.rl_learning_rate,
        config.rl_loss_fn,
    )

    for local_step in range(1, config.rl_steps + 1):
        batch, position = next_batch(rl_examples, order, batch_size, position, rng)
        step_metrics = await run_rl_step(
            trainer=trainer,
            client=client,
            tokenizer=tokenizer,
            examples=batch,
            config=config,
            alias=f"texttosql_rl_rollout_s{local_step}",
        )
        global_step = step_offset + local_step
        ml_logger.log_metrics({"phase": "rl_train", **step_metrics}, step=global_step)
        logging.info(
            "[rl step %03d] reward=%.3f compile=%.1f%% exec=%.1f%% rollouts=%s",
            local_step,
            float(step_metrics["reward"]),
            float(step_metrics["compile_rate"]) * 100,
            float(step_metrics["execution_match"]) * 100,
            int(step_metrics["num_rollouts"]),
        )

        if local_step % config.rl_eval_every == 0 or local_step == config.rl_steps:
            alias = f"texttosql_rl_s{local_step}"
            latest_exec, latest_sim = await snapshot_eval(trainer, client, tokenizer, alias, eval_examples, config)
            ml_logger.log_metrics(
                {"phase": "rl_eval", "execution_match": latest_exec, "similarity": latest_sim},
                step=global_step,
            )
            logging.info(
                "[rl eval %03d] eval_exec=%.1f%% eval_similarity=%.1f%%",
                local_step,
                latest_exec * 100,
                latest_sim * 100,
            )

    return latest_exec, latest_sim


async def run_training(config: Config, preset: str) -> dict[str, float | str]:
    if config.phase not in {"full", "sft_only", "rl_only"}:
        raise ValueError("phase must be one of: full, sft_only, rl_only")
    if config.phase == "rl_only" and not config.resume_state_path:
        raise ValueError("phase=rl_only requires resume_state_path=<tinker checkpoint path>")
    if config.rl_loss_fn not in {"importance_sampling", "ppo"}:
        raise ValueError("rl_loss_fn must be either importance_sampling or ppo")

    log_dir = Path(config.log_dir.replace("{preset}", preset))
    ml_logger = ml_log.setup_logging(log_dir=str(log_dir), config=config, do_configure_logging_module=True)
    metrics_path = log_dir / "metrics.jsonl"
    client = tinker.ServiceClient(
        api_key=os.getenv("TINKER_API_KEY", "tml-dummy-key"),
        base_url=config.base_url,
    )
    server_model = await require_server(client, config.base_url)
    logging.info("Server ready at %s | model=%s", config.base_url, server_model or "unset")

    trainer = await create_or_resume_trainer(client, config)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    dataset = load_dataset(DATASET, split="train").shuffle(seed=config.seed)
    dataset = dataset.select(range(min(config.dataset_limit, len(dataset))))
    if len(dataset) < 10:
        raise RuntimeError("dataset_limit is too small to create train/eval splits")
    split = dataset.train_test_split(test_size=min(2_500, max(1, len(dataset) // 5)), shuffle=False)

    sft_examples = build_examples(tokenizer, config.prompt_format, split["train"], config.train_limit)
    rl_examples = build_examples(
        tokenizer,
        config.prompt_format,
        split["train"],
        config.rl_train_limit,
        require_seed_data=True,
        require_target_rows=True,
    )
    eval_examples = build_examples(
        tokenizer,
        config.prompt_format,
        split["test"],
        config.eval_limit,
        require_seed_data=True,
        require_target_rows=True,
    )
    if config.phase in {"full", "sft_only"} and not sft_examples:
        raise RuntimeError("No SFT examples fit within the max sequence length.")
    if config.phase in {"full", "rl_only"} and not rl_examples:
        raise RuntimeError("No RL examples with executable target rows were found.")
    if not eval_examples:
        raise RuntimeError("No evaluation examples with executable seed data were found.")

    logging.info(
        "Data: %s SFT train, %s RL train, %s eval",
        len(sft_examples),
        len(rl_examples),
        len(eval_examples),
    )

    before_exec, before_sim = await snapshot_eval(trainer, client, tokenizer, "texttosql_before", eval_examples, config)
    ml_logger.log_metrics(
        {"phase": "eval_baseline", "execution_match": before_exec, "similarity": before_sim},
        step=0,
    )

    after_sft_exec = before_exec
    after_sft_sim = before_sim
    after_rl_exec = before_exec
    after_rl_sim = before_sim
    sft_state_path = ""
    final_state_path = ""
    step_offset = 0

    if config.phase in {"full", "sft_only"}:
        after_sft_exec, after_sft_sim = await run_sft_phase(
            trainer=trainer,
            client=client,
            tokenizer=tokenizer,
            train_examples=sft_examples,
            eval_examples=eval_examples,
            config=config,
            ml_logger=ml_logger,
            step_offset=0,
        )
        step_offset = config.sft_steps
        sft_state_path = await maybe_save_state(trainer, config.save_sft_state_name, "Post-SFT")

    if config.phase in {"full", "rl_only"}:
        after_rl_exec, after_rl_sim = await run_rl_phase(
            trainer=trainer,
            client=client,
            tokenizer=tokenizer,
            rl_examples=rl_examples,
            eval_examples=eval_examples,
            config=config,
            ml_logger=ml_logger,
            step_offset=step_offset,
        )
    else:
        after_rl_exec = after_sft_exec
        after_rl_sim = after_sft_sim

    final_state_path = await maybe_save_state(trainer, config.save_final_state_name, "Final")

    logging.info("Saved metrics to %s", metrics_path)
    logging.info(
        "[summary] execution=%.1f%%->%.1f%%->%.1f%% similarity=%.1f%%->%.1f%%->%.1f%%",
        before_exec * 100,
        after_sft_exec * 100,
        after_rl_exec * 100,
        before_sim * 100,
        after_sft_sim * 100,
        after_rl_sim * 100,
    )
    ml_logger.close()

    result: dict[str, float | str] = {
        "before_execution_match": before_exec,
        "after_sft_execution_match": after_sft_exec,
        "after_rl_execution_match": after_rl_exec,
        "before_similarity": before_sim,
        "after_sft_similarity": after_sft_sim,
        "after_rl_similarity": after_rl_sim,
        "metrics_path": str(metrics_path),
    }
    if sft_state_path:
        result["sft_state_path"] = sft_state_path
    if final_state_path:
        result["final_state_path"] = final_state_path
    return result


@chz.blueprint._entrypoint.exit_on_entrypoint_error
def cli() -> None:
    import sys

    logging.getLogger("tinker").setLevel(logging.WARNING)

    preset = sys.argv[1]
    blueprint = PRESETS[preset].clone()
    config = blueprint.make_from_argv(sys.argv[2:], allow_hyphens=True)
    asyncio.run(run_training(config, preset))


if __name__ == "__main__":
    cli()
