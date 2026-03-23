import asyncio
import json
import logging
import os
import random
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, cast

import chz
import tinker
from datasets import load_dataset
from tinker import types

BASE_MODEL = "google/gemma-3-1b-pt"
BASE_URL = "http://127.0.0.1:9003"
DATASET = "philschmid/gretel-synthetic-text-to-sql"
METRICS_PATH = Path(__file__).resolve().parent / "artifacts" / "texttosql_{preset}_metrics.jsonl"
MAX_SEQ_LENGTH = 512

USER_PROMPT = """Given the <USER_QUERY> and the <SCHEMA>, generate the corresponding SQL command to retrieve the desired data, considering the query's syntax, semantics, and schema constraints.

<SCHEMA>
{context}
</SCHEMA>

<USER_QUERY>
{question}
</USER_QUERY>
"""

logger = logging.getLogger(__name__)

os.environ.setdefault("TINKER_API_KEY", "tml-dummy-key")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


@chz.chz
class Config:
    steps: int
    batch_size: int
    rank: int
    learning_rate: float
    base_url: str = os.getenv("TINKER_BASE_URL") or os.getenv("OPEN_RL_BASE_URL") or BASE_URL
    grad_clip_norm: float = 0.3
    eval_every: int = 50
    train_limit: int = 2_048
    eval_limit: int = 128
    seed: int = 30
    metrics_path: str = str(METRICS_PATH)
    eval_max_tokens: int = 256


PRESETS = {
    "gemma": chz.Blueprint(Config).apply(
        {"steps": 400, "batch_size": 32, "rank": 16, "learning_rate": 2e-4, "train_limit": 10_000, "eval_limit": 100, "eval_every": 50},
        layer_name="gemma preset",
    ),
    "notebook": chz.Blueprint(Config).apply(
        {"steps": 375, "batch_size": 8, "rank": 16, "learning_rate": 2e-4, "train_limit": 10_000, "eval_limit": 2_500, "eval_every": 125},
        layer_name="notebook preset",
    ),
}


async def run_training(config: Config, preset: str) -> dict[str, float | str]:
    client = tinker.ServiceClient(api_key=os.getenv("TINKER_API_KEY", "tml-dummy-key"), base_url=config.base_url)
    server_model = await require_server(client, config.base_url)
    logger.info("Server ready at %s | model=%s", config.base_url, server_model or "unset")

    trainer = await client.create_lora_training_client_async(
        base_model=BASE_MODEL,
        rank=config.rank,
        train_mlp=True,
        train_attn=True,
        train_unembed=False,
    )
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")

    dataset = load_dataset(DATASET, split="train").shuffle(seed=config.seed)
    dataset = dataset.select(range(min(12_500, len(dataset))))
    split = dataset.train_test_split(test_size=2_500, shuffle=False)

    train_examples = [ex for row in list(split["train"])[:config.train_limit] if (ex := build_example(tokenizer, row)) is not None]
    eval_examples = [ex for row in list(split["test"])[:config.eval_limit] if (ex := build_example(tokenizer, row)) is not None]
    if not train_examples:
        raise RuntimeError("No training examples fit within max_seq_length.")

    batch_size = min(config.batch_size, len(train_examples))
    metrics_path = Path(config.metrics_path.replace("{preset}", preset))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text("", encoding="utf-8")

    def append_metric(row: dict[str, Any]) -> None:
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

    logger.info(
        "Data: %s train, %s eval | batch=%s rank=%s lr=%g",
        len(train_examples),
        len(eval_examples),
        batch_size,
        config.rank,
        config.learning_rate,
    )

    before_exact, before_sim = evaluate(client, trainer, tokenizer, "texttosql_before", eval_examples, max_tokens=config.eval_max_tokens)
    append_metric({"step": 0, "phase": "eval", "exact_match": before_exact, "similarity": before_sim})
    logger.info("[before] exact=%.1f%% similarity=%.1f%%", before_exact * 100, before_sim * 100)

    losses: list[float] = []
    eval_exact = [before_exact]
    eval_sim = [before_sim]
    rng = random.Random(config.seed)
    order = list(range(len(train_examples)))
    rng.shuffle(order)
    pos = 0

    for step in range(1, config.steps + 1):
        if pos + batch_size > len(order):
            rng.shuffle(order)
            pos = 0
        batch = [train_examples[order[i]] for i in range(pos, pos + batch_size)]
        pos += batch_size

        datums = [row["datum"] for row in batch]
        active_tokens = sum(row["active_tokens"] for row in batch)

        # Match Tinker-style dispatch: submit both requests first, then await each future.
        fwdbwd_future = await trainer.forward_backward_async(datums, "cross_entropy")
        optim_future = await trainer.optim_step_async(
            types.AdamParams(learning_rate=config.learning_rate, grad_clip_norm=config.grad_clip_norm)
        )

        fwdbwd = await fwdbwd_future
        await optim_future

        loss = float(fwdbwd.metrics.get("loss:sum", 0.0)) / max(1, active_tokens)
        losses.append(loss)
        append_metric({"step": step, "phase": "train", "loss": loss})
        logger.info("[train] step=%04d/%04d loss=%.4f", step, config.steps, loss)

        if step % config.eval_every == 0 or step == config.steps:
            exact, sim = evaluate(client, trainer, tokenizer, f"texttosql_s{step}", eval_examples, max_tokens=config.eval_max_tokens)
            eval_exact.append(exact)
            eval_sim.append(sim)
            append_metric({"step": step, "phase": "eval", "exact_match": exact, "similarity": sim})
            logger.info("[eval]  step=%04d exact=%.1f%% similarity=%.1f%%", step, exact * 100, sim * 100)

    loss_drop = (losses[0] - losses[-1]) / (abs(losses[0]) or 1.0)
    logger.info("Saved metrics to %s", metrics_path)
    logger.info(
        "[summary] exact=%.1f%%->%.1f%% similarity=%.1f%%->%.1f%% loss_drop=%.1f%%",
        before_exact * 100,
        eval_exact[-1] * 100,
        before_sim * 100,
        eval_sim[-1] * 100,
        loss_drop * 100,
    )

    return {
        "before_exact": before_exact,
        "after_exact": eval_exact[-1],
        "before_similarity": before_sim,
        "after_similarity": eval_sim[-1],
        "loss_drop": loss_drop,
        "metrics_path": str(metrics_path),
    }


async def require_server(service_client: tinker.ServiceClient, base_url: str) -> str | None:
    try:
        capabilities = await service_client.get_server_capabilities_async()
    except Exception as exc:
        raise RuntimeError(f"Open-RL server at {base_url} is not reachable. Start it with `make run-text-to-sql-server`.") from exc

    model_names = [model.model_name for model in capabilities.supported_models if getattr(model, "model_name", None)]
    return model_names[0] if model_names else None

def normalize_sql(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL)
    text = text.replace("<|im_start|>", " ").replace("<|im_end|>", " ")
    text = text.strip()
    text = re.sub(r"^assistant\s*[:\-]?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^sql\s*[:\-]?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```(?:sql)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = " ".join(text.split()).lower()
    text = re.sub(r"\s+([,;()])", r"\1", text)
    text = re.sub(r"([,(])\s+", r"\1", text)
    return text


def make_datum(full_tokens: list[int], weights: list[int]) -> types.Datum:
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=full_tokens[:-1]),
        loss_fn_inputs=cast(Any, {"weights": weights[1:], "target_tokens": full_tokens[1:]}),
    )


def build_example(tokenizer: Any, sample: dict[str, Any]) -> dict[str, Any] | None:
    messages = [
        {"role": "user", "content": USER_PROMPT.format(question=sample["sql_prompt"], context=sample["sql_context"])},
        {"role": "assistant", "content": sample["sql"]},
    ]
    prompt_text = tokenizer.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    full_tokens = tokenizer.encode(full_text, add_special_tokens=False)
    if len(full_tokens) <= len(prompt_tokens) or len(full_tokens) > MAX_SEQ_LENGTH:
        return None
    weights = [0] * len(prompt_tokens) + [1] * (len(full_tokens) - len(prompt_tokens))
    return {
        "question": sample["sql_prompt"],
        "target": sample["sql"],
        "prompt_tokens": prompt_tokens,
        "active_tokens": len(full_tokens) - len(prompt_tokens),
        "datum": make_datum(full_tokens, weights),
    }


def evaluate(client: tinker.ServiceClient, trainer: tinker.TrainingClient, tokenizer: Any, alias: str, examples: list[dict[str, Any]], max_tokens: int = 256) -> tuple[float, float]:
    sampler = client.create_sampling_client(trainer.save_weights_for_sampler(name=alias).result().path)
    # Fire all requests concurrently without blocking
    futures = []
    for example in examples:
        future = sampler.sample(
            prompt=types.ModelInput.from_ints(tokens=example["prompt_tokens"]),
            num_samples=1,
            sampling_params=types.SamplingParams(max_tokens=max_tokens, temperature=0.0),
        )
        futures.append((example, future))

    exact, similarity = 0.0, 0.0
    for idx, (example, future) in enumerate(futures):
        result = future.result()
        predicted = normalize_sql(tokenizer.decode(result.sequences[0].tokens if result.sequences else [], skip_special_tokens=True))
        target = normalize_sql(example["target"])

        # Print visual check for all evaluation items
        logger.info("\n--- [Visual Check %s Item %d] ---", alias, idx + 1)
        logger.info("Question: %s", example["question"])
        logger.info("Predicted: %s", predicted)
        logger.info("Target:    %s", target)

        if predicted == target:
            match_str = "\033[92mMATCH\033[0m" # Green
        else:
            match_str = "\033[91mNO MATCH\033[0m" # Red
        logger.info("Match:     %s\n", match_str)

        exact += float(predicted == target)
        similarity += SequenceMatcher(None, predicted, target).ratio()
    count = max(1, len(examples))
    return exact / count, similarity / count


@chz.blueprint._entrypoint.exit_on_entrypoint_error
def cli() -> None:
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("tinker").setLevel(logging.WARNING)

    preset = sys.argv[1]
    blueprint = PRESETS[preset].clone()
    config = blueprint.make_from_argv(sys.argv[2:], allow_hyphens=True)
    asyncio.run(run_training(config, preset))


if __name__ == "__main__":
    cli()
