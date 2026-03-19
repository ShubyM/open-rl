import os
import random
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, cast

import chz
import matplotlib.pyplot as plt
import requests
import tinker
from datasets import load_dataset
from tinker import types
from transformers import AutoTokenizer

BASE_MODEL = "google/gemma-3-1b-it"
TOKENIZER_MODEL = "google/gemma-3-1b-it"
BASE_URL = "http://127.0.0.1:9003"
DATASET = "philschmid/gretel-synthetic-text-to-sql"
PLOT_PATH = Path(__file__).resolve().parent / "artifacts" / "texttosql_{preset}_metrics.png"
MAX_SEQ_LENGTH = 512

USER_PROMPT = """Given the <USER_QUERY> and the <SCHEMA>, generate the corresponding SQL command to retrieve the desired data, considering the query's syntax, semantics, and schema constraints.

<SCHEMA>
{context}
</SCHEMA>

<USER_QUERY>
{question}
</USER_QUERY>
"""

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
    eval_every: int = 16
    train_limit: int = 2_048
    eval_limit: int = 128
    seed: int = 17


PRESETS = {
    "gemma": chz.Blueprint(Config).apply(
        {"steps": 400, "batch_size": 8, "rank": 16, "learning_rate": 2e-4, "train_limit": 10_000, "eval_limit": 20, "eval_every": 400},
        layer_name="gemma preset",
    ),
    "notebook": chz.Blueprint(Config).apply(
        {"steps": 375, "batch_size": 8, "rank": 16, "learning_rate": 2e-4, "train_limit": 10_000, "eval_limit": 2_500, "eval_every": 125},
        layer_name="notebook preset",
    ),
}


# ── Helpers ──────────────────────────────────────────────────────────


def normalize_sql(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL)
    text = text.replace("<|im_start|>", " ").replace("<|im_end|>", " ")
    text = text.strip()
    text = re.sub(r"^```(?:sql)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = re.sub(r"^assistant\s*[:\-]?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^sql\s*[:\-]?\s*", "", text, flags=re.IGNORECASE)
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



def evaluate(client: Any, trainer: Any, tokenizer: Any, alias: str, examples: list[dict[str, Any]], max_tokens: int = 256) -> tuple[float, float]:
    sampler = client.create_sampling_client(trainer.save_weights_for_sampler(name=alias).result().path)
    exact, similarity = 0.0, 0.0
    for ex in examples:
        result = sampler.sample(
            prompt=types.ModelInput.from_ints(tokens=ex["prompt_tokens"]),
            num_samples=1,
            sampling_params=types.SamplingParams(max_tokens=max_tokens, temperature=0.0),
        ).result()
        predicted = normalize_sql(tokenizer.decode(result.sequences[0].tokens if result.sequences else [], skip_special_tokens=True))
        target = normalize_sql(ex["target"])
        exact += float(predicted == target)
        similarity += SequenceMatcher(None, predicted, target).ratio()
    n = max(1, len(examples))
    return exact / n, similarity / n


def plot_metrics(losses: list[float], eval_steps: list[int], eval_exact: list[float], eval_similarity: list[float], plot_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(range(1, len(losses) + 1), losses, marker="o", markersize=3, color="#1f77b4")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("loss/token")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(eval_steps, [x * 100 for x in eval_exact], marker="o", label="Exact Match %", color="#1b9e77")
    axes[1].plot(eval_steps, [x * 100 for x in eval_similarity], marker="o", label="String Similarity %", color="#d95f02")
    axes[1].set_xticks(eval_steps)
    axes[1].set_title("Eval Quality")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("%")
    axes[1].set_ylim(0, 105)
    axes[1].legend(loc="lower right")
    axes[1].grid(True, alpha=0.3)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────


def run_training(config: Config, preset: str) -> dict[str, float]:
    # 1. Create training client
    try:
        resp = requests.get(f"{config.base_url.rstrip('/')}/api/v1/get_server_capabilities", timeout=5.0)
        resp.raise_for_status()
        caps = resp.json()
    except Exception as exc:
        raise RuntimeError(f"Open-RL server at {config.base_url} is not reachable. Start it with `make run-text-to-sql-server`.") from exc

    print(f"Server ready at {config.base_url} | model={caps.get('default_model') or 'unset'}")

    client = tinker.ServiceClient(api_key=os.getenv("TINKER_API_KEY", "tml-dummy-key"), base_url=config.base_url)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
    trainer = client.create_lora_training_client(base_model=BASE_MODEL, rank=config.rank, train_mlp=True, train_attn=True, train_unembed=False)

    # 2. Load & prepare data
    dataset = load_dataset(DATASET, split="train").shuffle(seed=config.seed)
    dataset = dataset.select(range(min(12_500, len(dataset))))
    split = dataset.train_test_split(test_size=2_500, shuffle=False)

    train_examples = [ex for row in list(split["train"])[:config.train_limit] if (ex := build_example(tokenizer, row)) is not None]
    eval_examples = [ex for row in list(split["test"])[:config.eval_limit] if (ex := build_example(tokenizer, row)) is not None]
    if not train_examples:
        raise RuntimeError("No training examples fit within max_seq_length.")

    batch_size = min(config.batch_size, len(train_examples))
    print(f"Data: {len(train_examples)} train, {len(eval_examples)} eval | batch={batch_size} rank={config.rank} lr={config.learning_rate:g}")

    # 3. Baseline eval
    before_exact, before_sim = evaluate(client, trainer, tokenizer, "texttosql_before", eval_examples)
    print(f"[before] exact={before_exact:.1%} similarity={before_sim:.1%}")

    # 4. Training loop
    losses: list[float] = []
    eval_steps, eval_exact, eval_sim = [0], [before_exact], [before_sim]
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

        datums = [r["datum"] for r in batch]
        active = sum(r["active_tokens"] for r in batch)

        # Forward-backward + optimizer step
        fwdbwd = trainer.forward_backward(datums, "cross_entropy").result()
        trainer.optim_step(types.AdamParams(learning_rate=config.learning_rate, grad_clip_norm=config.grad_clip_norm)).result()

        loss = float(fwdbwd.metrics.get("loss:sum", 0.0)) / max(1, active)
        losses.append(loss)
        print(f"[train] step={step:04d}/{config.steps} loss={loss:.4f}")

        if step % config.eval_every == 0 or step == config.steps:
            exact, sim = evaluate(client, trainer, tokenizer, f"texttosql_s{step}", eval_examples)
            eval_steps.append(step)
            eval_exact.append(exact)
            eval_sim.append(sim)
            print(f"[eval]  step={step:04d} exact={exact:.1%} similarity={sim:.1%}")

    # 5. Summary & plot
    plot_path = Path(str(PLOT_PATH).replace("{preset}", preset))
    plot_metrics(losses, eval_steps, eval_exact, eval_sim, plot_path)
    loss_drop = (losses[0] - losses[-1]) / (abs(losses[0]) or 1.0)
    print(f"Saved plot to {plot_path}")
    print(f"[summary] exact={before_exact:.1%}->{eval_exact[-1]:.1%} similarity={before_sim:.1%}->{eval_sim[-1]:.1%} loss_drop={loss_drop:.1%}")

    return {"before_exact": before_exact, "after_exact": eval_exact[-1], "before_similarity": before_sim, "after_similarity": eval_sim[-1], "loss_drop": loss_drop}


@chz.blueprint._entrypoint.exit_on_entrypoint_error
def cli() -> None:
    import sys

    argv = sys.argv[1:]
    preset = "gemma"
    if argv and argv[0] in PRESETS:
        preset = argv[0]
        argv = argv[1:]

    blueprint = PRESETS[preset].clone()
    config = blueprint.make_from_argv(argv, allow_hyphens=True)
    run_training(config, preset)


if __name__ == "__main__":
    cli()
