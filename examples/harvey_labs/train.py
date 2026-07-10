"""Train Gemma 4 on Harvey LAB with live tool-use rollouts."""

from __future__ import annotations

import asyncio
from pathlib import Path

import chz
from env import LabDatasetBuilder
from renderer import register_gemma4_tool_renderer
from tinker_cookbook.rl import train as rl_train
from tinker_utils import LimitedDatasetBuilder, force_rich_log_colors, resolve_base_url

MODEL_NAME = "google/gemma-4-E4B-it"
RENDERER_NAME = "gemma4"
LORA_RANK = 32
LEARNING_RATE = 3e-6
TEMPERATURE = 1.0
LOSS_FN = "importance_sampling"
COMMAND_TIMEOUT = 60
JUDGE_PARALLEL = 1
NUM_GROUPS_TO_LOG = 1
# Forty distinct, small LAB tasks selected by total source-document size, with
# duplicate document sets removed and no more than four tasks from one domain.
BOOTSTRAP_TASKS = (
  "employment-labor/draft-markup-of-settlement-agreement",
  "intellectual-property/extract-ip-tech-transactions",
  "employment-labor/offer-letter-to-employment-agreement",
  "structured-finance-securitization/extract-key-terms-from-warehouse-credit-facility-term-sheet",
  "trusts-estates-private-client/extract-client-intake-facts/scenario-01",
  "environmental-esg/extract-indemnification-terms-from-environmental-settlement-agreement",
  "corporate-ma/draft-markup-of-engagement-letter",
  "trusts-estates-private-client/compare-final-decree-of-divorce-against-mediated-settlement-agreement",
  "intellectual-property/compare-ip-tech-transactions",
  "corporate-ma/draft-issues-list-for-escrow-agreement",
  "corporate-ma/review-outside-counsel-engagement-letter",
  "arbitration-international-dispute-resolution/draft-markup-of-arbitration-agreement",
  "corporate-governance/review-nda-playbook-review",
  "trusts-estates-private-client/identify-issues-in-counterparty-postnuptial-agreement",
  "intellectual-property/extract-key-terms-from-technology-licensing-term-sheet",
  "funds-asset-management/extract-reporting-obligations-from-advisory-agreement",
  "corporate-ma/extract-key-terms-from-fund-term-sheet",
  "emerging-companies-venture-capital/draft-certificate-of-incorporation",
  "immigration/compare-i",
  "intellectual-property/review-inbound-nda-against-company-playbook",
  "banking-finance/draft-intercreditor-agreement",
  "funds-asset-management/draft-lpa/scenario-01",
  "funds-asset-management/draft-lpa/scenario-12",
  "corporate-governance/compare-bylaws-against-best-practices",
  "litigation-dispute-resolution/extract-key-terms-from-counterparty-complaint",
  "trusts-estates-private-client/identify-issues-in-counterpartys-draft-prenuptial-agreement",
  "employment-labor/draft-settlement-agreement",
  "emerging-companies-venture-capital/extract-key-terms-from-investors-rights-agreement",
  "funds-asset-management/draft-lpa/scenario-06",
  "capital-markets/extract-key-terms-from-underwriting-agreement",
  "real-estate/extract-psa-key-terms/scenario-01",
  "real-estate/extract-psa-key-terms/scenario-02",
  "corporate-governance/draft-action-by-incorporator",
  "immigration/compare-draft-eb",
  "structured-finance-securitization/compare-collateral-tape-against-eligibility-criteria",
  "litigation-dispute-resolution/identify-issues-in-matter-budget-proposal",
  "capital-markets/review-form-10",
  "banking-finance/identify-issues-in-compliance-certificate",
  "emerging-companies-venture-capital/compare-term-sheet-against-stock-purchase-agreement",
  "banking-finance/extract-credit-agreement-covenants",
)


@chz.chz
class RunConfig:
  """Small set of knobs for the LAB RL experiment."""

  base_url: str | None = None
  model_name: str = MODEL_NAME
  renderer_name: str = RENDERER_NAME
  lab_root: Path = Path("experiments/lab-traces/harvey-labs")
  split_path: Path | None = None
  task: str | None = None
  train_limit: int | None = 40
  eval_limit: int | None = 0
  batch_size: int = 1
  rollouts_per_example: int = 4
  max_steps: int = 40
  max_turns: int = 40
  max_tokens: int = 3072
  max_trajectory_tokens: int = 32 * 1024
  judge_model: str = "gemini-3.5-flash"
  max_reward_criteria: int | None = 3
  log_path: str = "artifacts/harvey-labs"


def build_dataset_builder(config: RunConfig) -> LabDatasetBuilder:
  return LabDatasetBuilder(
    lab_root=config.lab_root,
    split_path=config.split_path,
    task_names=[config.task] if config.task else list(BOOTSTRAP_TASKS),
    train_limit=config.train_limit,
    eval_limit=config.eval_limit,
    batch_size=config.batch_size,
    group_size=config.rollouts_per_example,
    model_name=config.model_name,
    renderer_name=config.renderer_name,
    max_turns=config.max_turns,
    command_timeout=COMMAND_TIMEOUT,
    judge_model=config.judge_model,
    judge_parallel=JUDGE_PARALLEL,
    max_reward_criteria=config.max_reward_criteria,
    max_trajectory_tokens=config.max_trajectory_tokens,
    max_generation_tokens=config.max_tokens,
  )


async def run(config: RunConfig) -> None:
  register_gemma4_tool_renderer(RENDERER_NAME)
  builder = LimitedDatasetBuilder(
    build_dataset_builder(config),
    max_batches=config.max_steps,
    max_eval_batches=config.eval_limit,
  )
  train_config = rl_train.Config(
    learning_rate=LEARNING_RATE,
    dataset_builder=builder,
    model_name=config.model_name,
    recipe_name="harvey_labs",
    renderer_name=config.renderer_name,
    lora_rank=LORA_RANK,
    max_tokens=config.max_tokens,
    temperature=TEMPERATURE,
    log_path=config.log_path,
    base_url=resolve_base_url(config.base_url),
    eval_every=0,
    save_every=0,
    max_steps=config.max_steps,
    loss_fn=LOSS_FN,
    num_substeps=1,
    kl_penalty_coef=0.0,
    kl_discount_factor=0.0,
    remove_constant_reward_groups=False,
    num_groups_to_log=NUM_GROUPS_TO_LOG,
  )
  await rl_train.main(train_config)


def main() -> None:
  force_rich_log_colors()
  config = chz.entrypoint(RunConfig, allow_hyphens=True)
  asyncio.run(run(config))


if __name__ == "__main__":
  main()
