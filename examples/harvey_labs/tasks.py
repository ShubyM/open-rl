"""LAB task selection and loading."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# One hundred distinct LAB tasks curated by source-document size, with
# duplicate document sets removed, spread across practice areas, and kept
# disjoint from EVAL_TASKS.
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
  "antitrust-competition/analyze-antitrust-hsr-strategy",
  "arbitration-international-dispute-resolution/analyze-arbitration-agreement-markup-analysis",
  "banking-finance/analyze-counterparty-markup-of-senior-secured-credit-facility-term-sheet",
  "bankruptcy-restructuring/analyze-counterparty-markup-of-plan-of-reorganization",
  "capital-markets/analyze-counterparty-markup-of-underwriting-agreement",
  "contracts/banking/account-control-agreement-counterparty-paper-review/scenario-01",
  "corporate-governance/analyze-compliance-program-gaps",
  "corporate-ma/analyze-change-of-control-provisions-across-targets-material-contracts",
  "data-privacy-cybersecurity/analyze-counterparty-markup-of-cross",
  "emerging-companies-venture-capital/analyze-counterparty-markup-of-bridge-loan-agreement",
  "employment-labor/analyze-counterparty-markup-of-executive-employment-agreement",
  "energy-natural-resources/analyze-counterparty-markup-of-concession-agreement",
  "environmental-esg/analyze-counterparty-markup-of-administrative-settlement-agreement",
  "funds-asset-management/analyze-counterparty-markup-of-investment-advisory-agreement",
  "healthcare-life-sciences/analyze-compliance-program-gaps",
  "immigration/compare-employer-corrective-action-plans-against-ice-regulatory-standards",
  "insurance/analyze-counterparty-markup-of-reinsurance-treaty",
  "intellectual-property/analyze-counterparty-markup-of-contract-amendment",
  "international-trade-sanctions/analyze-counterparty-markup-of-mitigation-agreement",
  "litigation-dispute-resolution/analyze-counterparty-motion-to-dismiss",
  "real-estate/analyze-counterparty-markup-of-commercial-lease-agreement",
  "structured-finance-securitization/analyze-counterparty-markup-of-indenture",
  "tax/analyze-counterparty-markup-of-proposed-stipulation-of-facts",
  "trusts-estates-private-client/analyze-counterparty-markup-of-parenting-plan",
  "white-collar-defense-investigations/analyze-counterparty-markup-of-deferred-prosecution-agreement",
  "antitrust-competition/analyze-counterparty-markup-of-protective-order",
  "arbitration-international-dispute-resolution/analyze-arbitration-award-for-new-york-convention-enforcement-defenses",
  "banking-finance/analyze-credit-agreement-markup",
  "bankruptcy-restructuring/analyze-counterparty-markup-of-restructuring-support-agreement",
  "capital-markets/compare-charter-against-offering",
  "contracts/banking/account-control-agreement-counterparty-paper-review/scenario-02",
  "corporate-governance/analyze-eu-ai-act-high",
  "corporate-ma/analyze-cim-deal-teaser/scenario-01",
  "data-privacy-cybersecurity/analyze-counterparty-markup-of-data-processing-agreement",
  "emerging-companies-venture-capital/analyze-counterparty-markup-of-investors-rights-agreement",
  "employment-labor/analyze-iss-employment-complaint",
  "energy-natural-resources/analyze-counterparty-markup-of-credit-agreement",
  "environmental-esg/analyze-counterparty-markup-of-environmental-indemnity-agreement",
  "funds-asset-management/analyze-counterparty-markup-of-limited-partnership-agreement",
  "healthcare-life-sciences/analyze-counterparty-markup-of-clinical-trial-agreement",
  "antitrust-competition/analyze-iss-antitrust-transaction-structure",
  "arbitration-international-dispute-resolution/analyze-counterparty-markup-of-arbitration-agreement",
  "banking-finance/compare-borrower-disclosures-against-due-diligence-findings",
  "bankruptcy-restructuring/analyze-counterparty-plan-objection-for-meritorious-and-deficient-arguments",
  "capital-markets/compare-closing-documents-against-closing-checklist",
  "contracts/banking/account-control-agreement-first-draft/scenario-01",
  "corporate-governance/analyze-flsa-overtime-rule-gap-against-current-employee-classifications",
  "corporate-ma/analyze-cim-deal-teaser/scenario-02",
  "data-privacy-cybersecurity/analyze-cpra-compliance-gaps-against-current-privacy-program",
  "emerging-companies-venture-capital/analyze-counterparty-markup-of-stock-purchase-agreement",
  "employment-labor/analyze-reasonable-accommodation-request-under-ada-requirements",
  "energy-natural-resources/analyze-counterparty-markup-of-engineering-procurement-construction-contract",
  "environmental-esg/analyze-counterparty-markup-of-settlement-agreement",
  "funds-asset-management/analyze-counterparty-markup-of-limited-partnership-interest-transfer-agreement",
  "healthcare-life-sciences/analyze-counterparty-markup-of-merger-agreement",
  "immigration/compare-job-requirements-against-candidate-credentials",
  "insurance/analyze-property-damage-claim-against-commercial-policy-exclusions",
  "intellectual-property/analyze-counterparty-markup-of-ip-assignment-agreement",
  "international-trade-sanctions/analyze-cross",
  "litigation-dispute-resolution/analyze-counterparty-requests-for-production-for-objectionable-and-overbroad-discovery-demands",
)

# Held-out progress evals: twenty tasks disjoint from BOOTSTRAP_TASKS, chosen
# by estimated extracted-document tokens (~5K-17K each, at most three per
# practice area, duplicate document sets excluded) so a full read of every
# source always fits even a conservative 32K trajectory budget.
EVAL_TASKS = (
  "immigration/compare-uscis-filing-receipt-against-original-petition-submission",
  "trusts-estates-private-client/extract-client-intake-facts/scenario-02",
  "trusts-estates-private-client/extract-creditor-claims-from-estate-correspondence",
  "corporate-ma/review-ma-transaction-invoice-against-fee-arrangement",
  "corporate-ma/compare-post",
  "corporate-ma/compare-ddrl-to-vdr-index/scenario-01",
  "corporate-governance/draft-batch-nda-generation",
  "immigration/identify-h1b-qualification-issues",
  "banking-finance/compare-borrower-covenant-compliance-analysis",
  "capital-markets/extract-key-terms-from-section-16-filings",
  "international-trade-sanctions/compare-transaction-records-against-sanctioned-parties-list",
  "international-trade-sanctions/identify-issues-in-customs-entry-filing",
  "banking-finance/extract-lien-and-debt-information-from-ucc-filings",
  "immigration/compare-immigration",
  "trusts-estates-private-client/extract-key-financial-facts-from-client-intake-questionnaire-and-supporting-documents",
  "intellectual-property/identify-prior-art-deficiencies-in-patent-infringement-defense",
  "international-trade-sanctions/compare-entity-details-against-ofac-sanctions-list",
  "white-collar-defense-investigations/extract-relevant-transactions-from-accounting-records",
  "litigation-dispute-resolution/review-litigation-invoice-against-outside-counsel-billing-guidelines",
  "antitrust-competition/draft-competitor-confidentiality-agreement-review",
)


@dataclass(frozen=True)
class LabTask:
  name: str
  instructions: str
  documents_dir: Path
  criteria_count: int


def task_slug(task_name: str) -> str:
  return task_name.replace("/", "__")


def load_task(lab_root: Path, task_name: str) -> LabTask:
  task_parts = task_name.split("/")
  if not task_name or any(part in {"", ".", ".."} for part in task_parts):
    raise ValueError(f"Invalid LAB task name: {task_name!r}")

  tasks_root = (lab_root / "tasks").resolve()
  task_dir = (tasks_root / Path(*task_parts)).resolve()
  if not task_dir.is_relative_to(tasks_root):
    raise ValueError(f"LAB task directory escapes tasks root: {task_dir}")

  config_path = task_dir / "task.json"
  if not config_path.is_file():
    raise FileNotFoundError(f"LAB task config not found: {config_path}")
  config = json.loads(config_path.read_text(encoding="utf-8"))
  instructions = config.get("instructions")
  if not instructions:
    instructions = (task_dir / "instructions.md").read_text(encoding="utf-8")
  documents_dir = (task_dir / "documents").resolve()
  if not documents_dir.is_relative_to(task_dir):
    raise ValueError(f"LAB documents directory escapes task directory: {documents_dir}")
  if not documents_dir.is_dir():
    raise FileNotFoundError(f"LAB documents directory not found: {documents_dir}")
  if not any(path.is_file() for path in documents_dir.rglob("*")):
    raise ValueError(f"LAB documents directory is empty: {documents_dir}")
  return LabTask(
    name=task_name,
    instructions=instructions,
    documents_dir=documents_dir,
    criteria_count=len(config.get("criteria") or []),
  )


def load_lab_tasks(lab_root: Path, task_names: list[str], limit: int | None = None) -> list[LabTask]:
  lab_root = lab_root.resolve()
  return [load_task(lab_root, task_name) for task_name in task_names[:limit]]
