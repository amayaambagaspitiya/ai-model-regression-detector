# AI Model Regression Detector

A small evaluation framework for detecting regressions in AI-powered email
classification. It runs a fixed golden dataset against prompts and models hosted
on Groq, records quality and reliability metrics, and compares evaluation runs.

Emails are classified into one of five labels:

- `billing`
- `support`
- `sales`
- `spam`
- `other`

## Requirements

- Python 3.12.12
- A [Groq API key](https://console.groq.com/keys)

## Setup

Create and activate a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Install the pinned dependencies:

```bash
python -m pip install -r requirements.txt
```

Create a `.env` file in the repository root:

```text
GROQ_API_KEY=your_groq_api_key
```

The `.env` file is ignored by Git and must never be committed.

## Run an evaluation

Evaluate a model with a prompt and save the result:

```bash
python evals/run_evaluation.py \
  --dataset data/golden_emails_v1.jsonl \
  --prompt prompts/email_classifier_v2.txt \
  --model openai/gpt-oss-20b \
  --output results/gpt_oss_20b_v1_results.json
```

Each run evaluates every example in the selected versioned dataset. The output
includes:

- end-to-end accuracy, where inference errors count as failures
- valid-response accuracy
- inference error rate
- retry rate
- per-category scores
- individual predictions and errors

Evaluation output is written to `results/`, which is ignored by Git.

To evaluate structured category and summary output:

```bash
python evals/run_evaluation.py \
  --dataset data/golden_emails_v2.jsonl \
  --prompt prompts/email_classifier_v3.txt \
  --model openai/gpt-oss-20b \
  --output results/v3_results.json \
  --structured-output
```

## Dataset versions

- `data/golden_emails_v1.jsonl` contains the original 15 cases with enriched
  summary, difficulty, and review-note metadata.
- `data/golden_emails_v2.jsonl` expands the suite to 50 balanced cases, with 10
  examples per category and deliberate easy, medium, hard, ambiguous, short,
  typo-heavy, mixed-language, and sarcastic coverage.

Every record contains `id`, `email`, `expected_label`, `expected_summary`,
`expected_difficulty`, and `notes`. Treat the expected labels and summaries as
ground truth only after human review.

Always compare runs from the same dataset version. Maintain a separate baseline
for each version; a 15-case v1 run is not comparable to a 50-case v2 run.

- `results/baseline_results.json` is the v1 baseline.
- `results/baseline_v2_results.json` is the approved v2 baseline:
  dataset v2, prompt v3, `openai/gpt-oss-20b`, structured output, 98%
  end-to-end accuracy.

## Continuous integration

GitHub Actions runs the v2 regression gate on every push and pull request to
`main`:

```text
data/golden_emails_v2.jsonl
        ↓
prompts/email_classifier_v3.txt
        ↓
openai/gpt-oss-20b
        ↓
evals/run_evaluation.py --structured-output
        ↓
Compare against results/baseline_v2_results.json
        ↓
PASS / FAIL
```

A `REGRESSION` status fails the workflow. Improvement or no change passes.
The workflow uses prompt v3, not v2, because that is the configuration used to
create the approved 50-case baseline.

## Compare two runs

Use one result as a baseline and compare a newer prompt or model against it:

```bash
python evals/compare_results.py \
  --baseline results/baseline_v2_results.json \
  --current results/current_results.json
```

The comparison reports overall metric changes, newly regressed examples, fixed
examples, changed predictions, and inference-error changes. Both files must
come from the same dataset version.

For a regression check against the approved v2 baseline:

```bash
python evals/run_evaluation.py \
  --dataset data/golden_emails_v2.jsonl \
  --prompt prompts/email_classifier_v3.txt \
  --model openai/gpt-oss-20b \
  --output results/current_results.json \
  --structured-output

python evals/compare_results.py \
  --baseline results/baseline_v2_results.json \
  --current results/current_results.json
```

## Compare multiple models

First run each model against the same prompt and golden dataset. For example:

```bash
python evals/run_evaluation.py \
  --dataset data/golden_emails_v1.jsonl \
  --prompt prompts/email_classifier_v2.txt \
  --model openai/gpt-oss-20b \
  --output results/gpt_oss_20b_results.json
```

Then generate a ranked JSON and Markdown report:

```bash
python evals/generate_model_comparison_report.py \
  --results \
    results/gpt_oss_20b_results.json \
    results/qwen_3_6_27b_results.json
```

Reports are written to `reports/` by default. This directory is also ignored by
Git because reports are generated artifacts.

## Repository structure

```text
data/       Versioned golden evaluation datasets
evals/      Evaluation, scoring, comparison, and reporting tools
prompts/    Versioned prompt templates
src/        Email classifier and Groq integration
```

Keep the dataset, prompts, source code, Python version, and pinned dependencies
under version control so future runs can reproduce the same evaluation setup.
