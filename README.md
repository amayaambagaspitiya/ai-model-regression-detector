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
  --prompt prompts/email_classifier_v2.txt \
  --model llama-3.1-8b-instant \
  --output results/llama_3_1_8b_results.json
```

Each run evaluates every example in `data/golden_emails.jsonl`. The output
includes:

- end-to-end accuracy, where inference errors count as failures
- valid-response accuracy
- inference error rate
- retry rate
- per-category scores
- individual predictions and errors

Evaluation output is written to `results/`, which is ignored by Git.

## Compare two runs

Use one result as a baseline and compare a newer prompt or model against it:

```bash
python evals/compare_results.py \
  --baseline results/baseline_results.json \
  --current results/current_results.json
```

The comparison reports overall metric changes, newly regressed examples, fixed
examples, changed predictions, and inference-error changes.

For a prompt regression check, run the same model with both prompt versions:

```bash
python evals/run_evaluation.py \
  --prompt prompts/email_classifier_v1.txt \
  --model llama-3.1-8b-instant \
  --output results/baseline_results.json

python evals/run_evaluation.py \
  --prompt prompts/email_classifier_v2.txt \
  --model llama-3.1-8b-instant \
  --output results/current_results.json
```

## Compare multiple models

First run each model against the same prompt and golden dataset. For example:

```bash
python evals/run_evaluation.py \
  --prompt prompts/email_classifier_v2.txt \
  --model openai/gpt-oss-20b \
  --output results/gpt_oss_20b_results.json
```

Then generate a ranked JSON and Markdown report:

```bash
python evals/generate_model_comparison_report.py \
  --results \
    results/llama_3_1_8b_results.json \
    results/gpt_oss_20b_results.json
```

Reports are written to `reports/` by default. This directory is also ignored by
Git because reports are generated artifacts.

## Repository structure

```text
data/       Golden evaluation dataset
evals/      Evaluation, scoring, comparison, and reporting tools
prompts/    Versioned prompt templates
src/        Email classifier and Groq integration
```

Keep the dataset, prompts, source code, Python version, and pinned dependencies
under version control so future runs can reproduce the same evaluation setup.
