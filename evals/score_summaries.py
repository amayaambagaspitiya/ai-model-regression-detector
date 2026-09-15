"""Score predicted summaries with DeepEval GEval."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams

from src.groq_judge import GroqJudge


SUMMARY_EVALUATION_STEPS = [
    "Compare the predicted summary with the expected summary and the original email.",
    "Check whether the predicted summary captures the same customer request or issue as the expected summary.",
    "Penalize invented facts that are not present in the email or the expected summary.",
    "Penalize summaries that are off-topic, incomplete, or miss the core request.",
    "Do not require identical wording; score semantic meaning, not exact string match.",
]


def resolve_project_path(path: str) -> Path:
    resolved_path = Path(path)
    if not resolved_path.is_absolute():
        resolved_path = PROJECT_ROOT / resolved_path
    return resolved_path


def load_results(file_path: Path) -> dict[str, Any]:
    if not file_path.exists():
        raise FileNotFoundError(f"Results file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_summary_metric(judge: GroqJudge, threshold: float) -> GEval:
    return GEval(
        name="Summary Quality",
        evaluation_steps=SUMMARY_EVALUATION_STEPS,
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        model=judge,
        threshold=threshold,
        async_mode=False,
        verbose_mode=False,
    )


def score_summaries(
    results_data: dict[str, Any],
    judge_model: str,
    threshold: float,
    limit: int | None = None,
) -> dict[str, Any]:
    metric = build_summary_metric(GroqJudge(model_name=judge_model), threshold)
    scored_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []

    rows = results_data.get("results") or []
    if limit is not None:
        rows = rows[:limit]

    for result in rows:
        email_id = result.get("id")
        predicted_summary = result.get("predicted_summary")
        expected_summary = result.get("expected_summary")
        email = result.get("email")

        if not predicted_summary or not expected_summary or not email:
            skipped_rows.append(
                {
                    "id": email_id,
                    "skipped": True,
                    "reason": "missing email, expected_summary, or predicted_summary",
                }
            )
            continue

        test_case = LLMTestCase(
            input=email,
            actual_output=predicted_summary,
            expected_output=expected_summary,
            name=email_id,
        )
        metric.measure(test_case)

        scored_rows.append(
            {
                "id": email_id,
                "expected_label": result.get("expected_label"),
                "expected_summary": expected_summary,
                "predicted_summary": predicted_summary,
                "score": metric.score,
                "reason": metric.reason,
                "success": metric.success,
                "threshold": threshold,
            }
        )

    scores = [row["score"] for row in scored_rows if row["score"] is not None]
    passed = sum(1 for row in scored_rows if row["success"])
    scored_count = len(scored_rows)
    mean_score = sum(scores) / len(scores) if scores else 0.0
    pass_rate = passed / scored_count if scored_count else 0.0
    worst = sorted(
        scored_rows,
        key=lambda row: (row["score"] is None, row["score"] or 0.0),
    )[:5]

    return {
        "configuration": {
            "results_file": None,
            "judge_model": judge_model,
            "threshold": threshold,
            "metric": "Summary Quality",
        },
        "scores": {
            "scored": scored_count,
            "skipped": len(skipped_rows),
            "passed": passed,
            "failed": scored_count - passed,
            "mean_score": mean_score,
            "pass_rate": pass_rate,
        },
        "worst_summaries": [
            {
                "id": row["id"],
                "score": row["score"],
                "reason": row["reason"],
                "predicted_summary": row["predicted_summary"],
                "expected_summary": row["expected_summary"],
            }
            for row in worst
        ],
        "results": scored_rows + skipped_rows,
    }


def print_report(report: dict[str, Any]) -> None:
    scores = report["scores"]
    print("Summary Quality")
    print("---------------")
    print(f'Scored: {scores["scored"]}')
    print(f'Skipped: {scores["skipped"]}')
    print(f'Passed: {scores["passed"]}')
    print(f'Failed: {scores["failed"]}')
    print(f'Mean score: {scores["mean_score"]:.2f}')
    print(f'Pass rate: {scores["pass_rate"]:.2%}')
    print()
    print("Worst summaries")
    print("---------------")

    if not report["worst_summaries"]:
        print("None")
        return

    for row in report["worst_summaries"]:
        print(
            f'{row["id"]}: score={row["score"]:.2f} | '
            f'reason={row["reason"]}'
        )


def save_report(report: dict[str, Any], output_path: str) -> None:
    resolved_output_path = resolve_project_path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)

    with resolved_output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(f"\nReport saved to: {resolved_output_path}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score predicted summaries with DeepEval GEval."
    )
    parser.add_argument(
        "--results",
        required=True,
        help="Path to an evaluation results JSON file.",
    )
    parser.add_argument(
        "--judge-model",
        default="openai/gpt-oss-20b",
        help="Groq model ID to use as the LLM judge.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Minimum GEval score required to pass.",
    )
    parser.add_argument(
        "--output",
        default="reports/summary_quality_report.json",
        help="Path where the summary-quality report will be saved.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Score only the first N result rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    results_path = resolve_project_path(args.results)
    results_data = load_results(results_path)
    report = score_summaries(
        results_data=results_data,
        judge_model=args.judge_model,
        threshold=args.threshold,
        limit=args.limit,
    )
    report["configuration"]["results_file"] = str(
        results_path.relative_to(PROJECT_ROOT)
        if results_path.is_relative_to(PROJECT_ROOT)
        else results_path
    )
    print_report(report)
    save_report(report, args.output)


if __name__ == "__main__":
    main()
