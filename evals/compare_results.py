import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load_results(file_path: str) -> dict[str, Any]:
    """Load an evaluation results JSON file."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def index_results_by_id(results_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Create a dictionary keyed by email ID."""
    return {
        result["id"]: result
        for result in results_data["results"]
    }


def _score_metric(scores: dict[str, Any], key: str, fallback: str = "accuracy") -> float:
    """Read a metric with fallback for older result files."""
    if key in scores:
        return scores[key]
    return scores.get(fallback, 0.0)


def compare_results(
    baseline: dict[str, Any],
    current: dict[str, Any],
) -> dict[str, Any]:
    """Compare current evaluation results against the baseline."""
    baseline_dataset = (baseline.get("configuration") or {}).get(
        "dataset_path"
    )
    current_dataset = (current.get("configuration") or {}).get(
        "dataset_path"
    )

    if baseline_dataset != current_dataset:
        raise ValueError(
            "Baseline and current results use different dataset versions: "
            f"{baseline_dataset!r} != {current_dataset!r}"
        )

    baseline_scores = baseline["scores"]
    current_scores = current["scores"]

    baseline_end_to_end = _score_metric(baseline_scores, "end_to_end_accuracy")
    current_end_to_end = _score_metric(current_scores, "end_to_end_accuracy")
    end_to_end_change = current_end_to_end - baseline_end_to_end

    baseline_valid = _score_metric(
        baseline_scores, "valid_response_accuracy", "accuracy"
    )
    current_valid = _score_metric(
        current_scores, "valid_response_accuracy", "accuracy"
    )
    valid_response_change = current_valid - baseline_valid

    baseline_error_rate = baseline_scores.get("inference_error_rate")
    if baseline_error_rate is None:
        total = baseline_scores.get("total") or 0
        errors = baseline_scores.get("errors") or 0
        baseline_error_rate = errors / total if total else 0.0

    current_error_rate = current_scores.get("inference_error_rate")
    if current_error_rate is None:
        total = current_scores.get("total") or 0
        errors = current_scores.get("errors") or 0
        current_error_rate = errors / total if total else 0.0

    error_rate_change = current_error_rate - baseline_error_rate

    baseline_retry_rate = baseline_scores.get("retry_rate", 0.0)
    current_retry_rate = current_scores.get("retry_rate", 0.0)
    retry_rate_change = current_retry_rate - baseline_retry_rate

    baseline_by_id = index_results_by_id(baseline)
    current_by_id = index_results_by_id(current)

    baseline_ids = set(baseline_by_id)
    current_ids = set(current_by_id)

    if baseline_ids != current_ids:
        missing_from_current = sorted(baseline_ids - current_ids)
        new_in_current = sorted(current_ids - baseline_ids)

        raise ValueError(
            "Baseline and current results contain different test cases.\n"
            f"Missing from current: {missing_from_current}\n"
            f"New in current: {new_in_current}"
        )

    new_regressions = []
    fixed_cases = []
    unchanged_failures = []
    unchanged_passes = []
    changed_predictions = []
    new_inference_errors = []
    fixed_inference_errors = []

    for email_id in sorted(baseline_ids):
        baseline_result = baseline_by_id[email_id]
        current_result = current_by_id[email_id]

        baseline_correct = baseline_result["correct"]
        current_correct = current_result["correct"]

        baseline_had_error = baseline_result.get("error") is not None
        current_had_error = current_result.get("error") is not None

        if (not baseline_had_error) and current_had_error:
            new_inference_errors.append(email_id)
        elif baseline_had_error and (not current_had_error):
            fixed_inference_errors.append(email_id)

        if baseline_correct and not current_correct:
            new_regressions.append(email_id)

        elif not baseline_correct and current_correct:
            fixed_cases.append(email_id)

        elif not baseline_correct and not current_correct:
            unchanged_failures.append(email_id)

        else:
            unchanged_passes.append(email_id)

        if (
            baseline_result["predicted_label"]
            != current_result["predicted_label"]
        ):
            changed_predictions.append(
                {
                    "id": email_id,
                    "expected_label": current_result["expected_label"],
                    "baseline_prediction": baseline_result["predicted_label"],
                    "current_prediction": current_result["predicted_label"],
                    "baseline_error": baseline_result.get("error"),
                    "current_error": current_result.get("error"),
                    "baseline_retry_count": baseline_result.get(
                        "retry_count", 0
                    ),
                    "current_retry_count": current_result.get(
                        "retry_count", 0
                    ),
                }
            )

    if new_regressions or new_inference_errors or end_to_end_change < 0:
        status = "REGRESSION"
    elif end_to_end_change > 0 or fixed_cases or fixed_inference_errors:
        status = "IMPROVEMENT"
    else:
        status = "NO CHANGE"

    return {
        "baseline_end_to_end_accuracy": baseline_end_to_end,
        "current_end_to_end_accuracy": current_end_to_end,
        "end_to_end_accuracy_change": end_to_end_change,
        "baseline_valid_response_accuracy": baseline_valid,
        "current_valid_response_accuracy": current_valid,
        "valid_response_accuracy_change": valid_response_change,
        "baseline_inference_error_rate": baseline_error_rate,
        "current_inference_error_rate": current_error_rate,
        "inference_error_rate_change": error_rate_change,
        "baseline_retry_rate": baseline_retry_rate,
        "current_retry_rate": current_retry_rate,
        "retry_rate_change": retry_rate_change,
        # Keep legacy keys for older callers/printing habits.
        "baseline_accuracy": baseline_end_to_end,
        "current_accuracy": current_end_to_end,
        "accuracy_change": end_to_end_change,
        "status": status,
        "new_regressions": new_regressions,
        "fixed_cases": fixed_cases,
        "unchanged_failures": unchanged_failures,
        "unchanged_passes": unchanged_passes,
        "new_inference_errors": new_inference_errors,
        "fixed_inference_errors": fixed_inference_errors,
        "changed_predictions": changed_predictions,
    }


def print_comparison(comparison: dict[str, Any]) -> None:
    """Print a readable comparison summary."""
    print("Regression Comparison")
    print("---------------------")
    print(
        f'End-to-end accuracy: '
        f'{comparison["baseline_end_to_end_accuracy"]:.2%} → '
        f'{comparison["current_end_to_end_accuracy"]:.2%} '
        f'({comparison["end_to_end_accuracy_change"]:+.2%})'
    )
    print(
        f'Valid-response accuracy: '
        f'{comparison["baseline_valid_response_accuracy"]:.2%} → '
        f'{comparison["current_valid_response_accuracy"]:.2%} '
        f'({comparison["valid_response_accuracy_change"]:+.2%})'
    )
    print(
        f'Inference error rate: '
        f'{comparison["baseline_inference_error_rate"]:.2%} → '
        f'{comparison["current_inference_error_rate"]:.2%} '
        f'({comparison["inference_error_rate_change"]:+.2%})'
    )
    print(
        f'Retry rate: '
        f'{comparison["baseline_retry_rate"]:.2%} → '
        f'{comparison["current_retry_rate"]:.2%} '
        f'({comparison["retry_rate_change"]:+.2%})'
    )

    print()
    print(f'Status: {comparison["status"]}')
    print(f'New regressions: {comparison["new_regressions"]}')
    print(f'Fixed cases: {comparison["fixed_cases"]}')
    print(f'Unchanged failures: {comparison["unchanged_failures"]}')
    print(f'New inference errors: {comparison["new_inference_errors"]}')
    print(f'Fixed inference errors: {comparison["fixed_inference_errors"]}')

    print("\nChanged Predictions")
    print("-------------------")

    if not comparison["changed_predictions"]:
        print("None")
        return

    for item in comparison["changed_predictions"]:
        print(
            f'{item["id"]}: '
            f'expected={item["expected_label"]} | '
            f'baseline={item["baseline_prediction"]} | '
            f'current={item["current_prediction"]} | '
            f'retries={item["baseline_retry_count"]}→'
            f'{item["current_retry_count"]}'
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare current evaluation results against a baseline."
    )

    parser.add_argument(
        "--baseline",
        required=True,
        help="Path to the baseline results JSON file.",
    )

    parser.add_argument(
        "--current",
        required=True,
        help="Path to the current results JSON file.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    baseline_results = load_results(args.baseline)
    current_results = load_results(args.current)

    comparison = compare_results(
        baseline=baseline_results,
        current=current_results,
    )

    print_comparison(comparison)

    if comparison["status"] == "REGRESSION":
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()