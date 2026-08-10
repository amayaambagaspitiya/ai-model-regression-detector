"""Generate a multi-model comparison report from evaluation result JSON files."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_results(file_path: Path) -> dict[str, Any]:
    if not file_path.exists():
        raise FileNotFoundError(f"Results file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def normalize_metrics(scores: dict[str, Any]) -> dict[str, float | int]:
    """Normalize older and newer result schemas into one metric set."""
    total = int(scores.get("total") or 0)
    correct = int(scores.get("correct") or 0)
    errors = int(scores.get("errors") or 0)
    retries = int(scores.get("retries") or 0)

    end_to_end = scores.get("end_to_end_accuracy")
    if end_to_end is None:
        end_to_end = scores.get("accuracy") or (correct / total if total else 0.0)

    valid_response = scores.get("valid_response_accuracy")
    if valid_response is None:
        valid_total = total - errors
        valid_response = correct / valid_total if valid_total else 0.0

    error_rate = scores.get("inference_error_rate")
    if error_rate is None:
        error_rate = errors / total if total else 0.0

    retry_rate = scores.get("retry_rate")
    if retry_rate is None:
        retry_rate = retries / total if total else 0.0

    return {
        "total": total,
        "correct": correct,
        "incorrect": int(scores.get("incorrect") or (total - correct - errors)),
        "errors": errors,
        "retries": retries,
        "end_to_end_accuracy": float(end_to_end),
        "valid_response_accuracy": float(valid_response),
        "inference_error_rate": float(error_rate),
        "retry_rate": float(retry_rate),
    }


def summarize_model(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    configuration = data.get("configuration") or {}
    metrics = normalize_metrics(data.get("scores") or {})

    failures = []
    for result in data.get("results") or []:
        if result.get("correct") and not result.get("error"):
            continue

        failures.append(
            {
                "id": result.get("id"),
                "expected_label": result.get("expected_label"),
                "predicted_label": result.get("predicted_label"),
                "error": result.get("error"),
                "retry_count": result.get("retry_count", 0),
            }
        )

    return {
        "source_file": str(path.relative_to(PROJECT_ROOT)),
        "model": configuration.get("model") or path.stem,
        "prompt_path": configuration.get("prompt_path"),
        "temperature": configuration.get("temperature"),
        "metrics": metrics,
        "failures": failures,
        "results_by_id": {
            result["id"]: result for result in data.get("results") or []
        },
    }


def rank_models(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank by end-to-end accuracy, then fewer errors, then fewer retries."""

    def sort_key(model: dict[str, Any]) -> tuple[float, float, float, float]:
        metrics = model["metrics"]
        return (
            -metrics["end_to_end_accuracy"],
            -metrics["valid_response_accuracy"],
            metrics["inference_error_rate"],
            metrics["retry_rate"],
        )

    ranked = sorted(models, key=sort_key)
    for index, model in enumerate(ranked, start=1):
        model["rank"] = index
    return ranked


def collect_disagreements(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    all_ids: set[str] = set()
    for model in models:
        all_ids.update(model["results_by_id"].keys())

    disagreements = []

    for email_id in sorted(all_ids):
        predictions = {}
        expected_labels = set()

        for model in models:
            result = model["results_by_id"].get(email_id)
            if result is None:
                predictions[model["model"]] = {
                    "predicted_label": None,
                    "correct": False,
                    "error": "missing from this run",
                    "retry_count": 0,
                }
                continue

            expected_labels.add(result.get("expected_label"))
            predictions[model["model"]] = {
                "predicted_label": result.get("predicted_label"),
                "correct": bool(result.get("correct")),
                "error": result.get("error"),
                "retry_count": result.get("retry_count", 0),
            }

        predicted_values = {
            (
                details["predicted_label"],
                details["error"] is not None,
            )
            for details in predictions.values()
        }
        any_incorrect = any(
            (not details["correct"]) or details["error"]
            for details in predictions.values()
        )

        if len(predicted_values) > 1 or any_incorrect:
            disagreements.append(
                {
                    "id": email_id,
                    "expected_label": (
                        next(iter(expected_labels))
                        if len(expected_labels) == 1
                        else sorted(expected_labels)
                    ),
                    "predictions": predictions,
                }
            )

    return disagreements


def format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def build_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Multi-Model Comparison Report")
    lines.append("")
    lines.append(f"Generated at: `{report['generated_at']}`")
    lines.append("")
    lines.append("## Ranking")
    lines.append("")
    lines.append(
        "| Rank | Model | End-to-end | Valid-response | Error rate | Retry rate | Correct | Errors | Retries |"
    )
    lines.append(
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )

    for model in report["models"]:
        metrics = model["metrics"]
        lines.append(
            "| "
            f"{model['rank']} | "
            f"`{model['model']}` | "
            f"{format_pct(metrics['end_to_end_accuracy'])} | "
            f"{format_pct(metrics['valid_response_accuracy'])} | "
            f"{format_pct(metrics['inference_error_rate'])} | "
            f"{format_pct(metrics['retry_rate'])} | "
            f"{metrics['correct']}/{metrics['total']} | "
            f"{metrics['errors']} | "
            f"{metrics['retries']} |"
        )

    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append("| Model | Prompt | Temperature | Source |")
    lines.append("| --- | --- | ---: | --- |")
    for model in report["models"]:
        lines.append(
            f"| `{model['model']}` | `{model['prompt_path']}` | "
            f"{model['temperature']} | `{model['source_file']}` |"
        )

    lines.append("")
    lines.append("## Failures And Inference Errors")
    lines.append("")

    any_failures = False
    for model in report["models"]:
        if not model["failures"]:
            continue

        any_failures = True
        lines.append(f"### `{model['model']}`")
        lines.append("")
        for failure in model["failures"]:
            if failure["error"]:
                detail = f"error={failure['error']!r}"
            else:
                detail = f"predicted={failure['predicted_label']}"

            lines.append(
                f"- `{failure['id']}`: expected=`{failure['expected_label']}` | "
                f"{detail} | retries={failure['retry_count']}"
            )
        lines.append("")

    if not any_failures:
        lines.append("No failures or inference errors.")
        lines.append("")

    lines.append("## Disagreements Across Models")
    lines.append("")

    if not report["disagreements"]:
        lines.append("All models agreed on every example.")
        lines.append("")
    else:
        for item in report["disagreements"]:
            lines.append(
                f"### `{item['id']}` (expected=`{item['expected_label']}`)"
            )
            lines.append("")
            for model_name, details in item["predictions"].items():
                if details["error"]:
                    status = f"ERROR ({details['error']})"
                elif details["correct"]:
                    status = "PASS"
                else:
                    status = "FAIL"

                lines.append(
                    f"- `{model_name}`: {status} | "
                    f"predicted=`{details['predicted_label']}` | "
                    f"retries={details['retry_count']}"
                )
            lines.append("")

    winner = report["models"][0]
    lines.append("## Recommendation")
    lines.append("")
    lines.append(
        f"Best end-to-end model under the shared prompt/dataset: "
        f"`{winner['model']}` "
        f"({format_pct(winner['metrics']['end_to_end_accuracy'])} end-to-end, "
        f"{format_pct(winner['metrics']['valid_response_accuracy'])} valid-response, "
        f"{format_pct(winner['metrics']['inference_error_rate'])} error rate, "
        f"{format_pct(winner['metrics']['retry_rate'])} retry rate)."
    )
    lines.append("")

    return "\n".join(lines)


def build_report(result_paths: list[Path]) -> dict[str, Any]:
    models = [
        summarize_model(path, load_results(path))
        for path in result_paths
    ]
    ranked = rank_models(models)
    disagreements = collect_disagreements(ranked)

    # Drop heavy per-id maps from the saved report payload.
    compact_models = []
    for model in ranked:
        compact = dict(model)
        compact.pop("results_by_id", None)
        compact_models.append(compact)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "result_files": [str(path.relative_to(PROJECT_ROOT)) for path in result_paths],
        "models": compact_models,
        "disagreements": disagreements,
        "winner": compact_models[0]["model"] if compact_models else None,
    }


def print_terminal_summary(report: dict[str, Any]) -> None:
    print("Multi-Model Comparison")
    print("----------------------")
    print(
        f"{'Rank':<5} {'Model':<28} {'E2E':>8} {'Valid':>8} "
        f"{'ErrRate':>8} {'Retry':>8}"
    )

    for model in report["models"]:
        metrics = model["metrics"]
        print(
            f"{model['rank']:<5} {model['model']:<28} "
            f"{format_pct(metrics['end_to_end_accuracy']):>8} "
            f"{format_pct(metrics['valid_response_accuracy']):>8} "
            f"{format_pct(metrics['inference_error_rate']):>8} "
            f"{format_pct(metrics['retry_rate']):>8}"
        )

    print()
    print(f"Winner: {report['winner']}")
    print(f"Disagreements: {len(report['disagreements'])}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one multi-model comparison report from evaluation "
            "result JSON files."
        )
    )
    parser.add_argument(
        "--results",
        nargs="+",
        required=True,
        help="One or more evaluation result JSON files.",
    )
    parser.add_argument(
        "--output-markdown",
        default="reports/model_comparison_report.md",
        help="Markdown report output path (relative to project root).",
    )
    parser.add_argument(
        "--output-json",
        default="reports/model_comparison_report.json",
        help="JSON report output path (relative to project root).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    result_paths = [resolve_path(path) for path in args.results]
    report = build_report(result_paths)

    markdown_path = resolve_path(args.output_markdown)
    json_path = resolve_path(args.output_json)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    markdown = build_markdown(report)
    markdown_path.write_text(markdown, encoding="utf-8")

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print_terminal_summary(report)
    print()
    print(f"Markdown report: {markdown_path}")
    print(f"JSON report: {json_path}")


if __name__ == "__main__":
    main()
