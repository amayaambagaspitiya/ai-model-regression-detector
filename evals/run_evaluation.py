import argparse
import json
import sys
from pathlib import Path

# Allow Python to import from the project root before local imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evals.scoring import calculate_scores, print_scores
from src.email_classifier import (
    ClassificationError,
    ClassificationResult,
    VALID_LABELS,
    classify_email_with_retry,
)


REQUIRED_DATASET_FIELDS = {
    "id",
    "email",
    "expected_label",
    "expected_summary",
    "expected_difficulty",
    "notes",
}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}


def resolve_project_path(path: str) -> Path:
    resolved_path = Path(path)
    if not resolved_path.is_absolute():
        resolved_path = PROJECT_ROOT / resolved_path
    return resolved_path


def load_golden_dataset(dataset_path: Path) -> list[dict]:
    records = []
    seen_ids = set()

    with dataset_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {error}"
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object on line {line_number}"
                )

            missing_fields = REQUIRED_DATASET_FIELDS - record.keys()
            if missing_fields:
                raise ValueError(
                    f"Missing fields on line {line_number}: "
                    f"{sorted(missing_fields)}"
                )

            for field in REQUIRED_DATASET_FIELDS:
                if not isinstance(record[field], str) or not record[field].strip():
                    raise ValueError(
                        f"Field {field!r} must be a non-empty string "
                        f"on line {line_number}"
                    )

            email_id = record["id"]
            if email_id in seen_ids:
                raise ValueError(
                    f"Duplicate email ID on line {line_number}: {email_id}"
                )

            if record["expected_label"] not in VALID_LABELS:
                raise ValueError(
                    f"Invalid expected_label on line {line_number}: "
                    f"{record['expected_label']!r}"
                )

            if record["expected_difficulty"] not in VALID_DIFFICULTIES:
                raise ValueError(
                    f"Invalid expected_difficulty on line {line_number}: "
                    f"{record['expected_difficulty']!r}"
                )

            seen_ids.add(email_id)
            records.append(record)

    return records


def run_evaluation(
    prompt_path: str,
    model: str,
    dataset_path: str,
    structured_output: bool = False,
) -> list[dict]:
    dataset = load_golden_dataset(resolve_project_path(dataset_path))
    results = []

    for item in dataset:
        expected_label = item["expected_label"]
        error = None
        predicted_label = None
        predicted_summary = None
        retry_count = 0

        try:
            classification, retry_count = classify_email_with_retry(
                email_text=item["email"],
                prompt_path=prompt_path,
                model=model,
                structured_output=structured_output,
            )

            if isinstance(classification, ClassificationResult):
                predicted_label = classification.category
                predicted_summary = classification.summary
            else:
                predicted_label = classification

            correct = predicted_label == expected_label
        except ClassificationError as exc:
            predicted_label = None
            correct = False
            error = str(exc)
            retry_count = exc.retry_count
        except Exception as exc:
            predicted_label = None
            correct = False
            error = str(exc)
            retry_count = 0

        results.append(
            {
                "id": item["id"],
                "email": item["email"],
                "expected_label": expected_label,
                "expected_summary": item["expected_summary"],
                "expected_difficulty": item["expected_difficulty"],
                "notes": item["notes"],
                "predicted_label": predicted_label,
                "predicted_summary": predicted_summary,
                "correct": correct,
                "error": error,
                "retry_count": retry_count,
            }
        )

    return results


def print_results(results: list[dict]) -> None:
    for result in results:
        retry_note = (
            f' | retries={result["retry_count"]}'
            if result.get("retry_count", 0)
            else ""
        )

        if result["error"]:
            print(
                f'{result["id"]}: ERROR | '
                f'expected={result["expected_label"]} | '
                f'error={result["error"]}{retry_note}'
            )
            continue

        status = "PASS" if result["correct"] else "FAIL"

        print(
            f'{result["id"]}: {status} | '
            f'expected={result["expected_label"]} | '
            f'predicted={result["predicted_label"]}{retry_note}'
        )

    total = len(results)
    correct = sum(result["correct"] for result in results)
    errors = sum(result["error"] is not None for result in results)
    retries = sum(1 for result in results if result.get("retry_count", 0) > 0)

    print()
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Incorrect: {total - correct - errors}")
    print(f"Inference errors: {errors}")
    print(f"Retried examples: {retries}")


def save_results(
    results: list[dict],
    scores: dict,
    prompt_path: str,
    model: str,
    output_path: str,
    dataset_path: str,
    structured_output: bool = False,
) -> None:
    resolved_output_path = PROJECT_ROOT / output_path
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "configuration": {
            "prompt_path": prompt_path,
            "model": model,
            "dataset_path": dataset_path,
            "temperature": 0,
            "structured_output": structured_output,
        },
        "scores": scores,
        "results": results,
    }

    with resolved_output_path.open("w", encoding="utf-8") as file:
        json.dump(output_data, file, indent=2)

    print(f"\nResults saved to: {resolved_output_path}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an email classifier."
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to a versioned golden dataset.",
    )

    parser.add_argument(
        "--prompt",
        required=True,
        help="Path to the prompt file.",
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Groq model ID to evaluate.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path where results will be saved.",
    )

    parser.add_argument(
        "--structured-output",
        action="store_true",
        help="Parse and validate category and summary JSON output.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    evaluation_results = run_evaluation(
        prompt_path=args.prompt,
        model=args.model,
        dataset_path=args.dataset,
        structured_output=args.structured_output,
    )

    print_results(evaluation_results)

    scores = calculate_scores(evaluation_results)
    print_scores(scores)

    save_results(
        results=evaluation_results,
        scores=scores,
        prompt_path=args.prompt,
        model=args.model,
        output_path=args.output,
        dataset_path=args.dataset,
        structured_output=args.structured_output,
    )

