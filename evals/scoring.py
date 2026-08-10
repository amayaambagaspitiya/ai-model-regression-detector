from collections import defaultdict


def calculate_scores(results: list[dict]) -> dict:
    total = len(results)
    correct = sum(result["correct"] for result in results)
    error_count = sum(result.get("error") is not None for result in results)
    incorrect = total - correct - error_count
    retry_count = sum(
        1 for result in results if result.get("retry_count", 0) > 0
    )

    valid_responses = total - error_count

    # End-to-end: errors count against accuracy (user-facing quality).
    end_to_end_accuracy = correct / total if total else 0

    # Valid-response: accuracy only among successful inferences.
    valid_response_accuracy = (
        correct / valid_responses if valid_responses else 0
    )

    inference_error_rate = error_count / total if total else 0
    retry_rate = retry_count / total if total else 0

    category_stats = defaultdict(
        lambda: {"total": 0, "correct": 0, "errors": 0, "retries": 0}
    )

    for result in results:
        category = result["expected_label"]

        category_stats[category]["total"] += 1

        if result.get("retry_count", 0) > 0:
            category_stats[category]["retries"] += 1

        if result.get("error") is not None:
            category_stats[category]["errors"] += 1
        elif result["correct"]:
            category_stats[category]["correct"] += 1

    category_scores = {}

    for category, stats in category_stats.items():
        category_total = stats["total"]
        category_correct = stats["correct"]
        category_errors = stats["errors"]

        category_scores[category] = {
            "total": category_total,
            "correct": category_correct,
            "errors": category_errors,
            "incorrect": category_total - category_correct - category_errors,
            "retries": stats["retries"],
            "accuracy": (
                category_correct / category_total
                if category_total
                else 0
            ),
        }

    return {
        "total": total,
        "correct": correct,
        "incorrect": incorrect,
        "errors": error_count,
        "retries": retry_count,
        # Keep "accuracy" as end-to-end for backward compatibility.
        "accuracy": end_to_end_accuracy,
        "end_to_end_accuracy": end_to_end_accuracy,
        "valid_response_accuracy": valid_response_accuracy,
        "inference_error_rate": inference_error_rate,
        "retry_rate": retry_rate,
        "categories": category_scores,
    }


def print_scores(scores: dict) -> None:
    print("\nEvaluation Summary")
    print("------------------")
    print(f'Total: {scores["total"]}')
    print(f'Correct: {scores["correct"]}')
    print(f'Incorrect: {scores["incorrect"]}')
    print(f'Inference errors: {scores["errors"]}')
    print(f'Retried examples: {scores["retries"]}')
    print(f'End-to-end accuracy: {scores["end_to_end_accuracy"]:.2%}')
    print(f'Valid-response accuracy: {scores["valid_response_accuracy"]:.2%}')
    print(f'Inference error rate: {scores["inference_error_rate"]:.2%}')
    print(f'Retry rate: {scores["retry_rate"]:.2%}')

    print("\nCategory Scores")
    print("---------------")

    for category, stats in sorted(scores["categories"].items()):
        print(
            f'{category}: '
            f'{stats["correct"]}/{stats["total"]} correct '
            f'({stats["accuracy"]:.2%}) | '
            f'errors={stats["errors"]} | '
            f'retries={stats["retries"]}'
        )
