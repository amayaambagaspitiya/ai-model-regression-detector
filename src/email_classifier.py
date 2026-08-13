import json
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, ValidationError


load_dotenv()

VALID_LABELS = {"billing", "support", "sales", "spam", "other"}


class ClassificationResult(BaseModel):
    category: Literal["billing", "support", "sales", "spam", "other"]
    summary: str


def parse_classification_response(raw_output: str) -> ClassificationResult:
    try:
        parsed = json.loads(raw_output.strip())
        return ClassificationResult.model_validate(parsed)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(
            f"Invalid classification output: {raw_output}"
        ) from exc


class ClassificationError(Exception):
    """Raised when classification fails after optional retry."""

    def __init__(self, message: str, retry_count: int = 0):
        super().__init__(message)
        self.retry_count = retry_count

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise RuntimeError(
        "GROQ_API_KEY is missing. Add it to your .env file."
    )

client = Groq(api_key=api_key)


def load_prompt(prompt_path: str) -> str:
    """Load a prompt template from a text file."""
    path = Path(prompt_path)

    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    return path.read_text(encoding="utf-8")


def build_prompt(prompt_template: str, email_text: str) -> str:
    """Insert the email into the prompt template."""
    return prompt_template.replace("{{email}}", email_text)


def get_completion_settings(model: str) -> dict:
    if model.startswith("qwen/"):
        return {
            "max_completion_tokens": 1500,
            "reasoning_format": "hidden",
        }

    if model.startswith("openai/gpt-oss"):
        return {
            "max_completion_tokens": 200,
            "reasoning_format": "hidden",
        }

    return {
        "max_completion_tokens": 50,
    }


def normalize_label(raw_output: str) -> str:
    label = raw_output.strip().lower()

    if label not in VALID_LABELS:
        raise ValueError(
            f"Invalid model output: {raw_output!r}"
        )

    return label


def classify_email(
    email_text: str,
    prompt_path: str = "prompts/email_classifier_v1.txt",
    model: str = "llama-3.1-8b-instant",
    structured_output: bool = False,
) -> str | ClassificationResult:
    """Classify an email using a model hosted on Groq."""
    prompt_template = load_prompt(prompt_path)
    final_prompt = build_prompt(prompt_template, email_text)

    settings = get_completion_settings(model)

    completion_kwargs = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": final_prompt,
            }
        ],
        "temperature": 0,
        "max_completion_tokens": settings["max_completion_tokens"],
    }

    if "reasoning_format" in settings:
        completion_kwargs["reasoning_format"] = settings["reasoning_format"]

    completion = client.chat.completions.create(**completion_kwargs)

    raw_output = completion.choices[0].message.content

    if raw_output is None:
        raise ValueError("The model returned an empty response.")

    # Debug: expose the real model output before normalization.
    print(f"\nMODEL: {model}")
    print(f"RAW OUTPUT: {raw_output!r}")

    if structured_output:
        return parse_classification_response(raw_output)

    return normalize_label(raw_output)


def classify_email_with_retry(
    email_text: str,
    prompt_path: str = "prompts/email_classifier_v1.txt",
    model: str = "llama-3.1-8b-instant",
    structured_output: bool = False,
) -> tuple[str | ClassificationResult, int]:
    """Classify once; on empty/invalid output, retry once.

    Returns:
        (predicted_label, retry_count)

    retry_count is 0 on first-attempt success, 1 if a retry was used.
    If the retry also fails with an empty/invalid output, raises
    ClassificationError with retry_count=1.
    """
    try:
        return classify_email(
            email_text=email_text,
            prompt_path=prompt_path,
            model=model,
            structured_output=structured_output,
        ), 0
    except ValueError as first_error:
        print(
            f"First attempt failed ({first_error}); retrying once..."
        )

        try:
            result = classify_email(
                email_text=email_text,
                prompt_path=prompt_path,
                model=model,
                structured_output=structured_output,
            )
            return result, 1
        except ValueError as second_error:
            raise ClassificationError(
                str(second_error),
                retry_count=1,
            ) from second_error


if __name__ == "__main__":
    test_email = "I was charged twice for my subscription."

    predicted_label, retry_count = classify_email_with_retry(test_email)

    print(f"Predicted label: {predicted_label}")
    print(f"Retry count: {retry_count}")