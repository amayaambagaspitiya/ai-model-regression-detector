"""Groq-backed LLM judge for DeepEval metrics."""

from __future__ import annotations

import json
import re
from typing import Any, Type

from groq import Groq
from pydantic import BaseModel

from deepeval.models.base_model import DeepEvalBaseLLM

from src.email_classifier import client as groq_client
from src.email_classifier import get_completion_settings


def extract_json_object(raw_output: str) -> dict[str, Any]:
    """Parse the first JSON object from a model response."""
    text = raw_output.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Judge response is not valid JSON: {raw_output!r}")

    parsed = json.loads(text[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError(f"Judge JSON must be an object: {raw_output!r}")

    return parsed


class GroqJudge(DeepEvalBaseLLM):
    """Wrap a Groq chat model so DeepEval can use it as an LLM-as-judge."""

    def __init__(
        self,
        model_name: str = "openai/gpt-oss-20b",
        client: Groq | None = None,
    ):
        self.model_name = model_name
        self.client = client or groq_client

    def load_model(self) -> Groq:
        return self.client

    def get_model_name(self) -> str:
        return self.model_name

    def _complete(self, prompt: str) -> str:
        settings = get_completion_settings(self.model_name)
        completion_kwargs = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_completion_tokens": max(
                1500,
                settings["max_completion_tokens"],
            ),
        }

        if "reasoning_format" in settings:
            completion_kwargs["reasoning_format"] = settings["reasoning_format"]

        completion = self.client.chat.completions.create(**completion_kwargs)
        content = completion.choices[0].message.content

        if not content or not content.strip():
            raise ValueError("The judge model returned an empty response.")

        return content

    def generate(
        self,
        prompt: str,
        schema: Type[BaseModel] | None = None,
    ) -> str | BaseModel:
        raw_output = self._complete(prompt)

        if schema is None:
            return raw_output

        parsed = extract_json_object(raw_output)
        return schema.model_validate(parsed)

    async def a_generate(
        self,
        prompt: str,
        schema: Type[BaseModel] | None = None,
    ) -> str | BaseModel:
        return self.generate(prompt, schema=schema)
