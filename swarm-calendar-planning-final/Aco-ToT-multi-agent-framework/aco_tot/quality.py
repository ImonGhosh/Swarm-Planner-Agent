"""Quality scoring utilities."""

from __future__ import annotations

import inspect
import json
import os
import re
from typing import Awaitable, Callable

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import NotFoundError

from .openrouter_resilience import (
    OPENROUTER_BASE_URL,
    ainvoke_with_backoff,
    is_openrouter_base_url,
    provider_preferences_from_env,
)


ALLOWED_SCORES = {1.0, 0.75, 0.5, 0.25, 0.0}

# Change this constant if you want to switch the scorer model.
SCORER_MODEL = "gpt-5-mini"
SCORER_BASE_URL = OPENROUTER_BASE_URL
SCORER_TEMPERATURE = 0.0


QualityScoreFn = Callable[[str, str | None, str], Awaitable[float] | float]
ResolvedQualityScoreFn = Callable[[str, str | None, str], Awaitable[float]]

_SCORER_LLM: ChatOpenAI | None = None


def _raise_model_unavailable(model: str, err: NotFoundError) -> None:
    raise RuntimeError(
        "Configured endpoint could not serve scorer model "
        f"'{model}' (404/not found). "
        "Set a valid scorer model for your selected backend and retry."
    ) from err


def _get_scorer_llm() -> ChatOpenAI:
    global _SCORER_LLM
    if _SCORER_LLM is not None:
        return _SCORER_LLM

    base_url = os.getenv("SCORER_BASE_URL", SCORER_BASE_URL).strip()
    use_openrouter = is_openrouter_base_url(base_url)
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key and not use_openrouter:
        api_key = "local"
    if not api_key and use_openrouter:
        raise EnvironmentError(
            "Set OPENROUTER_API_KEY (or OPENAI_API_KEY) for training-time quality scoring."
        )

    client_kwargs: dict[str, object] = {
        "base_url": base_url,
        "api_key": api_key,
        "temperature": SCORER_TEMPERATURE,
    }
    if use_openrouter:
        client_kwargs["extra_body"] = {"provider": provider_preferences_from_env()}

    _SCORER_LLM = ChatOpenAI(
        model=SCORER_MODEL,
        **client_kwargs,
    )
    return _SCORER_LLM


def _extract_json_object(text: str) -> dict | None:
    if not text:
        return None
    content = text.strip()
    content = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", content)
    content = re.sub(r"\s*```$", "", content).strip()
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    snippet = content[start : end + 1]
    try:
        parsed = json.loads(snippet)
        if isinstance(parsed, dict):
            return parsed
        return None
    except json.JSONDecodeError:
        return None


def _normalize_score(raw_score: object) -> float:
    try:
        score = float(raw_score)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    for allowed in ALLOWED_SCORES:
        if abs(score - allowed) < 1e-9:
            return allowed
    return 0.0


async def default_quality_score_fn(
    prediction: str,
    golden_plan: str | None,
    task_text: str,
) -> float:
    """
    Strong-LLM rubric scorer for training rewards.

    Allowed outputs: {1.0, 0.75, 0.5, 0.25, 0.0}
    """
    if not prediction:
        return 0.0
    if not task_text:
        return 0.0
    if not golden_plan:
        return 0.0

    system_prompt = (
        "You are a strict calendar-scheduling reward model. "
        "Given TASK, GOLDEN_PLAN, and PREDICTION, return only JSON with key 'score'. "
        "Allowed score values only: 1.0, 0.75, 0.5, 0.25, 0.0.\n\n"
        "Scoring rubric:\n"
        "1.0 = exact match with GOLDEN_PLAN\n"
        "0.75 = valid slot satisfying calendars and preferences, but not GOLDEN_PLAN\n"
        "0.50 = valid slot satisfying hard calendar constraints, but misses soft preference\n"
        "0.25 = parseable slot with right duration/work-hours, but has calendar conflict\n"
        "0.00 = unparsable or clearly invalid\n\n"
        "Output format strictly:\n"
        "{\"score\": <one of [1.0, 0.75, 0.5, 0.25, 0.0]>}\n"
        "No extra keys. No explanation."
    )
    user_prompt = (
        f"TASK:\n{task_text.strip()}\n\n"
        f"GOLDEN_PLAN:\n{golden_plan.strip()}\n\n"
        f"PREDICTION:\n{prediction.strip()}\n"
    )

    llm = _get_scorer_llm()
    response = await ainvoke_with_backoff(
        llm,
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)],
        model_label=SCORER_MODEL,
        on_not_found=_raise_model_unavailable,
    )
    content = getattr(response, "content", "")
    if isinstance(content, list):
        text = "\n".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    else:
        text = str(content)

    parsed = _extract_json_object(text)
    if parsed is None:
        return 0.0
    return _normalize_score(parsed.get("score"))


def resolve_quality_score_fn(
    fn: QualityScoreFn | None = None,
) -> ResolvedQualityScoreFn:
    """Resolve quality function and always return an async callable."""
    scorer = fn or default_quality_score_fn

    async def _resolved(
        prediction: str,
        golden_plan: str | None,
        task_text: str,
    ) -> float:
        out = scorer(prediction, golden_plan, task_text)
        if inspect.isawaitable(out):
            out = await out  # type: ignore[assignment]
        return _normalize_score(out)

    return _resolved
