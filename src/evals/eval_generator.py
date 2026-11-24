import json
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "2000")

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.models import OpenRouterModel
from deepeval.test_case import LLMTestCase


EVALUATION_PATH = (
    PROJECT_ROOT / "data" / "evaluation" / "eval_dataset" / "evaluation_dataset.json"
)
THRESHOLD = 0.7

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY is missing.")
if not EVALUATION_PATH.is_file():
    raise FileNotFoundError(f"Evaluation dataset not found: {EVALUATION_PATH}")

with EVALUATION_PATH.open(encoding="utf-8") as file:
    records = json.load(file)
if not isinstance(records, list) or not records:
    raise ValueError(f"Expected a non-empty list of test cases in {EVALUATION_PATH}")

test_cases = [
    LLMTestCase(
        input=record["input"],
        actual_output=record["actual_output"],
        expected_output=record["expected_output"],
        retrieval_context=record["retrieval_context"],
    )
    for record in records
]

judge_model = OpenRouterModel(
    model="openai/gpt-4o-mini",
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
    generation_kwargs={"max_tokens": 2048},
)
metrics = [
    FaithfulnessMetric(
        threshold=THRESHOLD,
        model=judge_model,
        include_reason=False,
        async_mode=False,
    ),
    AnswerRelevancyMetric(
        threshold=THRESHOLD,
        model=judge_model,
        include_reason=False,
        async_mode=False,
    ),
]

print(f"Evaluating {len(test_cases)} generated answers sequentially.")
evaluate(
    test_cases=test_cases,
    metrics=metrics,
    async_config=AsyncConfig(run_async=False, max_concurrent=1, throttle_value=3),
)
