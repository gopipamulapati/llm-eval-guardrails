from pathlib import Path

import pytest

from llm_eval.runner import load_dataset
from llm_eval.targets import load_kb

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def kb():
    return load_kb(ROOT / "data" / "kb.jsonl")


@pytest.fixture
def cases():
    return load_dataset(ROOT / "data" / "golden.jsonl")
