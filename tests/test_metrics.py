import pytest

from llm_eval.metrics import (
    context_recall,
    exact_match,
    faithfulness,
    is_refusal,
    rouge_l,
    token_f1,
)

CTX = ["The maximum total DTI is 45% for manually underwritten loans."]


def test_exact_match_normalises_case_and_punctuation():
    assert exact_match("The answer is 620.", "the answer is 620") == 1.0


def test_token_f1_partial_overlap():
    assert 0 < token_f1("max DTI is 45%", "The maximum total DTI is 45%") < 1
    assert token_f1("banana", "620 credit score") == 0.0


def test_rouge_l_identical_is_one():
    assert rouge_l("a b c d", "a b c d") == pytest.approx(1.0)


def test_faithfulness_grounded_vs_hallucinated():
    assert faithfulness("The maximum total DTI is 45% for manually underwritten loans.", CTX) == 1.0
    # right words, wrong number -> unsupported
    assert faithfulness("The maximum total DTI is 55% for manually underwritten loans.", CTX) == 0.0


def test_faithfulness_partial():
    ans = "The maximum DTI is 45% for manually underwritten loans. Pets are allowed in all condos."
    assert faithfulness(ans, CTX) == 0.5


def test_refusal_detection_and_faithfulness():
    assert is_refusal("I don't know - that is not covered in the provided documents.")
    assert not is_refusal("The minimum score is 620.")
    assert faithfulness("I don't know.", []) == 1.0


def test_context_recall():
    assert context_recall("DTI 45%", CTX) == 1.0
    assert context_recall("credit score 620", CTX) == 0.0
