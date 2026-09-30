.PHONY: install test lint eval eval-bad baseline serve

install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check . && ruff format --check .

eval:            ## offline baseline eval, gated
	llm-eval run --name extractive --baseline baselines/extractive.summary.json

eval-bad:        ## demo: a hallucinating system fails the gate
	llm-eval run --target hallucinating --name hallucinating --baseline baselines/extractive.summary.json

baseline:        ## accept current results as the new baseline
	llm-eval run --name extractive --baseline baselines/extractive.summary.json --update-baseline

serve:
	uvicorn llm_eval.api:app --port 8080
