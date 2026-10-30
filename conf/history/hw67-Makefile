.PHONY: install up down data fetch-data sweep train report candidate baseline-init evaluate eval check-eval check-negative promote check-storage check check-all baseline-update predict-candidate test test-legacy submission fetch-eval
install:
	uv sync --frozen
up:
	docker compose -p mlops-hw6 up --build -d --wait
down:
	docker compose -p mlops-hw6 down
fetch-data:
	uv run --locked python scripts/fetch_source.py
data:
	uv run --locked dvc repro tokenize
sweep:
	uv run --locked python -m src.sweep
train: sweep
report:
	uv run --locked python -m src.experiment_report
candidate:
	uv run --locked python -m evaluation.cli candidate
baseline-init:
	uv run --locked python -m evaluation.cli baseline-init
evaluate:
	uv run --locked python -m evaluation.cli evaluate
eval:
	uv run --locked python -m evaluation.cli gate
check-eval:
	uv run --locked python -m unittest discover -s tests -p 'test_*.py' -v
check-negative:
	uv run --locked python -m evaluation.cli negative
promote:
	uv run --locked python -m evaluation.cli promote
check-storage:
	uv run --locked python tests/check_storage.py
check:
	bash tests/check.sh
check-all: check check-eval
baseline-update:
	uv run --locked python -m evaluation.cli baseline-update
predict-candidate:
	uv run --locked python -m src.predict --uri models:/planner-qwen-adapter@candidate --prompt "покажи список включённых будильников"
test: check-eval

test-legacy:
	uv run --locked python -m unittest discover -s tests/legacy -p 'test_*.py' -v

fetch-eval:
	uv run --locked python scripts/fetch_eval_data.py
submission:
	uv run --locked python scripts/export_hw67.py
