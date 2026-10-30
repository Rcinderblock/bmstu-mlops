.PHONY: install check check-release check-negative check-eval test test-legacy prefect release-init release-restore release release-reject api api-bench release-reevaluate up down
install:
	uv sync --frozen
check: check-release
check-release:
	uv run --locked python scripts/check_release.py
check-negative:
	uv run --locked python scripts/check_release.py --negative
check-eval:
	uv run --locked python -m unittest discover -s tests -p 'test_*.py' -v
test: check-release
test-legacy:
	uv run --locked python -m unittest discover -s tests/legacy -p 'test_*.py' -v
prefect:
	PREFECT_HOME=$(CURDIR)/.prefect PREFECT_SERVER_ANALYTICS_ENABLED=false uv run --locked prefect server start --host 127.0.0.1 --port 4208
release-init:
	uv run --locked python -m src.release_bridge --workspace deployments/planner
release-restore:
	uv run --locked python scripts/restore_release.py --workspace deployments/planner
release:
	uv run --locked python lab.py --workspace deployments/planner run --manifest manifest-baseline.json --report release-report-baseline.json
release-reject:
	uv run --locked python lab.py --workspace deployments/planner run --manifest manifest-candidate.json --report release-report-candidate.json
release-reevaluate:
	uv run --locked python scripts/reevaluate_release.py
api:
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --locked python service.py --workspace deployments/planner --port 8091
api-bench:
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --locked python scripts/benchmark_api.py --out notes/work/hw08-09-10/api-$(shell date +%Y%m%d-%H%M%S)
up:
	docker compose -p mlops-hw6 up -d --no-build --wait
down:
	docker compose -p mlops-hw6 down
