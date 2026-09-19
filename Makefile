.PHONY: install generate bench inspect check check-hw1 check-hw2 data fetch-data diff test clean

install:
	uv sync --locked

generate:
	uv run --locked python -m src.generate

bench:
	uv run --locked python -m src.bench

inspect:
	uv run --locked python -m src.inspect_model

check-hw1:
	bash tests/check_hw1.sh

check:
	bash tests/check.sh

test:
	uv run --locked python -m unittest discover -s tests -p 'test_*.py' -v

clean:
	rm -rf docs/bench.json out1.txt out2.txt params.yaml.bak

check-hw2:
	bash tests/check_hw2.sh

fetch-data:
	uv run --locked python scripts/fetch_source.py

data:
	uv run --locked dvc repro

diff:
	uv run --locked dvc metrics diff hw03-v1 hw03-v2
