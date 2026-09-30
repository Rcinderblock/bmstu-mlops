.PHONY: tokenize check-hw3 check-hw4 install generate bench inspect check check-hw1 check-hw2 data fetch-data diff test clean train train-all train-freeze compare plot distclean

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

check-hw3:
	bash tests/check_hw3.sh

check-hw4:
	bash tests/check_hw4.sh

train:
	uv run --locked dvc repro train_all train_freeze curves

train-all:
	uv run --locked dvc repro train_all

train-freeze:
	uv run --locked dvc repro train_freeze

plot:
	uv run --locked dvc repro curves

compare:
	uv run --locked dvc repro compare

tokenize:
	uv run --locked dvc repro tokenize

test:
	uv run --locked python -m unittest discover -s tests -p 'test_*.py' -v

clean:
	rm -rf models metrics/train_all_layers.json metrics/train_freeze14.json metrics/compare_all_layers.json docs/curves.png docs/compare.md docs/bench.json out1.txt out2.txt params.yaml.bak

distclean: clean
	rm -rf .venv

check-hw2:
	bash tests/check_hw2.sh

fetch-data:
	uv run --locked python scripts/fetch_source.py

data:
	uv run --locked dvc repro tokenize

diff:
	uv run --locked dvc metrics diff hw03-v1 hw03-v2 --targets metrics/clean.json
