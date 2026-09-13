.PHONY: install generate bench inspect check check-hw1 test clean

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
