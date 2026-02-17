.PHONY: test lint check all

test:
	uv run python -m unittest discover tests

lint:
	uv run ruff check .

mypy:
	uv run mypy .

check: lint mypy test

all: check
