.PHONY: test lint check all

test:
	uv run pytest tests

lint:
	uv run ruff check .

mypy:
	uv run mypy src

check: lint mypy test

play:
	uv run hex-play

train:
	uv run hex-train

all: check
