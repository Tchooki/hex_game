.PHONY: test lint check all profile profile-data

test:
	uv run pytest tests

lint:
	uv run ruff check . --config ruff.toml

mypy:
	uv run mypy src

check: lint mypy test

play:
	uv run hex-play

train:
	uv run hex-train

profile:
	uv run python scripts/profile_mcts.py
	uv run snakeviz mcts_multi_profile.prof

profile-data:
	uv run python scripts/profile_generate_data.py
	uv run snakeviz generate_data_profile.prof

all: check
