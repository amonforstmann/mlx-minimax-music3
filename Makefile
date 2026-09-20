.PHONY: check

check:
	uv run ruff check .
	uv run pytest -q tests/unit
	uv run pytest -q tests/integration
	uv run python dev/check_public_tree.py
	uv build --no-sources
