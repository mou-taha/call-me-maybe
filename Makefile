PYTHON = python3
MODULE = src
SRC_DIR = src

all:run

install:
	uv sync

run:
	uv run $(PYTHON) -m $(MODULE)

debug:
	uv run python3 -m pdb -m $(MODULE)

clean:
	find . -type d \( -name "__pycache__" -o -name ".mypy_cache" \) -exec rm -rf {} +

lint:
	flake8 $(SRC_DIR)
	mypy $(SRC_DIR) --warn-return-any
	--warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs
	--check-untyped-defs