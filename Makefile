.PHONY: clean again lint ruff ty test

all: build

build:
	uv sync
	uv sync --group notebooks

clean:
	rm -rvf build dist vrlsnmr.egg-info src/vrlsnmr.egg-info
	rm -rvf vrlsnmr/__pycache__ test/__pycache__
	rm -vf vrlsnmr/_ops.so

again: clean build
	uv sync --reinstall-package vrlsnmr

lint: ruff ty

ruff:
	uv run ruff format --check
	uv run ruff check

ty:
	uv run ty check

test:
	uv run pytest
