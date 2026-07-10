install:
	pip install -e .[dev]

generate:
	python scripts/generate_data.py

train:
	python scripts/train.py

test:
	pytest

lint:
	ruff check .

demo:
	python scripts/run_demo.py
