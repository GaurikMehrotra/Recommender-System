PYTHON ?= python
CONFIG ?= configs/debug_cpu.yaml
.PHONY: install install-cpu download test lint format debug tune final eda serve demo
install:
	$(PYTHON) -m pip install -r requirements.txt -e .
install-cpu:
	$(PYTHON) -m pip install --require-hashes -r requirements-cpu.lock --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu
	$(PYTHON) -m pip install --no-deps --no-build-isolation -e .
download:
	$(PYTHON) -m src.data.download --size 100k
test:
	$(PYTHON) -m pytest -q
lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .
format:
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .
debug:
	$(PYTHON) -m src.train --config $(CONFIG) --model mf --output runs/debug/mf
eda:
	$(PYTHON) -m src.report
tune:
	$(PYTHON) -m src.tune --config $(CONFIG) --stage tune
final:
	$(PYTHON) -m src.tune --config $(CONFIG) --stage final
serve:
	$(PYTHON) -m uvicorn src.api:create_app --factory --host 127.0.0.1 --port 8000
demo:
	$(PYTHON) -m app.app
