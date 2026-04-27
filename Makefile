.PHONY: install train serve test compose-up compose-down

PYTHON ?= python3
PYTHONPATH := .

install:
	pip install -r requirements.txt

train:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m src.train.train data/seed_labeled.csv --register

serve:
	PYTHONPATH=$(PYTHONPATH) uvicorn src.serve.app:app --reload --host 0.0.0.0 --port 8000

test:
	PYTHONPATH=$(PYTHONPATH) pytest -q

compose-up:
	docker compose -f deploy/docker-compose.yml up -d

compose-down:
	docker compose -f deploy/docker-compose.yml down -v

drift-report:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m src.monitor.run_evidently
