.PHONY: install lint test train eda serve bench docker-up docker-down business-case

export PYTHONPATH := src

install:        ## dependências de desenvolvimento
	pip install -r requirements-dev.txt

lint:           ## ruff (lint + formatação)
	ruff check src tests scripts && ruff format --check src tests scripts

test:           ## testes unitários + integração com gate de cobertura
	pytest --cov=fraudguard --cov-report=term-missing

train:          ## treina (usa data/raw/creditcard.csv se existir)
	python -m fraudguard.modeling.train

eda:            ## relatório de EDA em reports/EDA.md
	python -m fraudguard.analysis.eda

business-case:  ## projeção financeira para o volume da plataforma
	python scripts/business_case.py --tx-per-minute 3000

serve:          ## API local em http://localhost:8000
	uvicorn fraudguard.api.main:app --host 0.0.0.0 --port 8000

bench:          ## teste de carga contra a API em execução
	python scripts/benchmark.py --requests 5000 --concurrency 32

docker-up:
	docker compose up --build -d && docker compose ps

docker-down:
	docker compose down
