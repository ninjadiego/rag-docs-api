.PHONY: dev test lint fmt docker-up

dev: ## Run with auto-reload
	uvicorn app.main:app --reload --port 8000

test: ## Run tests (no network, no model download)
	pytest

lint:
	ruff check . && ruff format --check .

fmt:
	ruff format . && ruff check --fix .

docker-up:
	docker compose up --build -d
