.PHONY: install dev test lint run docker-build docker-run clean

# Install production deps
install:
	pip install -e .

# Install with dev deps
dev:
	pip install -e ".[dev]"

# Run tests
test:
	python -m pytest tests/ -v --tb=short

# Run specific test file
test-pruner:
	python -m pytest tests/test_pruner.py -v

test-cb:
	python -m pytest tests/test_circuit_breaker.py -v

# Lint
lint:
	ruff check ctxpact/ tests/
	ruff format --check ctxpact/ tests/

# Format
fmt:
	ruff format ctxpact/ tests/
	ruff check --fix ctxpact/ tests/

# Type check
typecheck:
	mypy ctxpact/

# Run the server
run:
	python -m ctxpact.server

# Run with custom config
run-config:
	python -m ctxpact.server --config config.yaml

# Docker
docker-build:
	docker build -t ctxpact:latest .

docker-run:
	docker run -p 8000:8000 -v $(PWD)/config.yaml:/app/config.yaml ctxpact:latest

# Quick health check
ping:
	curl -s http://localhost:8000/health | python -m json.tool

# List models through the proxy
models:
	curl -s http://localhost:8000/v1/models | python -m json.tool

# Test chat completion
chat-test:
	curl -s http://localhost:8000/v1/chat/completions \
		-H "Content-Type: application/json" \
		-H "X-Session-ID: test-session-001" \
		-d '{"model": "Nanbeige/Nanbeige4.1-3B", "messages": [{"role": "user", "content": "Hello, how are you?"}]}' \
		| python -m json.tool

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache dist/ build/ *.egg-info
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.pyc" -delete
