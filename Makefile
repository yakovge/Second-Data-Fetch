# Data Fetch Abstraction System Makefile
# This Makefile provides convenient commands for development and deployment

.PHONY: help install test lint format docker-build docker-test docker-dev clean

# Default target
help:
	@echo "📋 Available commands:"
	@echo ""
	@echo "🛠️  Development:"
	@echo "  install      Install dependencies in virtual environment"
	@echo "  test         Run all tests"
	@echo "  test-unit    Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  lint         Run linting checks"
	@echo "  format       Format code with black"
	@echo "  clean        Clean up temporary files"
	@echo ""
	@echo "🐳 Docker:"
	@echo "  docker-build Build Docker images"
	@echo "  docker-test  Run tests in Docker"
	@echo "  docker-dev   Start development environment"
	@echo "  docker-demo  Run demo in Docker"
	@echo "  docker-clean Clean Docker images and volumes"
	@echo ""
	@echo "📊 Monitoring:"
	@echo "  docker-logs  Show application logs"
	@echo "  docker-health Check service health"
	@echo ""
	@echo "🚀 Production:"
	@echo "  docker-prod  Start production environment"
	@echo "  docker-scale Scale services"

# Python environment setup
VENV_PATH = venv
PYTHON = $(VENV_PATH)/bin/python
PIP = $(VENV_PATH)/bin/pip
PYTEST = $(VENV_PATH)/bin/pytest

# Create virtual environment and install dependencies
install:
	@echo "🔧 Setting up virtual environment..."
	python -m venv $(VENV_PATH)
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt
	@echo "✅ Installation complete!"

# Testing
test: install
	@echo "🧪 Running all tests..."
	$(PYTEST) tests/ -v --cov=src --cov-report=term-missing

test-unit: install
	@echo "🧪 Running unit tests..."
	$(PYTEST) tests/ -m unit -v

test-integration: install
	@echo "🧪 Running integration tests..."
	$(PYTEST) tests/ -m integration -v

test-watch: install
	@echo "👀 Running tests in watch mode..."
	$(PYTEST) tests/ -v --cov=src -f

# Code quality
lint: install
	@echo "🔍 Running linting checks..."
	$(VENV_PATH)/bin/flake8 src/ tests/
	$(VENV_PATH)/bin/mypy src/
	@echo "✅ Linting complete!"

format: install
	@echo "🎨 Formatting code..."
	$(VENV_PATH)/bin/black src/ tests/
	@echo "✅ Formatting complete!"

# Docker commands
docker-build:
	@echo "🐳 Building Docker images..."
	docker-compose build

docker-test:
	@echo "🧪 Running tests in Docker..."
	docker-compose run --rm datafetch test

docker-test-unit:
	@echo "🧪 Running unit tests in Docker..."
	docker-compose run --rm datafetch test-unit

docker-test-integration:
	@echo "🧪 Running integration tests in Docker..."
	docker-compose up -d redis localstack
	@echo "⏳ Waiting for services to be ready..."
	sleep 10
	docker-compose run --rm datafetch test-integration
	docker-compose down

docker-dev:
	@echo "🛠️ Starting development environment..."
	docker-compose --profile dev up -d
	@echo "✅ Development environment started!"
	@echo "📝 Access Jupyter at: http://localhost:8888"

docker-demo:
	@echo "📋 Running demo in Docker..."
	docker-compose up -d redis localstack
	sleep 5
	docker-compose run --rm datafetch demo
	docker-compose down

docker-prod:
	@echo "🚀 Starting production environment..."
	docker-compose up -d datafetch redis
	@echo "✅ Production environment started!"

# Service management
docker-up:
	@echo "⬆️ Starting all services..."
	docker-compose up -d

docker-down:
	@echo "⬇️ Stopping all services..."
	docker-compose down

docker-restart:
	@echo "🔄 Restarting services..."
	docker-compose restart

docker-logs:
	@echo "📋 Showing application logs..."
	docker-compose logs -f datafetch

docker-logs-all:
	@echo "📋 Showing all service logs..."
	docker-compose logs -f

docker-health:
	@echo "🏥 Checking service health..."
	docker-compose ps
	@echo ""
	@echo "🔍 Health check details:"
	docker-compose exec datafetch health || true
	docker-compose exec redis redis-cli ping || true
	docker-compose exec localstack curl -f http://localhost:4566/_localstack/health || true

# Scaling
docker-scale:
	@echo "📈 Scaling services..."
	docker-compose up -d --scale datafetch=3

# Jupyter notebook
jupyter:
	@echo "📓 Starting Jupyter notebook..."
	docker-compose --profile notebook up jupyter

# Database operations
redis-cli:
	@echo "💾 Connecting to Redis..."
	docker-compose exec redis redis-cli

redis-flush:
	@echo "🧹 Flushing Redis cache..."
	docker-compose exec redis redis-cli FLUSHALL

# S3 operations (LocalStack)
s3-create-bucket:
	@echo "📦 Creating S3 bucket..."
	docker-compose exec localstack awslocal s3 mb s3://test-datafetch-bucket

s3-list:
	@echo "📋 Listing S3 contents..."
	docker-compose exec localstack awslocal s3 ls s3://test-datafetch-bucket --recursive

# Cleanup
clean:
	@echo "🧹 Cleaning up..."
	rm -rf $(VENV_PATH)
	rm -rf __pycache__ .pytest_cache .coverage htmlcov
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -name "*.pyc" -delete
	@echo "✅ Cleanup complete!"

docker-clean:
	@echo "🐳 Cleaning Docker resources..."
	docker-compose down -v
	docker system prune -f
	docker volume prune -f
	@echo "✅ Docker cleanup complete!"

docker-clean-all: docker-clean
	@echo "🧹 Removing Docker images..."
	docker rmi $(shell docker images -q --filter "reference=second-data-fetch*") 2>/dev/null || true

# Development helpers
dev-setup: install
	@echo "🛠️ Setting up development environment..."
	$(PIP) install pre-commit ipython jupyter
	$(VENV_PATH)/bin/pre-commit install || true
	mkdir -p logs temp notebooks
	@echo "✅ Development setup complete!"

dev-shell:
	@echo "🐚 Starting development shell..."
	docker-compose run --rm datafetch-dev shell

dev-python:
	@echo "🐍 Starting Python shell..."
	docker-compose run --rm datafetch-dev python

# Monitoring and debugging
monitor:
	@echo "📊 Starting monitoring dashboard..."
	@echo "Redis: docker-compose exec redis redis-cli monitor"
	@echo "Logs: docker-compose logs -f"
	@echo "Stats: watch docker-compose ps"

debug:
	@echo "🐛 Debug mode - starting with shell access..."
	docker-compose run --rm --entrypoint bash datafetch-dev

# Documentation
docs-serve: install
	@echo "📖 Serving documentation..."
	# Add documentation serving command when implemented

# CI/CD helpers
ci-test:
	@echo "🤖 Running CI tests..."
	docker-compose --profile test run --rm test-runner

ci-build:
	@echo "🤖 CI build process..."
	docker-compose build --no-cache

# Security
security-scan:
	@echo "🔒 Running security scan..."
	$(PIP) install safety bandit || true
	$(VENV_PATH)/bin/safety check || true
	$(VENV_PATH)/bin/bandit -r src/ || true

# Performance testing
perf-test:
	@echo "⚡ Running performance tests..."
	docker-compose run --rm datafetch test -m "not slow"

# Version management
version:
	@echo "📌 Current version info:"
	@echo "Python: $(shell python --version)"
	@echo "Docker: $(shell docker --version)"
	@echo "Docker Compose: $(shell docker-compose --version)"

# Environment validation
validate-env:
	@echo "✅ Validating environment..."
	@echo "Required tools:"
	@which python3 > /dev/null || echo "❌ Python 3 not found"
	@which docker > /dev/null || echo "❌ Docker not found"  
	@which docker-compose > /dev/null || echo "❌ Docker Compose not found"
	@echo "Environment variables:"
	@echo "GEMINI_API_KEY: $(if $(GEMINI_API_KEY),✅ Set,⚠️  Not set)"
	@echo "AWS_ACCESS_KEY_ID: $(if $(AWS_ACCESS_KEY_ID),✅ Set,⚠️  Not set)"

# Quick start
quickstart: docker-build docker-demo
	@echo ""
	@echo "🎉 Quickstart complete!"
	@echo "💡 Try these commands:"
	@echo "  make docker-dev    # Start development environment"
	@echo "  make docker-test   # Run tests"
	@echo "  make jupyter       # Start Jupyter notebook"