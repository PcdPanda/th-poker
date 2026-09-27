# Modern Python Template Makefile
# This Makefile provides common development tasks for the PyTemplate project

.PHONY: help setup-dev install install-dev test test-cov lint format clean build doc serve-doc distclean

# Default target
help: ## Show this help message
	@echo "PyTemplate - Modern Python Template"
	@echo "=================================="
	@echo ""
	@echo "Available commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Variables
PYTHON ?= python3
PIP ?= pip
PYTEST ?= pytest
PYTHONPATH := $(shell pwd)/src/python:$(PYTHONPATH)

# Development setup
setup-dev: ## Install development dependencies
	$(PIP) install -r requirements.txt

# Installation
install: ## Install the package
	$(PIP) install .

install-dev: ## Install the package in development mode
	$(PIP) install -e .

# Building
build: ## Build the package and extensions
	$(PYTHON) setup.py build_ext --inplace

build-cpp: ## Build C++ components
	mkdir -p src/cpp/build
	cd src/cpp/build && cmake .. -DCMAKE_BUILD_TYPE=Release
	cd src/cpp/build && cmake --build . --config Release

# Testing
test: ## Run tests
	PYTHONPATH=$(PYTHONPATH) $(PYTEST) src/python/pytemplate/tests/ tests/

test-cov: ## Run tests with coverage
	PYTHONPATH=$(PYTHONPATH) $(PYTEST) --cov=pytemplate --cov-report=term-missing src/python/pytemplate/tests/ tests/

# Code quality
lint: ## Run linting checks
	ruff check .
	black --check --diff .

format: ## Format code
	black .
	ruff format .
	ruff check --fix .

# Documentation
doc: ## Build documentation
	cd doc && make html

serve-doc: doc ## Build and serve documentation locally
	cd doc/build/html && python -m http.server 8000

# Cleaning
clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.so" -delete
	find . -type f -name "*.cpp" -delete

clean-cpp: ## Clean C++ build artifacts
	rm -rf src/cpp/build/
	rm -rf src/python/cpp/lib/

distclean: clean clean-cpp ## Clean all generated files
	rm -rf .venv/
	rm -rf venv/
	rm -rf .tox/
	rm -rf doc/build/

# Development utilities
check: lint test ## Run all checks (lint, test)

# Legacy compatibility
setup: setup-dev ## Alias for setup-dev
install-py: install ## Alias for install
install-py-dev: install-dev ## Alias for install-dev