.PHONY: help install install-web install-backend dev dev-web dev-backend dev-worker dev-worker-watch dev-worker-burst dev-worker-high dev-all build build-web start-web lint lint-web lint-backend format-backend type-check test test-backend clean migrate migrate-down migrate-new migrate-status health queue-stats queue-monitor queue-list queue-clear queue-health queue-test eval eval-routing eval-tools eval-quality eval-langsmith sandbox-pull-images sandbox-build-app-image

# Colors
CYAN := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RESET := \033[0m

help: ## Show this help message
	@echo "$(CYAN)HyperAgent$(RESET) - AI Agent Platform"
	@echo ""
	@echo "$(GREEN)Usage:$(RESET)"
	@echo "  make $(YELLOW)<target>$(RESET)"
	@echo ""
	@echo "$(GREEN)Targets:$(RESET)"
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ { printf "  $(YELLOW)%-15s$(RESET) %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

# =============================================================================
# Installation
# =============================================================================

install: install-web install-backend ## Install all dependencies

install-web: ## Install frontend dependencies
	@echo "$(CYAN)Installing frontend dependencies...$(RESET)"
	cd web && npm install

install-backend: ## Install backend dependencies
	@echo "$(CYAN)Installing backend dependencies...$(RESET)"
	cd backend && uv sync --all-extras

# =============================================================================
# Development
# =============================================================================

dev: ## Start both frontend and backend (requires tmux or run in separate terminals)
	@echo "$(CYAN)Starting development servers...$(RESET)"
	@echo "$(YELLOW)Run 'make dev-web' and 'make dev-backend' in separate terminals$(RESET)"

dev-web: ## Start frontend development server
	@echo "$(CYAN)Starting frontend on http://localhost:5000$(RESET)"
	cd web && npm run dev -- -p 5000

dev-backend: ## Start backend development server
	@echo "$(CYAN)Starting backend on http://localhost:8080$(RESET)"
	cd backend && uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8080

dev-worker: ## Start background worker for async tasks
	@echo "$(CYAN)Starting background worker...$(RESET)"
	cd backend && uv run python worker.py

dev-worker-watch: ## Start background worker with auto-reload (development)
	@echo "$(CYAN)Starting background worker with auto-reload...$(RESET)"
	cd backend && uv run python worker.py --watch

dev-worker-burst: ## Process all queued jobs and exit (useful for testing)
	@echo "$(CYAN)Running worker in burst mode...$(RESET)"
	cd backend && uv run python worker.py --burst

dev-worker-high: ## Start worker with high concurrency (20 jobs)
	@echo "$(CYAN)Starting worker with high concurrency...$(RESET)"
	cd backend && uv run python worker.py --concurrency 20

dev-all: ## Start all services concurrently (frontend, backend, worker)
	@echo "$(CYAN)Starting all services...$(RESET)"
	@echo "$(YELLOW)Note: Ensure PostgreSQL and Redis are running$(RESET)"
	$(MAKE) dev-backend &
	$(MAKE) dev-worker &
	$(MAKE) dev-web

# =============================================================================
# Build
# =============================================================================

build: build-web ## Build all projects

build-web: ## Build frontend for production
	@echo "$(CYAN)Building frontend...$(RESET)"
	cd web && npm run build

start-web: ## Start production server (requires build first)
	@echo "$(CYAN)Starting production server on http://localhost:3000$(RESET)"
	cd web && npm start

# =============================================================================
# Linting & Testing
# =============================================================================

lint: lint-web lint-backend ## Run all linters

lint-web: ## Lint frontend code
	@echo "$(CYAN)Linting frontend...$(RESET)"
	cd web && npm run lint

type-check: ## Type-check frontend code
	@echo "$(CYAN)Type-checking frontend...$(RESET)"
	cd web && npx tsc --noEmit

lint-backend: ## Lint backend code
	@echo "$(CYAN)Linting backend...$(RESET)"
	cd backend && uv run ruff check . && uv run ruff format --check .

format-backend: ## Format backend code
	@echo "$(CYAN)Formatting backend...$(RESET)"
	cd backend && uv run ruff format .

test: test-backend ## Run all tests

test-backend: ## Run backend tests
	@echo "$(CYAN)Running backend tests...$(RESET)"
	cd backend && uv run pytest -v

# =============================================================================
# Agent Evaluations
# =============================================================================

eval: eval-routing eval-tools eval-quality ## Run all agent evaluations

eval-routing: ## Run routing accuracy evaluations
	@echo "$(CYAN)Running routing evaluations...$(RESET)"
	cd backend && uv run pytest evals/test_routing.py -v --tb=short

eval-tools: ## Run tool/skill selection evaluations
	@echo "$(CYAN)Running tool selection evaluations...$(RESET)"
	cd backend && uv run pytest evals/test_tool_selection.py -v --tb=short

eval-quality: ## Run response quality evaluations
	@echo "$(CYAN)Running response quality evaluations...$(RESET)"
	cd backend && uv run pytest evals/test_response_quality.py -v --tb=short

eval-langsmith: ## Run evals with LangSmith tracking
	@echo "$(CYAN)Running evaluations with LangSmith...$(RESET)"
	cd backend && LANGCHAIN_TRACING_V2=true uv run pytest evals/ -v -m langsmith

# =============================================================================
# Migrations
# =============================================================================

migrate: ## Apply all pending database migrations
	@echo "$(CYAN)Applying database migrations...$(RESET)"
	cd backend && uv run alembic upgrade head

migrate-down: ## Rollback last database migration
	@echo "$(CYAN)Rolling back last migration...$(RESET)"
	cd backend && uv run alembic downgrade -1

migrate-new: ## Create new migration (usage: make migrate-new msg='description')
	@if [ -z "$(msg)" ]; then \
		echo "$(YELLOW)Usage: make migrate-new msg='migration description'$(RESET)"; \
		exit 1; \
	fi
	@echo "$(CYAN)Creating new migration: $(msg)$(RESET)"
	cd backend && uv run alembic revision --autogenerate -m "$(msg)"

migrate-status: ## Show current migration status
	@echo "$(CYAN)Migration status:$(RESET)"
	cd backend && uv run alembic current
	@echo ""
	@echo "$(CYAN)Migration history:$(RESET)"
	cd backend && uv run alembic history

# =============================================================================
# Utilities
# =============================================================================

clean: ## Clean build artifacts and caches
	@echo "$(CYAN)Cleaning build artifacts...$(RESET)"
	rm -rf web/.next
	rm -rf web/node_modules/.cache
	rm -rf backend/__pycache__
	rm -rf backend/.pytest_cache
	rm -rf backend/.ruff_cache
	find backend -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

health: ## Check health of all services
	@echo "$(CYAN)Checking service health...$(RESET)"
	@echo -n "Backend API: "
	@curl -s http://localhost:8080/api/v1/health | jq -r '.status' 2>/dev/null || echo "$(YELLOW)Not running$(RESET)"
	@echo -n "Frontend: "
	@curl -s -o /dev/null -w "%{http_code}" http://localhost:5000 2>/dev/null && echo "$(GREEN)Running$(RESET)" || echo "$(YELLOW)Not running$(RESET)"

# =============================================================================
# Job Queue Management
# =============================================================================

queue-stats: ## Show job queue statistics
	@echo "$(CYAN)Job Queue Statistics:$(RESET)"
	@cd backend && uv run python -c "import asyncio; from app.workers.task_queue import task_queue; asyncio.run((lambda: task_queue.get_queue_stats())()).then(print)"

queue-monitor: ## Monitor job queue in real-time (requires redis-cli)
	@echo "$(CYAN)Monitoring job queue (Ctrl+C to stop)...$(RESET)"
	@redis-cli -u "${REDIS_URL:-redis://localhost:6379}" MONITOR

queue-list: ## List all queued jobs
	@echo "$(CYAN)Queued Jobs:$(RESET)"
	@redis-cli -u "${REDIS_URL:-redis://localhost:6379}" KEYS "arq:job:*" | head -20

queue-clear: ## Clear all jobs from queue (DESTRUCTIVE - use with caution)
	@echo "$(YELLOW)WARNING: This will delete all queued jobs!$(RESET)"
	@read -p "Are you sure? (yes/no): " confirm && [ "$$confirm" = "yes" ] || exit 1
	@echo "$(CYAN)Clearing job queue...$(RESET)"
	@redis-cli -u "${REDIS_URL:-redis://localhost:6379}" KEYS "arq:*" | xargs -r redis-cli -u "${REDIS_URL:-redis://localhost:6379}" DEL
	@echo "$(GREEN)Queue cleared$(RESET)"

queue-health: ## Check worker and queue health
	@echo "$(CYAN)Worker & Queue Health Check$(RESET)"
	@echo ""
	@echo "$(YELLOW)Worker Process:$(RESET)"
	@pgrep -f "worker.py" > /dev/null && echo "  $(GREEN)Running (PID: $$(pgrep -f 'worker.py'))$(RESET)" || echo "  $(YELLOW)Not running$(RESET)"
	@echo ""
	@echo "$(YELLOW)Redis Connection:$(RESET)"
	@redis-cli PING > /dev/null 2>&1 && echo "  $(GREEN)Connected$(RESET)" || echo "  $(YELLOW)Not connected$(RESET)"
	@echo ""
	@echo "$(YELLOW)Queue Length:$(RESET)"
	@echo "  $$(redis-cli LLEN arq:queue) jobs queued"
	@echo ""
	@echo "$(YELLOW)Recent ARQ Activity:$(RESET)"
	@redis-cli KEYS "arq:*" | wc -l | awk '{print "  " $$1, "ARQ keys in Redis"}'

queue-test: ## Submit a test task to verify worker is processing
	@echo "$(CYAN)Testing worker with a sample research task...$(RESET)"
	@cd backend && uv run python test_worker.py

# =============================================================================
# Sandbox Management
# =============================================================================

sandbox-pull-images: ## Pull all Docker images required by BoxLite local sandbox
	@echo "$(CYAN)Pulling BoxLite Docker images...$(RESET)"
	cd backend && python3 scripts/pull_boxlite_images.py

sandbox-build-app-image: ## Build the pre-cached app sandbox Docker image
	@echo "$(CYAN)Building app sandbox image with pre-cached templates...$(RESET)"
	docker build -f backend/docker/app-sandbox.Dockerfile -t hyperagent/app-sandbox:latest backend/docker/

# Default target
.DEFAULT_GOAL := help
