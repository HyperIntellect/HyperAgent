# Development Guide

## Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | Next.js 16, React 18, TypeScript, Tailwind CSS, shadcn/ui |
| Backend | Python 3.11+, FastAPI, LangGraph |
| Database | PostgreSQL, Redis |
| Storage | Cloudflare R2 |
| Sandbox | E2B |

## Prerequisites

- Node.js 20+
- Python 3.11+
- Docker (optional, for PostgreSQL and Redis)

## Quick Start

### 1. Clone and Setup Environment

```bash
cd HyperAgent
cp backend/.env.example backend/.env
cp web/.env.example web/.env
# Edit backend/.env and add your API keys
```

### 2. Start Database Services (Docker)

```bash
docker-compose up -d postgres redis
```

Or run PostgreSQL and Redis locally.

### 3. Install Dependencies

```bash
make install
```

Or install separately:
```bash
make install-web  # Frontend dependencies
make install-backend  # Backend dependencies
```

### 4. Start Backend

```bash
make dev-backend
```

### 5. Start Frontend

```bash
make dev-web
```

Or start both in separate terminals:
```bash
make dev  # Shows instructions to run dev-web and dev-backend separately
```

### 6. Open Application

Visit [http://localhost:5000](http://localhost:5000)

## Project Structure

```
HyperAgent/
├── web/                    # Next.js frontend
│   ├── app/               # App router pages
│   ├── components/        # React components
│   │   ├── ui/           # shadcn/ui components
│   │   ├── chat/         # Chat components
│   │   ├── research/     # Research components
│   │   └── layout/       # Layout components
│   └── lib/              # Utilities, stores, types
│
├── backend/                # Python backend
│   ├── app/
│   │   ├── routers/      # API endpoints
│   │   ├── agents/       # LangGraph agents
│   │   │   └── tools/    # Tool implementations (incl. codeact.py)
│   │   ├── sandbox/      # Sandbox managers (unified, execution, app, desktop)
│   │   │   └── hyperagent_lib/  # Helper library for CodeAct mode
│   │   ├── services/     # Business logic (incl. snapshot_service.py)
│   │   └── models/       # Pydantic schemas
│   └── pyproject.toml
│
├── docker-compose.yml      # Docker services
└── .env.example           # Environment template
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | Health check |
| `/api/v1/chat/send` | POST | Send chat message |
| `/api/v1/chat/stream` | POST | Stream chat response |
| `/api/v1/research/start` | POST | Start research task |
| `/api/v1/research/stream/{id}` | GET | Stream research progress |

## Environment Variables

### Backend (`backend/.env`)

| Variable | Description | Required |
|----------|-------------|----------|
| `ANTHROPIC_API_KEY` | Anthropic API key | Yes* |
| `OPENAI_API_KEY` | OpenAI API key | Yes* |
| `GOOGLE_API_KEY` | Google Gemini API key | Yes* |
| `DATABASE_URL` | PostgreSQL connection | No |
| `REDIS_URL` | Redis connection | No |
| `E2B_API_KEY` | E2B sandbox API key | No |
| `SANDBOX_PROVIDER` | Sandbox provider (`e2b` or `boxlite`) | No |
| `SNAPSHOT_MAX_SIZE_BYTES` | Max snapshot size (default: 100MB) | No |
| `SNAPSHOT_RETENTION_HOURS` | Snapshot retention (default: 24h) | No |
| `STORAGE_BACKEND` | Storage backend (`r2` or `local`) | No |
| `CORS_ORIGINS` | Allowed origins | No |

*At least one LLM provider key is required.

### Web (`web/.env`)

| Variable | Description | Required |
|----------|-------------|----------|
| `NEXT_PUBLIC_API_URL` | Backend API URL | No |

## Development Commands

All commands use the Makefile. Run `make help` to see all available commands.

### Installation

```bash
make install          # Install all dependencies
make install-web      # Install frontend dependencies only
make install-backend  # Install backend dependencies only
```

### Development

```bash
make dev-web          # Start frontend dev server (port 5000)
make dev-backend          # Start backend dev server (port 8080)
make dev-worker       # Start background worker
make dev-all          # Start all services (frontend, backend, worker)
```

### Build

```bash
make build            # Build frontend for production
make build-web        # Build frontend only
```

### Linting & Formatting

```bash
make lint             # Run all linters
make lint-web         # Lint frontend code
make lint-backend     # Lint backend code
make format-backend   # Format backend code
```

### Testing

```bash
make test             # Run all tests
make test-backend     # Run backend tests
```

### Database Migrations

```bash
make migrate          # Apply all pending migrations
make migrate-down     # Rollback last migration
make migrate-new msg='description'  # Create new migration
make migrate-status   # Show migration status
```

### Utilities

```bash
make clean            # Clean build artifacts and caches
make health           # Check health of all services
```

### Job Queue Management

```bash
make queue-stats      # Show job queue statistics
make queue-monitor    # Monitor job queue in real-time
make queue-list       # List all queued jobs
make queue-clear      # Clear all jobs from queue (DESTRUCTIVE)
make queue-health     # Check worker and queue health
make queue-test       # Submit a test task to verify worker
```
