# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Documentation References

Refer to these docs for detailed guidelines:

- `docs/Development.md` — Setup, tech stack, project structure, and API reference
- `docs/Design-Style-Guide.md` — UI components, semantic color tokens, typography, and design patterns
- `docs/Agent-System-Design.md` — Multi-agent architecture, skills, guardrails, HITL
- `docs/Agent-Evals-Design.md` — Agent evaluation framework and testing

## Development Commands

```bash
# Installation
make install      # Install all dependencies (frontend + backend)
make install-web  # Install frontend dependencies
make install-backend  # Install backend dependencies

# Development
make dev-web      # Start Next.js dev server on http://localhost:5000
make dev-backend  # Start backend API server on http://localhost:8080
make dev-worker   # Start background worker for async tasks
make dev-worker-watch  # Start worker with auto-reload (development)
make dev-worker-burst  # Process all queued jobs and exit (useful for testing)
make dev-worker-high   # Start worker with high concurrency (20 jobs)
make dev-all      # Start all services concurrently (frontend, backend, worker)

# Build & Production
make build-web    # Build frontend for production
make start-web    # Start production server (requires build first)

# Code Quality
make lint         # Run all linters (frontend + backend)
make lint-web     # Run ESLint
make lint-backend # Lint backend code (ruff check + format check)
make format-backend # Format backend code
make type-check   # Type-check frontend without emitting files
make test         # Run all tests
make test-backend # Run backend tests

# Agent Evaluations
make eval           # Run all agent evaluations
make eval-routing   # Run routing accuracy evaluations
make eval-tools     # Run tool/skill selection evaluations
make eval-quality   # Run response quality evaluations
make eval-langsmith # Run evals with LangSmith tracking

# Database Migrations
make migrate           # Apply all pending database migrations
make migrate-down      # Rollback last database migration
make migrate-new msg='description'  # Create new migration
make migrate-status    # Show current migration status

# Utilities
make health       # Check health of all services
make clean        # Remove build artifacts and caches

# Job Queue Management
make queue-stats   # Show job queue statistics
make queue-monitor # Monitor job queue in real-time
make queue-list    # List all queued jobs
make queue-clear   # Clear all jobs from queue (DESTRUCTIVE)
make queue-health  # Check worker and queue health
make queue-test    # Submit a test task to verify worker
```

## Architecture Overview

### Frontend Framework
- **Next.js 16** (App Router with React 18)
- **Internationalization**: `next-intl` with locale prefix set to "never" (locales: en, zh-CN)
- **API Proxy**: `/api/v1/*` routes proxy to backend at `NEXT_PUBLIC_API_URL` (default: http://localhost:8080)

### State Management
Zustand stores with persistence to localStorage:

1. **Chat Store** (`lib/stores/chat-store.ts`) — Conversations, messages, streaming, API sync
2. **Task Store** (`lib/stores/task-store.ts`) — Research tasks with scenarios (academic, market, technical, news)
3. **Computer Store** (`lib/stores/computer-store.ts`) — Virtual computer panel: terminal, browser, file viewer, plan view (per-conversation state)
4. **Agent Progress Store** (`lib/stores/agent-progress-store.ts`) — Real-time agent event tracking, sources, stages
5. **Settings Store** (`lib/stores/settings-store.ts`) — Provider, tier, custom model selection
6. **Preview Store** (`lib/stores/preview-store.ts`) — File/slide preview panel
7. **Sidebar Store** (`lib/stores/sidebar-store.ts`) — Sidebar open/close state

**Important**: Chat and Task stores use `hasHydrated` flag. Always check hydration before rendering store-dependent UI to prevent SSR/client mismatches.

### Design System
- **Theme**: Warm stone (light) / Refined ink (dark) with auto mode following system preference
- **Typography**: DM Sans (display), JetBrains Mono (code) via `next/font/google`, system font fallbacks
- **Color System**: OKLCH-based CSS variables in `app/globals.css` for theme colors, HSL for semantic colors (success, warning, info)
- **Theme Hook**: `lib/hooks/use-theme.ts` returns `theme` (preference: auto/light/dark), `resolvedTheme` (actual: light/dark), and `setTheme`
- **Icons**: Lucide React
- **UI Components**: Radix UI primitives + custom components in `components/ui/`

### Component Structure

```
components/
├── layout/
│   ├── main-layout.tsx          # Root layout with sidebar + right panels
│   ├── desktop-sidebar.tsx      # Desktop sidebar wrapper
│   └── mobile-sidebar.tsx       # Mobile sidebar with swipe-to-close
├── ui/
│   ├── recent-tasks.tsx         # Unified timeline (conversations + tasks)
│   ├── preferences-panel.tsx    # Theme + language selector
│   ├── task-progress-panel.tsx  # Research task progress
│   ├── task-plan-panel.tsx      # Task planning steps with complexity
│   ├── app-preview-panel.tsx    # Inline web app preview (iframe)
│   ├── computer-viewer.tsx      # Computer panel embedding
│   └── ...                      # Radix-based primitives (button, input, select, etc.)
├── query/
│   └── chat-interface.tsx       # Main chat/research interface with SSE streaming
├── chat/
│   ├── message-bubble.tsx       # Message rendering (markdown, code, images, artifacts)
│   ├── chat-input.tsx           # Input with file upload, voice recording
│   └── file-upload-button.tsx   # Drag-and-drop file uploads
├── computer/
│   ├── virtual-computer-panel.tsx  # Right panel with terminal/browser/file/plan modes
│   ├── computer-terminal-view.tsx  # Shell output rendering
│   ├── computer-browser-view.tsx   # Sandbox browser stream viewer
│   └── ...                         # File viewer, plan view, playback controls
├── artifacts/
│   └── artifacts-preview-panel.tsx # File/slide preview with download
├── hitl/
│   └── interrupt-dialog.tsx     # HITL approval/decision/input dialogs
├── task/                        # Research task components
├── projects/                    # Project management UI
├── skills/                      # Skill cards and forms
└── auth/
    └── user-menu.tsx            # Authentication UI
```

### Key Patterns

**Sidebar Recent Items**:
- Combines conversations and tasks into unified timeline
- Groups by time: Today, Yesterday, This Week, Earlier
- Type indicator badges: Chat vs Research
- Status indicators for tasks: running (spinner), completed (check), failed (alert)
- Lazy rendering: initial 30 items, loads 20 more via IntersectionObserver

**Translations**:
- Messages in `messages/en.json` and `messages/zh-CN.json`
- Access via `useTranslations("namespace")` hook
- Common namespaces: sidebar, home, research, task, chat, agents, computer, preview

**Authentication**:
- NextAuth with Google OAuth
- Config in `lib/auth/config.ts`
- Environment variables: `NEXTAUTH_URL`, `NEXTAUTH_SECRET`, `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`

## Important Implementation Notes

### State Hydration
Always check `hasHydrated` before rendering store-dependent UI:
```tsx
const { hasHydrated } = useChatStore();
if (!hasHydrated) return null; // or loading state
```

### Theme System
- Theme preference stored as `theme-preference` in localStorage
- Auto mode listens to `prefers-color-scheme` media query
- Always use `theme` for UI display (shows user preference)
- Use `resolvedTheme` if you need actual light/dark value

### API Integration
- Backend assumed at `http://localhost:8080` (configurable via env)
- API routes proxied through Next.js rewrites
- No direct fetch to external API URLs from client

### Type Imports
- Main types in `lib/types/index.ts`
- Store types co-located with stores: `ResearchTask` in task-store, `Conversation` in chat-store
- Use `type` imports for type-only imports

### UI Component Guidelines
- Follow minimal, clean design philosophy
- Use existing color variables instead of hardcoding colors
- Prefer grid layouts for option groups (theme selector uses 3-col grid)
- Active states: inverted colors (foreground bg, background text)
- Icons should be 3.5-4px (w-3.5 h-3.5 or w-4 h-4)

## Backend Architecture

### Multi-Agent System

HyperAgent uses a **hybrid architecture** with two agents:

**Task Agent** (`backend/app/agents/subagents/task.py`):
- General-purpose ReAct loop with tool calling
- Handles ~80% of requests: chat, data analysis, app building, image generation, slide creation, browser automation
- Dedicated mode bypass: app/image/slide modes directly invoke corresponding skills without LLM routing
- Supports planned execution via `task_planning` skill with `plan_overview`/`plan_step_completed` events
- Todo file persistence: writes execution plan to `/home/user/.hyperagent/todo.md` in sandbox, injects as system context each iteration
- Anti-repetition: detects consecutive identical tool calls (MD5 hash), injects variation prompts after 3 repeats
- Optional CodeAct mode: `execute_script` tool with `hyperagent` helper library (gated behind `execution_mode: "codeact"`)

**Research Agent** (`backend/app/agents/subagents/research.py`):
- Multi-step research: search → analyze → synthesize → write report
- Used for deep research requiring 10+ sources

**Supervisor** (`backend/app/agents/supervisor.py`):
- Routes queries to Task or Research agent
- Manages agent handoff (Task → Research, max 3 per request)
- Handles event streaming and state management

**Skills System** (7 builtin skills):
- `image_generation` — AI image generation (Gemini/DALL-E)
- `code_generation` — Generate code snippets
- `web_research` — Focused web research with summarization
- `data_analysis` — Full data analysis with planning, code execution, and summarization
- `slide_generation` — Create PPTX presentations
- `app_builder` — Build web apps with live preview (planning uses MAX tier, codegen uses PRO)
- `task_planning` — Analyze complex tasks and create execution plans

Skills are LangGraph subgraphs invoked as tools using `invoke_skill` and `list_skills`.

**LLM Providers:**
- Built-in: Anthropic, OpenAI, Gemini
- Custom: Any OpenAI-compatible API (DeepSeek, Kimi, Qwen, MiniMax, Ollama, etc.) via `CUSTOM_PROVIDERS` env var
- Three-tier model routing: MAX (best quality), PRO (balanced), FLASH (fast/cheap)
- Per-tier provider overrides: `MAX_MODEL_PROVIDER`, `PRO_MODEL_PROVIDER`, `FLASH_MODEL_PROVIDER`
- Thinking mode: `ThinkingAwareChatOpenAI` in `backend/app/ai/thinking.py` handles `reasoning_content` for thinking-mode providers

**Sandbox Providers:**
- **E2B** — Cloud sandboxes for code execution, browser automation, app hosting (requires `E2B_API_KEY`)
- **BoxLite** — Local Docker-based sandboxes (requires Docker)
- Configured via `SANDBOX_PROVIDER` env var (`e2b` or `boxlite`)
- **Unified Sandbox Manager** (`backend/app/sandbox/unified_sandbox_manager.py`) — Single shared `SandboxRuntime` per user+task for code execution and app development (desktop stays separate). Session key: `unified:{user_id}:{task_id}`
- **Persistent Snapshots** (`backend/app/services/snapshot_service.py`) — Workspace snapshots saved to R2/local storage on disconnect, restored on reconnect. Config: `SNAPSHOT_MAX_SIZE_BYTES` (100MB), `SNAPSHOT_RETENTION_HOURS` (24h)
- **Handoff Artifacts** (`backend/app/sandbox/artifact_transfer.py`) — Files transferred between sandboxes during agent handoffs (max 10 files / 50MB)

**Context Compression:**
- LLM-based summarization for long conversations
- Preserves recent messages (configurable, default: 10)
- Triggers automatically at token threshold (default: 60k)
- Falls back to truncation if compression fails
- KV-cache-friendly prompt construction: stable prefix (system prompt + tool schemas) vs dynamic suffix, `prefix_hash` tracking for cache optimization
- Soft-disable tools via system message instead of removing schemas to preserve prefix stability

**Safety Guardrails:**
Comprehensive safety scanning at multiple integration points:
- **Input Scanner** — Prompt injection, jailbreak detection (`backend/app/guardrails/scanners/input_scanner.py`)
- **Output Scanner** — Toxicity, PII, harmful content (`backend/app/guardrails/scanners/output_scanner.py`)
- **Tool Scanner** — URL validation, code safety (`backend/app/guardrails/scanners/tool_scanner.py`)

Configuration via environment variables:
- `GUARDRAILS_ENABLED` — Master toggle (default: true)
- `GUARDRAILS_VIOLATION_ACTION` — block, warn, or log (default: block)

**Human-in-the-Loop (HITL):**
- Redis pub/sub for real-time interrupt lifecycle
- Three types: APPROVAL, DECISION, INPUT
- `ask_user` tool available to all agents
- Configurable timeouts (120s approval, 300s decision)

See `docs/Agent-System-Design.md` for detailed architecture documentation.

### API Endpoints

**Core:**
- `POST /api/v1/query` — Main query endpoint (SSE streaming)
- `POST /api/v1/query/stream` — Streaming query endpoint

**Conversations:**
- `GET /api/v1/conversations` — List conversations
- `POST /api/v1/conversations` — Create conversation
- `GET /api/v1/conversations/:id` — Get conversation details
- `POST /api/v1/conversations/:id/generate-title` — Generate title

**Tasks:**
- `GET /api/v1/tasks` — List research tasks
- `POST /api/v1/tasks` — Create research task
- `GET /api/v1/tasks/:id` — Get task status

**Skills:**
- `GET /api/v1/skills` — List available skills
- `GET /api/v1/skills/:id` — Get skill details

**Files:**
- `POST /api/v1/files/upload` — Upload files
- `GET /api/v1/files/download/:key` — Download files
- `GET /api/v1/files/generated/:path` — Get generated files

**Sandbox:**
- `GET /api/v1/sandbox/app/:sandbox_id/*` — App preview proxy
- `GET /api/v1/sandbox/files/content` — File content from sandbox

**HITL:**
- `POST /api/v1/hitl/respond/:threadId` — Submit interrupt response
- `GET /api/v1/hitl/pending/:threadId` — Get pending interrupts

**Providers:**
- `GET /api/v1/providers` — List available LLM providers

**Health:**
- `GET /api/v1/health` — Health check

Ensure backend is running on port 8080 or configure `NEXT_PUBLIC_API_URL`.
