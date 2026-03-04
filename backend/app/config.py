import json
from functools import lru_cache
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_ignore_empty=True,
    )

    # App
    app_name: str = "HyperAgent API"
    debug: bool = False
    environment: Literal["development", "staging", "production"] = "development"

    # API
    api_prefix: str = "/api/v1"
    cors_origins: str = "http://localhost:5000"

    # LLM Providers
    default_provider: str = "anthropic"
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    gemini_api_key: str = ""

    # Per-tier provider override (blank = use default_provider)
    max_model_provider: str = ""
    pro_model_provider: str = ""
    lite_model_provider: str = ""

    # Vision provider override (blank = use default_provider)
    vision_model_provider: str = ""

    # Image generation provider override (blank = use default_provider)
    image_model_provider: str = ""

    # Custom OpenAI-Compatible Providers (JSON array)
    custom_providers: str = ""

    # GCP / Vertex AI (use Vertex AI instead of Google AI Studio for Gemini)
    gemini_use_vertex_ai: bool = False
    gcp_project_id: str = ""
    gcp_location: str = "us-central1"

    # Model Tier Configuration
    # Models for each tier and provider combination
    tier_max_anthropic: str = "claude-opus-4-20250514"
    tier_max_openai: str = "gpt-4o"
    tier_max_gemini: str = "gemini-2.5-pro"

    tier_pro_anthropic: str = "claude-sonnet-4-20250514"
    tier_pro_openai: str = "gpt-4o-mini"
    tier_pro_gemini: str = "gemini-2.5-flash"

    tier_lite_anthropic: str = "claude-haiku-4-5-20251001"
    tier_lite_openai: str = "gpt-4o-mini"
    tier_lite_gemini: str = "gemini-2.0-flash"

    # Multimodal Model Configuration
    # Image Understanding (Vision) - per-provider models
    vision_model_gemini: str = "gemini-2.5-flash"
    vision_model_openai: str = "gpt-4o"

    # Image Generation - per-provider models
    image_gen_model_gemini: str = "gemini-3-pro-image-preview"
    image_gen_model_openai: str = "dall-e-3"
    image_gen_safety_filter: str = "block_some"  # block_none, block_some, block_most
    image_gen_openai_quality: str = "standard"  # standard or hd

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/hyperagent"
    redis_url: str = "redis://localhost:6379"

    # Storage
    storage_backend: Literal["r2", "local"] = "local"  # r2 for production, local for dev
    r2_access_key_id: str = ""
    r2_secret_access_key: str = ""
    r2_bucket_name: str = "hyperagent"
    r2_endpoint_url: str = ""
    local_storage_path: str = "./uploads"  # For local development

    # LLM Client Configuration
    llm_request_timeout: int = 120  # Timeout in seconds for LLM API requests
    llm_max_retries: int = 2  # Max retries for transient LLM failures

    # LangGraph Configuration
    langgraph_recursion_limit: int = (
        50  # Maximum recursion depth for graph execution (increased for handoff-heavy workflows)
    )
    subgraph_timeout: int = 300  # Timeout in seconds for subgraph invocations (5 minutes)
    routing_timeout: int = 30  # Timeout in seconds for routing decisions
    research_task_timeout: int = 1800  # Timeout in seconds for research tasks (30 minutes)

    # Context Compression
    context_compression_enabled: bool = True  # Enable LLM-based context compression
    context_compression_token_threshold: int = (
        60000  # Token threshold to trigger compression (60% of 100k budget)
    )
    context_compression_preserve_recent: int = 10  # Number of recent messages to always preserve

    # Sandbox Provider
    sandbox_provider: Literal["e2b", "boxlite"] = "e2b"

    # E2B Code Sandbox
    e2b_api_key: str = ""
    e2b_template_id: str = (
        ""  # Optional: custom template with pre-installed packages for faster startup
    )
    e2b_code_timeout: int = 300  # 5 minutes for code execution sandbox
    e2b_session_timeout_minutes: int = 10  # Session timeout for code sandbox manager

    # E2B Desktop Sandbox
    e2b_desktop_timeout: int = 900  # 15 minutes (longer for browser startup)
    e2b_desktop_default_browser: str = "google-chrome"
    e2b_desktop_session_timeout_minutes: int = 15  # Session timeout for browser sandbox manager
    e2b_desktop_stream_ready_wait_ms: int = 3000  # Wait time for stream to be ready before actions

    # BoxLite Local Sandbox
    boxlite_code_image: str = "python:3.12-slim"
    boxlite_desktop_image: str = "boxlite/desktop:latest"
    boxlite_app_image: str = "hyperagent/app-sandbox:latest"
    boxlite_cpus: int = 2
    boxlite_memory_mib: int = 1024
    boxlite_disk_size_gb: int = 4  # Disk size in GB (needed for npm/node scaffolding)
    boxlite_working_dir: str = "/home/user"
    boxlite_code_timeout: int = 300
    boxlite_desktop_timeout: int = 900
    boxlite_desktop_default_browser: str = "chromium-browser"
    boxlite_auto_remove: bool = True
    boxlite_app_host_port_start: int = 10000

    # Sandbox Snapshots
    snapshot_max_size_bytes: int = 100 * 1024 * 1024  # 100MB max per snapshot
    snapshot_retention_hours: int = 24  # Default retention period
    snapshot_default_paths_execution: list[str] = ["/home/user", "/tmp/outputs"]
    snapshot_default_paths_app: list[str] = ["/home/user/app"]

    # Search
    tavily_api_key: str = ""

    # Additional search providers
    serper_api_key: str = ""  # Google Search via Serper API
    jina_api_key: str = ""    # Jina Reader for deep content extraction

    # Web research (agentic search) tuning
    web_research_max_refinements: int = 3
    web_research_confidence_threshold: float = 0.7
    web_research_max_sub_queries: int = 6

    # Rate Limiting
    rate_limit_enabled: bool = True
    rate_limit_rpm: int = 30  # requests per minute

    # Human-in-the-Loop (HITL) Settings
    hitl_enabled: bool = True  # Enable HITL by default
    hitl_approval_timeout: int = 120  # 2 minutes for tool approvals
    hitl_decision_timeout: int = 300  # 5 minutes for decision interrupts
    hitl_default_risk_threshold: str = "high"  # high, medium, or all

    # Logging
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "console"

    # Guardrails
    guardrails_enabled: bool = True
    guardrails_input_enabled: bool = True
    guardrails_output_enabled: bool = True
    guardrails_tool_enabled: bool = True
    guardrails_violation_action: Literal["block", "warn", "log"] = "block"
    guardrails_timeout_ms: int = 500

    # ReAct Loop Configuration
    react_max_iterations: int = 5  # Maximum number of tool-calling iterations

    # Feature flags for progressive rollout
    policy_engine_v1: bool = True
    planner_verifier_v1: bool = True
    parallel_executor_v1: bool = True
    run_ledger_v1: bool = True
    parallel_executor_max_agents: int = 4
    routing_mode: Literal["deterministic", "llm"] = "deterministic"

    # MCP (Model Context Protocol) settings
    mcp_enabled: bool = True
    mcp_servers: str = ""  # JSON array of server configs: [{"name": "...", "transport": "stdio", "command": "npx", "args": ["@playwright/mcp@latest"]}]

    # Authentication
    nextauth_secret: str = ""
    google_client_id: str = ""
    auth_enabled: bool = True

    @model_validator(mode="after")
    def validate_auth_config(self):
        """Ensure NEXTAUTH_SECRET is set when auth is enabled."""
        if self.auth_enabled and not self.nextauth_secret:
            raise ValueError("NEXTAUTH_SECRET must be set when auth is enabled")
        return self

    @model_validator(mode="after")
    def register_custom_providers(self):
        """Parse CUSTOM_PROVIDERS JSON and register them in the provider registry."""
        if self.custom_providers:
            from app.core.provider_registry import CustomProviderConfig, provider_registry

            try:
                configs = json.loads(self.custom_providers)
            except json.JSONDecodeError as e:
                raise ValueError(f"CUSTOM_PROVIDERS contains invalid JSON: {e}") from e

            for i, cfg in enumerate(configs):
                try:
                    provider_registry.register(CustomProviderConfig(**cfg))
                except Exception as e:
                    raise ValueError(f"CUSTOM_PROVIDERS entry {i} is malformed: {e}") from e
        return self

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins from comma-separated string."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    """Get cached settings."""
    return Settings()


settings = get_settings()
