from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from app.ai.model_tiers import ModelTier


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class LLMProvider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GEMINI = "gemini"


class QueryMode(str, Enum):
    TASK = "task"
    RESEARCH = "research"
    DATA = "data"
    APP = "app"
    IMAGE = "image"
    SLIDE = "slide"


class ResearchScenario(str, Enum):
    ACADEMIC = "academic"
    MARKET_ANALYSIS = "market"
    TECHNICAL = "technical"
    NEWS = "news"


# Chat Schemas
class ChatMessage(BaseModel):
    """A single chat message."""

    role: MessageRole
    content: str
    metadata: dict[str, Any] | None = None


class ChatRequest(BaseModel):
    """Request to send a chat message."""

    message: str
    conversation_id: str | None = None
    provider: str | None = None
    model: str | None = None
    history: list[ChatMessage] = Field(default_factory=list)

    @field_validator("provider")
    @classmethod
    def check_provider(cls, v: str | None) -> str | None:
        if v is None or v == "auto":
            return None
        from app.core.provider_registry import provider_registry

        if not provider_registry.is_known(v):
            raise ValueError(f"Unknown provider: {v}")
        return v


class ChatResponse(BaseModel):
    """Response from chat endpoint."""

    id: str
    content: str
    model: str
    provider: str
    tokens: int | None = None


class StreamEvent(BaseModel):
    """Server-sent event for streaming."""

    type: Literal["token", "step", "source", "complete", "error"]
    data: str


# Research Schemas
class ResearchDepth(str, Enum):
    FAST = "fast"
    DEEP = "deep"


class ResearchRequest(BaseModel):
    """Request to start a research task."""

    query: str
    depth: ResearchDepth = ResearchDepth.FAST
    scenario: ResearchScenario = ResearchScenario.ACADEMIC
    conversation_id: str | None = None


class ResearchStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ResearchStepType(str, Enum):
    THINKING = "thinking"
    CONFIG = "config"
    SEARCH = "search"
    SEARCH_TOOLS = "search_tools"
    COLLECT = "collect"
    ANALYZE = "analyze"
    SYNTHESIZE = "synthesize"
    WRITE = "write"
    REPORT = "report"


class ResearchStep(BaseModel):
    """A step in the research process."""

    id: str
    type: ResearchStepType
    description: str
    status: ResearchStatus
    output: str | None = None


class Source(BaseModel):
    """A source found during research."""

    id: str
    title: str
    url: str
    snippet: str | None = None


class ResearchTaskResponse(BaseModel):
    """Response when starting a research task."""

    task_id: str
    status: ResearchStatus


class ResearchResult(BaseModel):
    """Result of a completed research task."""

    task_id: str
    query: str
    status: ResearchStatus
    steps: list[ResearchStep]
    sources: list[Source]
    summary: str | None = None
    report: str | None = None
    created_at: datetime
    completed_at: datetime | None = None


# Unified Query Schemas
class UnifiedQueryRequest(BaseModel):
    """Unified request for both chat and research modes."""

    message: str
    mode: QueryMode = QueryMode.TASK
    scenario: ResearchScenario | None = None
    depth: ResearchDepth = ResearchDepth.FAST
    conversation_id: str | None = None
    task_id: str | None = None  # Optional task ID for research mode (frontend-generated)
    provider: str | None = None
    model: str | None = None
    tier: ModelTier | None = None  # Optional tier override
    history: list[ChatMessage] = Field(default_factory=list)
    attachment_ids: list[str] = Field(default_factory=list)
    locale: str = Field(default="en", description="User's preferred language (e.g., 'en', 'zh-CN')")
    budget: dict[str, Any] | None = Field(
        default=None,
        description="Optional run budget controls (tokens, cost, tool_calls, wall_clock_seconds)",
    )
    execution_mode: Literal["auto", "guided", "strict", "codeact"] = Field(default="auto")
    run_labels: dict[str, str] = Field(default_factory=dict)

    @field_validator("provider")
    @classmethod
    def check_provider(cls, v: str | None) -> str | None:
        if v is None or v == "auto":
            return None
        from app.core.provider_registry import provider_registry

        if not provider_registry.is_known(v):
            raise ValueError(f"Unknown provider: {v}")
        return v


class UnifiedQueryResponse(BaseModel):
    """Unified response for both chat and research modes."""

    id: str
    mode: QueryMode
    task_id: str | None = None
    content: str | None = None
    model: str
    provider: str


# Conversation Schemas
class ConversationType(str, Enum):
    TASK = "task"
    RESEARCH = "research"
    DATA = "data"
    APP = "app"
    IMAGE = "image"
    SLIDE = "slide"


class AttachmentInfo(BaseModel):
    """Attachment info in messages."""

    id: str
    filename: str
    contentType: str  # camelCase to match frontend
    fileSize: int  # camelCase to match frontend
    previewUrl: str | None = None  # camelCase to match frontend
    status: str = "uploaded"


class ConversationMessageResponse(BaseModel):
    """Response model for a conversation message."""

    id: str
    conversation_id: str
    role: MessageRole
    content: str
    metadata: dict[str, Any] | None = None
    created_at: str
    attachments: list[AttachmentInfo] = Field(default_factory=list)


class ConversationResponse(BaseModel):
    """Response model for a conversation."""

    id: str
    title: str
    type: ConversationType
    user_id: str
    project_id: str | None = None
    created_at: str
    updated_at: str
    messages: list[ConversationMessageResponse] = Field(default_factory=list)


class ConversationListResponse(BaseModel):
    """Response model for conversation list (without messages)."""

    id: str
    title: str
    type: ConversationType
    user_id: str
    project_id: str | None = None
    created_at: str
    updated_at: str


class CreateConversationRequest(BaseModel):
    """Request to create a new conversation."""

    title: str
    type: ConversationType = ConversationType.TASK


class UpdateConversationRequest(BaseModel):
    """Request to update a conversation."""

    title: str | None = None


class CreateMessageRequest(BaseModel):
    """Request to add a message to a conversation."""

    role: MessageRole
    content: str
    metadata: dict[str, Any] | None = None
    attachment_ids: list[str] = Field(default_factory=list)


class UpdateMessageRequest(BaseModel):
    """Request to update a message."""

    content: str


# Project Schemas
class CreateProjectRequest(BaseModel):
    """Request to create a new project."""

    name: str
    description: str | None = None
    color: str | None = None


class UpdateProjectRequest(BaseModel):
    """Request to update a project."""

    name: str | None = None
    description: str | None = None
    color: str | None = None


class ProjectListResponse(BaseModel):
    """Response model for project list with item counts."""

    id: str
    name: str
    description: str | None = None
    color: str | None = None
    user_id: str
    created_at: str
    updated_at: str
    conversation_count: int = 0
    research_task_count: int = 0


class ProjectResponse(BaseModel):
    """Response model for a single project with items."""

    id: str
    name: str
    description: str | None = None
    color: str | None = None
    user_id: str
    created_at: str
    updated_at: str
    conversations: list[ConversationListResponse] = Field(default_factory=list)
    research_tasks: list[dict] = Field(default_factory=list)


class AssignItemRequest(BaseModel):
    """Request to assign or remove items from a project."""

    conversation_ids: list[str] = Field(default_factory=list)
    research_task_ids: list[str] = Field(default_factory=list)


# Multimodal Schemas
class ImageGenerationResult(BaseModel):
    """Single image generation result (used by image generation service)."""

    base64_data: str | None = None
    url: str | None = None
    revised_prompt: str | None = None
