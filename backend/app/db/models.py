"""Database ORM models for research tasks."""

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, JSON, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class User(Base):
    """User model for authentication."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    image: Mapped[str | None] = mapped_column(Text, nullable=True)
    google_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    research_tasks: Mapped[list["ResearchTask"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    conversations: Mapped[list["Conversation"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    projects: Mapped[list["Project"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    files: Mapped[list["File"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    skill_executions: Mapped[list["SkillExecution"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "email": self.email,
            "name": self.name,
            "image": self.image,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class ResearchTask(Base):
    """Research task model."""

    __tablename__ = "research_tasks"
    __table_args__ = (
        # Composite indexes for common query patterns
        Index("ix_research_tasks_user_status", "user_id", "status"),
        Index("ix_research_tasks_user_created", "user_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    depth: Mapped[str] = mapped_column(String(20), default="standard")
    scenario: Mapped[str] = mapped_column(String(20), default="academic")
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True)
    report: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Worker tracking fields
    worker_job_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    worker_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    progress: Mapped[int] = mapped_column(default=0)
    retry_count: Mapped[int] = mapped_column(default=0)
    priority: Mapped[int] = mapped_column(default=0)

    # Project association
    project_id: Mapped[str | None] = mapped_column(
        ForeignKey("projects.id", ondelete="SET NULL"), nullable=True, index=True
    )

    # Relationships
    user: Mapped["User"] = relationship(back_populates="research_tasks")
    project: Mapped["Project | None"] = relationship(back_populates="research_tasks")
    steps: Mapped[list["ResearchStep"]] = relationship(
        back_populates="task", cascade="all, delete-orphan", order_by="ResearchStep.created_at"
    )
    sources: Mapped[list["ResearchSource"]] = relationship(
        back_populates="task", cascade="all, delete-orphan"
    )

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        from sqlalchemy import inspect

        # Check if relationships are loaded
        steps_list = []
        if inspect(self).attrs.steps.loaded_value:
            steps_list = [step.to_dict() for step in self.steps]

        sources_list = []
        if inspect(self).attrs.sources.loaded_value:
            sources_list = [source.to_dict() for source in self.sources]

        return {
            "id": self.id,
            "query": self.query,
            "depth": self.depth,
            "scenario": self.scenario,
            "status": self.status,
            "report": self.report,
            "error": self.error,
            "user_id": self.user_id,
            "project_id": self.project_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "progress": self.progress,
            "worker_job_id": self.worker_job_id,
            "steps": steps_list,
            "sources": sources_list,
        }


class ResearchStep(Base):
    """Research step model."""

    __tablename__ = "research_steps"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("research_tasks.id", ondelete="CASCADE"))
    type: Mapped[str] = mapped_column(String(20), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    output: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    task: Mapped["ResearchTask"] = relationship(back_populates="steps")

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "type": self.type,
            "description": self.description,
            "status": self.status,
            "output": self.output,
        }


class ResearchSource(Base):
    """Research source model."""

    __tablename__ = "research_sources"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("research_tasks.id", ondelete="CASCADE"))
    title: Mapped[str] = mapped_column(Text, nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    snippet: Mapped[str | None] = mapped_column(Text, nullable=True)
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    relevance_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Relationships
    task: Mapped["ResearchTask"] = relationship(back_populates="sources")

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "relevance_score": self.relevance_score,
        }


class BackgroundTask(Base):
    """Generic background task model for non-research async work."""

    __tablename__ = "background_tasks"
    __table_args__ = (
        # Composite indexes for worker polling and status queries
        Index("ix_background_tasks_status_priority", "status", "priority"),
        Index("ix_background_tasks_user_status", "user_id", "status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    task_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True)
    priority: Mapped[int] = mapped_column(default=0)

    # Input/Output
    payload: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON input
    result: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON result
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Tracking
    worker_job_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    worker_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    progress: Mapped[int] = mapped_column(default=0)
    retry_count: Mapped[int] = mapped_column(default=0)
    max_retries: Mapped[int] = mapped_column(default=3)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    scheduled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # User association
    user_id: Mapped[str | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "task_type": self.task_type,
            "status": self.status,
            "priority": self.priority,
            "payload": self.payload,
            "result": self.result,
            "error": self.error,
            "progress": self.progress,
            "worker_job_id": self.worker_job_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "scheduled_at": self.scheduled_at.isoformat() if self.scheduled_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class Project(Base):
    """Project model for organizing conversations and research tasks."""

    __tablename__ = "projects"
    __table_args__ = (
        Index("ix_projects_user_created", "user_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    color: Mapped[str | None] = mapped_column(String(20), nullable=True)
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    user: Mapped["User"] = relationship(back_populates="projects")
    conversations: Mapped[list["Conversation"]] = relationship(
        back_populates="project"
    )
    research_tasks: Mapped[list["ResearchTask"]] = relationship(
        back_populates="project"
    )

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "color": self.color,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class Conversation(Base):
    """Conversation model for chat sessions."""

    __tablename__ = "conversations"
    __table_args__ = (
        # Composite indexes for common query patterns (listing user conversations by time)
        Index("ix_conversations_user_created", "user_id", "created_at"),
        Index("ix_conversations_user_updated", "user_id", "updated_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    type: Mapped[str] = mapped_column(String(20), nullable=False, default="task")
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    project_id: Mapped[str | None] = mapped_column(
        ForeignKey("projects.id", ondelete="SET NULL"), nullable=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    user: Mapped["User"] = relationship(back_populates="conversations")
    project: Mapped["Project | None"] = relationship(back_populates="conversations")
    messages: Mapped[list["ConversationMessage"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan", order_by="ConversationMessage.created_at"
    )

    def to_dict(self, include_messages: bool = True) -> dict:
        """Convert to dictionary for API responses."""
        from sqlalchemy import inspect

        result = {
            "id": self.id,
            "title": self.title,
            "type": self.type,
            "user_id": self.user_id,
            "project_id": self.project_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
        if include_messages:
            # Check if messages relationship is loaded
            if inspect(self).attrs.messages.loaded_value:
                result["messages"] = [msg.to_dict() for msg in self.messages]
            else:
                result["messages"] = []
        return result


class ConversationMessage(Base):
    """Message model for conversation messages."""

    __tablename__ = "conversation_messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    message_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    conversation: Mapped["Conversation"] = relationship(back_populates="messages")
    attachments: Mapped[list["MessageAttachment"]] = relationship(
        back_populates="message", cascade="all, delete-orphan"
    )

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        from sqlalchemy import inspect

        # Check if attachments relationship is loaded
        attachments_list = []
        if inspect(self).attrs.attachments.loaded_value:
            for att in self.attachments:
                # Check if file relationship is loaded on each attachment
                if inspect(att).attrs.file.loaded_value:
                    # Generate preview URL for the file
                    preview_url = f"/api/v1/files/download/{att.file.storage_key}"

                    attachments_list.append({
                        "id": att.file.id,
                        "filename": att.file.original_filename,
                        "contentType": att.file.content_type,  # camelCase for frontend
                        "fileSize": att.file.file_size,  # camelCase for frontend
                        "previewUrl": preview_url,  # Preview URL for images
                        "status": "uploaded",  # Status for frontend FileAttachment interface
                    })

        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "role": self.role,
            "content": self.content,
            "metadata": self.message_metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "attachments": attachments_list,
        }


class File(Base):
    """File model for uploaded files."""

    __tablename__ = "files"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # File metadata
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    content_type: Mapped[str] = mapped_column(String(100), nullable=False)
    file_size: Mapped[int] = mapped_column(nullable=False)
    file_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)  # SHA256 hash for deduplication

    # Storage info
    storage_key: Mapped[str] = mapped_column(String(500), nullable=False, unique=True)
    storage_bucket: Mapped[str] = mapped_column(String(100), nullable=False)

    # Content extraction (for LLM context)
    extracted_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    extraction_status: Mapped[str] = mapped_column(String(20), default="pending")

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    user: Mapped["User"] = relationship(back_populates="files")
    message_attachments: Mapped[list["MessageAttachment"]] = relationship(
        back_populates="file", cascade="all, delete-orphan"
    )

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "filename": self.original_filename,
            "content_type": self.content_type,
            "file_size": self.file_size,
            "extraction_status": self.extraction_status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class MessageAttachment(Base):
    """Junction table linking files to conversation messages."""

    __tablename__ = "message_attachments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    message_id: Mapped[str] = mapped_column(
        ForeignKey("conversation_messages.id", ondelete="CASCADE"), nullable=False, index=True
    )
    file_id: Mapped[str] = mapped_column(
        ForeignKey("files.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Relationships
    message: Mapped["ConversationMessage"] = relationship(back_populates="attachments")
    file: Mapped["File"] = relationship(back_populates="message_attachments")

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "message_id": self.message_id,
            "file_id": self.file_id,
        }


class SkillDefinition(Base):
    """Skill definition model for dynamic skills."""

    __tablename__ = "skill_definitions"

    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    version: Mapped[str] = mapped_column(String(20), default="1.0.0")
    description: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(String(50), nullable=False, index=True)

    # Skill code and configuration
    module_path: Mapped[str] = mapped_column(String(500), nullable=False)
    metadata_json: Mapped[str] = mapped_column(Text, nullable=False)

    # Dynamic skills (marketplace)
    source_code: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_code_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Status
    enabled: Mapped[bool] = mapped_column(default=True, index=True)
    is_builtin: Mapped[bool] = mapped_column(default=False)

    # Metadata
    author: Mapped[str] = mapped_column(String(255), default="hyperagent")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Usage tracking
    invocation_count: Mapped[int] = mapped_column(default=0)
    last_invoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    executions: Mapped[list["SkillExecution"]] = relationship(
        back_populates="skill", cascade="all, delete-orphan"
    )

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        result = {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "category": self.category,
            "enabled": self.enabled,
            "is_builtin": self.is_builtin,
            "author": self.author,
            "invocation_count": self.invocation_count,
            "last_invoked_at": self.last_invoked_at.isoformat() if self.last_invoked_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

        # Enrich with fields from metadata_json blob
        if self.metadata_json:
            try:
                import json

                meta = json.loads(self.metadata_json)
                result["parameters"] = meta.get("parameters", [])
                result["required_tools"] = meta.get("required_tools", [])
                result["risk_level"] = meta.get("risk_level")
                result["tags"] = meta.get("tags", [])
                result["output_schema"] = meta.get("output_schema", {})
            except (json.JSONDecodeError, TypeError):
                pass

        return result


class SkillExecution(Base):
    """Execution history for skills."""

    __tablename__ = "skill_executions"
    __table_args__ = (
        # Composite indexes for common query patterns
        Index("ix_skill_executions_user_status", "user_id", "status"),
        Index("ix_skill_executions_skill_status", "skill_id", "status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    skill_id: Mapped[str] = mapped_column(ForeignKey("skill_definitions.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    # Execution details
    input_params: Mapped[str] = mapped_column(Text, nullable=False)  # JSON
    output_data: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Timing
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    execution_time_ms: Mapped[int | None] = mapped_column(nullable=True)

    # Context
    agent_type: Mapped[str] = mapped_column(String(50), nullable=False)
    task_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    skill: Mapped["SkillDefinition"] = relationship(back_populates="executions")
    user: Mapped["User"] = relationship(back_populates="skill_executions")

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "skill_id": self.skill_id,
            "user_id": self.user_id,
            "status": self.status,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "execution_time_ms": self.execution_time_ms,
            "agent_type": self.agent_type,
            "task_id": self.task_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Memory(Base):
    """Persistent cross-session memory entry."""

    __tablename__ = "memories"
    __table_args__ = (
        Index("idx_memories_user_id", "user_id"),
        Index("idx_memories_type", "user_id", "memory_type"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False)
    memory_type: Mapped[str] = mapped_column(String(50), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True, server_default="{}")
    source_conversation_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    last_accessed: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    access_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default=func.text("0"))

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        import json as _json

        metadata = {}
        if self.metadata_json:
            try:
                metadata = _json.loads(self.metadata_json)
            except (_json.JSONDecodeError, TypeError):
                pass

        return {
            "id": self.id,
            "user_id": self.user_id,
            "type": self.memory_type,
            "content": self.content,
            "metadata": metadata,
            "source_conversation_id": self.source_conversation_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "access_count": self.access_count,
        }


class AgentRun(Base):
    """Durable run ledger entry for agent executions."""

    __tablename__ = "agent_runs"
    __table_args__ = (
        Index("ix_agent_runs_user_created", "user_id", "created_at"),
        Index("ix_agent_runs_status", "status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    task_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    conversation_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    user_id: Mapped[str | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    mode: Mapped[str] = mapped_column(String(20), nullable=False, default="task")
    objective: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="running")
    execution_mode: Mapped[str] = mapped_column(String(20), nullable=False, default="auto")
    run_labels: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    budget_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    outcome_label: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    outcome_reason_code: Mapped[str | None] = mapped_column(String(128), nullable=True)
    quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "task_id": self.task_id,
            "conversation_id": self.conversation_id,
            "user_id": self.user_id,
            "mode": self.mode,
            "objective": self.objective,
            "status": self.status,
            "execution_mode": self.execution_mode,
            "run_labels": self.run_labels or {},
            "budget": self.budget_json or {},
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "last_error": self.last_error,
            "outcome_label": self.outcome_label,
            "outcome_reason_code": self.outcome_reason_code,
            "quality_score": self.quality_score,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class AgentStep(Base):
    """Step-level transitions for planner/executor runs."""

    __tablename__ = "agent_steps"
    __table_args__ = (
        Index("ix_agent_steps_run_created", "run_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("agent_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    step_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    step_type: Mapped[str] = mapped_column(String(32), nullable=False, default="unknown")
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="running")
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    details: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class AgentEvent(Base):
    """Normalized event stream for replay and audit."""

    __tablename__ = "agent_events"
    __table_args__ = (
        Index("ix_agent_events_run_created", "run_id", "created_at"),
        Index("ix_agent_events_dedup_key", "dedup_key"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("agent_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    event_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    step_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    dedup_key: Mapped[str | None] = mapped_column(String(255), nullable=True)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    payload_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ToolExecution(Base):
    """Tool execution audit records with policy decision."""

    __tablename__ = "tool_executions"
    __table_args__ = (
        Index("ix_tool_executions_run_created", "run_id", "created_at"),
        Index("ix_tool_executions_tool_name", "tool_name"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("agent_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    step_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    tool_call_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    tool_name: Mapped[str] = mapped_column(String(100), nullable=False)
    tool_args: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="running")
    policy_decision: Mapped[str | None] = mapped_column(String(32), nullable=True)
    policy_reason: Mapped[str | None] = mapped_column(String(100), nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    result_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class SandboxSnapshot(Base):
    """Persistent sandbox workspace snapshot for restoring state after disconnect."""

    __tablename__ = "sandbox_snapshots"
    __table_args__ = (
        Index("ix_sandbox_snapshots_user_task_type", "user_id", "task_id", "sandbox_type"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    task_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    sandbox_type: Mapped[str] = mapped_column(String(20), nullable=False)  # execution, app
    storage_key: Mapped[str] = mapped_column(String(500), nullable=False, unique=True)
    paths_included: Mapped[list] = mapped_column(JSON, nullable=False)  # list of paths
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "task_id": self.task_id,
            "sandbox_type": self.sandbox_type,
            "storage_key": self.storage_key,
            "paths_included": self.paths_included,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }
