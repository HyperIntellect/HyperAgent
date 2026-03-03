"""Skill registry service for managing and loading skills."""

from enum import IntEnum
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.skills.skill_base import (
    Skill,
    SkillContext,
    SkillMetadata,
    SkillParameter,
    SkillState,
    ToolSkill,
)
from app.core.logging import get_logger
from app.db.models import SkillDefinition
from app.skills.validator import skill_code_validator

logger = get_logger(__name__)


class SkillLoadLevel(IntEnum):
    """Progressive skill loading levels.

    Level 1: Metadata only (~100 tokens) -- loaded at startup for all skills
    Level 2: Full instructions (<5k tokens) -- loaded when skill is triggered
    Level 3: Resources/scripts -- fetched on-demand during execution
    """

    METADATA = 1
    INSTRUCTIONS = 2
    RESOURCES = 3


def _safe_import(
    name: str,
    globals_dict: dict[str, Any] | None = None,
    locals_dict: dict[str, Any] | None = None,
    fromlist: tuple[str, ...] | list[str] = (),
    level: int = 0,
):
    """Restricted __import__ for dynamic skill execution.

    Only modules explicitly allowed by the skill validator can be imported.
    """
    del globals_dict, locals_dict
    if level != 0:
        raise ImportError("Relative imports are not allowed in dynamic skills")

    allowed_imports = skill_code_validator.ALLOWED_IMPORTS
    is_allowed = any(
        name == allowed or name.startswith(allowed + ".")
        for allowed in allowed_imports
    )
    if not is_allowed:
        raise ImportError(f"Import not allowed in dynamic skills: {name}")

    return __import__(name, {}, {}, fromlist, level)


# Safe builtins for dynamic skill execution
# Only include functions that are safe and necessary for skill code
SAFE_BUILTINS = {
    # Class definition support (required for `class X(...)` in dynamic skills)
    "__build_class__": __build_class__,
    "object": object,
    "type": type,
    # Type constructors
    "bool": bool,
    "int": int,
    "float": float,
    "str": str,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
    "frozenset": frozenset,
    "bytes": bytes,
    "bytearray": bytearray,
    # Functional tools
    "len": len,
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "sorted": sorted,
    "reversed": reversed,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "round": round,
    "all": all,
    "any": any,
    # Type checking
    "isinstance": isinstance,
    "issubclass": issubclass,
    "callable": callable,
    # String and formatting
    "repr": repr,
    "format": format,
    "chr": chr,
    "ord": ord,
    # Iteration
    "iter": iter,
    "next": next,
    # Exceptions (for raising/catching)
    "Exception": Exception,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "RuntimeError": RuntimeError,
    # Other safe operations
    "print": print,  # Safe for logging in skills
    "id": id,
    "hash": hash,
    "slice": slice,
    "property": property,
    "staticmethod": staticmethod,
    "classmethod": classmethod,
    # Controlled imports for dynamic skills
    "__import__": _safe_import,
    # None, True, False are automatically available
}


def _create_safe_namespace() -> dict[str, Any]:
    """Create a restricted namespace for dynamic skill execution.

    This namespace includes only safe operations and explicitly allowed imports.
    Dangerous operations like exec, eval, and open are excluded.

    Returns:
        Dictionary to use as globals for exec()
    """
    namespace: dict[str, Any] = {
        "__builtins__": SAFE_BUILTINS,
        # Skill base classes (required for all skills)
        "Skill": Skill,
        "ToolSkill": ToolSkill,
        "SkillContext": SkillContext,
        "SkillMetadata": SkillMetadata,
        "SkillParameter": SkillParameter,
        "SkillState": SkillState,
    }

    # Import allowed modules and add them to namespace
    # These must match the ALLOWED_IMPORTS in validator.py
    try:
        from typing import Any as TypingAny
        from typing import Literal, TypedDict
        from typing import Optional as TypingOptional

        namespace["Any"] = TypingAny
        namespace["Optional"] = TypingOptional
        namespace["Literal"] = Literal
        namespace["TypedDict"] = TypedDict

        import json as json_module

        namespace["json"] = json_module

        import re as re_module

        namespace["re"] = re_module

        import datetime as datetime_module

        namespace["datetime"] = datetime_module

        from langgraph.graph import END, StateGraph

        namespace["StateGraph"] = StateGraph
        namespace["END"] = END

        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

        namespace["AIMessage"] = AIMessage
        namespace["HumanMessage"] = HumanMessage
        namespace["SystemMessage"] = SystemMessage
        namespace["ToolMessage"] = ToolMessage

        from pydantic import BaseModel, Field

        namespace["BaseModel"] = BaseModel
        namespace["Field"] = Field

        # App-specific imports for skills
        from app.agents import events

        namespace["events"] = events

        from app.ai.llm import LLMService, llm_service

        namespace["llm_service"] = llm_service
        namespace["LLMService"] = LLMService

        from app.services.search import search_service

        namespace["search_service"] = search_service

        from app.core.logging import get_logger

        namespace["get_logger"] = get_logger

    except ImportError as e:
        logger.warning("safe_namespace_import_failed", error=str(e))

    return namespace


class SkillRegistry:
    """Central registry for managing and loading skills.

    Supports progressive loading with three levels:
      - Level 1 (METADATA): Only metadata loaded at startup (~100 tokens per skill).
      - Level 2 (INSTRUCTIONS): Full skill class instantiated and graph compiled.
      - Level 3 (RESOURCES): External resources fetched on-demand (placeholder).
    """

    def __init__(self):
        self._loaded_skills: dict[str, Skill] = {}
        self._builtin_skills: dict[str, type[Skill]] = {}
        self._skill_metadata_cache: dict[str, SkillMetadata] = {}
        self._skill_load_levels: dict[str, SkillLoadLevel] = {}

    async def initialize(self, db: AsyncSession):
        """Initialize registry with builtin skill metadata (Level 1) and load from database.

        Builtin skills are registered at Level 1 (metadata only) to minimise
        startup cost.  Full instantiation is deferred to first use.

        Args:
            db: Database session for loading skill definitions
        """
        logger.info("skill_registry_initializing")
        await self._register_builtin_skills()
        await self._load_from_database(db)
        logger.info(
            "skill_registry_initialized",
            metadata_count=len(self._skill_metadata_cache),
            loaded_count=len(self._loaded_skills),
        )

    async def _register_builtin_skills(self):
        """Auto-discover and register builtin skills at Level 1 (metadata only).

        Skill classes are imported and instantiated *temporarily* to extract
        their metadata, then the instance is discarded.  The class reference
        is kept in ``_builtin_skills`` so that ``_load_skill_full`` can
        re-instantiate on demand (Level 2).
        """
        try:
            # Import builtin skills
            from app.agents.skills.builtin import (
                AgenticSearchSkill,
                AppBuilderSkill,
                CodeGenerationSkill,
                DataAnalysisSkill,
                DeepResearchSkill,
                ImageGenerationSkill,
                SlideGenerationSkill,
                TaskPlanningSkill,
            )

            # Register each skill at Level 1 (metadata only)
            # Note: WebResearchSkill is now an alias for AgenticSearchSkill
            # (both share ID "web_research"), so we only register AgenticSearchSkill.
            for skill_class in [
                AgenticSearchSkill,
                DeepResearchSkill,
                DataAnalysisSkill,
                ImageGenerationSkill,
                CodeGenerationSkill,
                TaskPlanningSkill,
                AppBuilderSkill,
                SlideGenerationSkill,
            ]:
                skill = skill_class()
                metadata = skill.metadata
                self._builtin_skills[metadata.id] = skill_class
                self._skill_metadata_cache[metadata.id] = metadata
                self._skill_load_levels[metadata.id] = SkillLoadLevel.METADATA
                logger.info(
                    "builtin_skill_registered",
                    skill_id=metadata.id,
                    skill_name=metadata.name,
                    load_level="metadata",
                )
                # Do NOT store the full skill instance -- defer to Level 2
        except ImportError as e:
            logger.warning("builtin_skills_import_failed", error=str(e))

    async def _load_from_database(self, db: AsyncSession):
        """Load enabled skills from database.

        Args:
            db: Database session
        """
        try:
            # Query enabled non-builtin skills
            result = await db.execute(
                select(SkillDefinition).where(
                    SkillDefinition.enabled.is_(True),
                    SkillDefinition.is_builtin.is_(False),
                )
            )
            skill_defs = result.scalars().all()

            for skill_def in skill_defs:
                try:
                    # Load dynamic skill with validation
                    if skill_def.source_code:
                        await self._load_dynamic_skill(skill_def)
                    else:
                        logger.info(
                            "dynamic_skill_skipped",
                            skill_id=skill_def.id,
                            reason="no_source_code",
                        )
                except Exception as e:
                    logger.error(
                        "skill_load_failed",
                        skill_id=skill_def.id,
                        error=str(e),
                    )
        except Exception as e:
            logger.error("database_skill_load_failed", error=str(e))

    async def _load_dynamic_skill(self, skill_def: SkillDefinition):
        """Load and validate a dynamic skill from source code.

        Args:
            skill_def: Skill definition with source code

        Raises:
            ValueError: If validation fails or skill cannot be loaded
        """
        # Validate source code
        is_valid, error = skill_code_validator.validate(skill_def.source_code)
        if not is_valid:
            logger.error(
                "dynamic_skill_validation_failed",
                skill_id=skill_def.id,
                error=error,
            )
            raise ValueError(f"Skill validation failed: {error}")

        # Verify hash matches
        computed_hash = skill_code_validator.compute_hash(skill_def.source_code)
        if skill_def.source_code_hash and computed_hash != skill_def.source_code_hash:
            logger.error(
                "dynamic_skill_hash_mismatch",
                skill_id=skill_def.id,
            )
            raise ValueError("Source code hash mismatch - possible tampering")

        # Execute source code in restricted namespace with safe builtins only.
        # __import__ is restricted to validator-approved modules via _safe_import().
        namespace = _create_safe_namespace()
        try:
            # Use compile() first to catch syntax errors with better error messages
            compiled_code = compile(skill_def.source_code, f"<skill:{skill_def.id}>", "exec")
            exec(compiled_code, namespace)
        except SyntaxError as e:
            logger.error(
                "dynamic_skill_syntax_error",
                skill_id=skill_def.id,
                error=str(e),
                line=e.lineno,
            )
            raise ValueError(f"Syntax error in skill code: {e}")
        except NameError as e:
            logger.error(
                "dynamic_skill_name_error",
                skill_id=skill_def.id,
                error=str(e),
            )
            raise ValueError(
                f"Undefined name in skill code (may be using forbidden operation): {e}"
            )
        except Exception as e:
            logger.error(
                "dynamic_skill_execution_failed",
                skill_id=skill_def.id,
                error=str(e),
            )
            raise ValueError(f"Failed to execute skill code: {e}")

        # Find and instantiate the skill class
        skill_class = None
        for item in namespace.values():
            if (
                isinstance(item, type)
                and issubclass(item, Skill)
                and item not in (Skill, ToolSkill)
            ):
                skill_class = item
                break

        if not skill_class:
            raise ValueError("No Skill subclass found in source code")

        # Instantiate and register
        skill = skill_class()
        self._loaded_skills[skill.metadata.id] = skill
        # Dynamic skills are fully loaded (Level 2) and also cached in metadata
        self._skill_metadata_cache[skill.metadata.id] = skill.metadata
        self._skill_load_levels[skill.metadata.id] = SkillLoadLevel.INSTRUCTIONS

        logger.info(
            "dynamic_skill_loaded",
            skill_id=skill_def.id,
            skill_name=skill.metadata.name,
        )

    # ------------------------------------------------------------------
    # Progressive loading helpers
    # ------------------------------------------------------------------

    def _load_skill_full(self, skill_id: str) -> Skill | None:
        """Load a skill to Level 2 (full instructions + graph compilation).

        For builtin skills this re-instantiates the skill class.  For dynamic
        skills this is a no-op (they are already fully loaded during
        ``_load_dynamic_skill``).

        Returns:
            The loaded Skill instance, or None if the skill cannot be loaded.
        """
        if skill_id in self._builtin_skills:
            try:
                skill_class = self._builtin_skills[skill_id]
                skill = skill_class()
                return skill
            except Exception as e:
                logger.error("skill_full_load_failed", skill_id=skill_id, error=str(e))
        return None

    async def ensure_loaded(
        self,
        skill_id: str,
        level: SkillLoadLevel = SkillLoadLevel.INSTRUCTIONS,
    ) -> Skill | None:
        """Ensure a skill is loaded to the requested level.

        Level 1 (METADATA): Already loaded at startup -- no-op.
        Level 2 (INSTRUCTIONS): Load full skill class and compile graph.
        Level 3 (RESOURCES): Load any external resources/data (placeholder).

        Returns:
            The loaded Skill or None if not found.
        """
        current_level = self._skill_load_levels.get(skill_id, SkillLoadLevel.METADATA)

        if current_level >= level:
            return self._loaded_skills.get(skill_id)

        if skill_id not in self._skill_metadata_cache:
            return None

        # Level 2: Load full skill
        if level >= SkillLoadLevel.INSTRUCTIONS and current_level < SkillLoadLevel.INSTRUCTIONS:
            skill = self._load_skill_full(skill_id)
            if skill:
                self._loaded_skills[skill_id] = skill
                self._skill_load_levels[skill_id] = SkillLoadLevel.INSTRUCTIONS
                logger.info("skill_loaded_level2", skill_id=skill_id)

        # Level 3: Resources (currently no-op, placeholder for future)
        if level >= SkillLoadLevel.RESOURCES:
            self._skill_load_levels[skill_id] = SkillLoadLevel.RESOURCES

        return self._loaded_skills.get(skill_id)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def unload_skill(self, skill_id: str) -> bool:
        """Remove a skill from the in-memory registry.

        Removes the full skill instance (Level 2) and demotes to Level 1
        if metadata is still cached.

        Args:
            skill_id: Unique identifier for the skill

        Returns:
            True if the skill was found and removed, False otherwise
        """
        if skill_id in self._loaded_skills:
            del self._loaded_skills[skill_id]
            # Demote back to Level 1 if metadata still exists
            if skill_id in self._skill_metadata_cache:
                self._skill_load_levels[skill_id] = SkillLoadLevel.METADATA
            else:
                self._skill_load_levels.pop(skill_id, None)
            return True
        return False

    def get_skill(self, skill_id: str) -> Optional[Skill]:
        """Get a skill, auto-loading to Level 2 if needed.

        If the skill is only at Level 1 (metadata cached), it will be
        promoted to Level 2 (full instantiation) automatically.

        Args:
            skill_id: Unique identifier for the skill

        Returns:
            Skill instance if found, None otherwise
        """
        if skill_id in self._loaded_skills:
            return self._loaded_skills[skill_id]

        # Auto-promote from Level 1 to Level 2
        if skill_id in self._skill_metadata_cache:
            skill = self._load_skill_full(skill_id)
            if skill:
                self._loaded_skills[skill_id] = skill
                self._skill_load_levels[skill_id] = SkillLoadLevel.INSTRUCTIONS
                logger.info("skill_auto_promoted_to_level2", skill_id=skill_id)
                return skill

        return None

    def list_skills(
        self,
        category: Optional[str] = None,
        enabled_only: bool = True,
    ) -> list[SkillMetadata]:
        """List available skills using Level 1 metadata (no full loading required).

        This uses the metadata cache so that listing skills does not trigger
        full skill instantiation.

        Args:
            category: Filter by category (e.g., "research", "code", "data")
            enabled_only: Only return enabled skills

        Returns:
            List of skill metadata
        """
        results: list[SkillMetadata] = []
        for skill_id, metadata in self._skill_metadata_cache.items():
            if enabled_only and not metadata.enabled:
                continue
            if category and metadata.category != category:
                continue
            results.append(metadata)
        return results

    def get_load_stats(self) -> dict:
        """Get statistics about skill loading levels.

        Returns:
            Dictionary with total_skills, loaded_full count, and
            a breakdown by level name.
        """
        levels: dict[str, int] = {}
        for skill_id in self._skill_metadata_cache:
            level = self._skill_load_levels.get(skill_id, SkillLoadLevel.METADATA)
            level_name = level.name.lower()
            levels[level_name] = levels.get(level_name, 0) + 1

        return {
            "total_skills": len(self._skill_metadata_cache),
            "loaded_full": len(self._loaded_skills),
            "by_level": levels,
        }

    async def register_skill(
        self,
        db: AsyncSession,
        skill: Skill,
        is_builtin: bool = False,
    ):
        """Register a new skill in the database.

        Args:
            db: Database session
            skill: Skill instance to register
            is_builtin: Whether this is a builtin skill
        """
        # Create skill definition in database
        skill_def = SkillDefinition(
            id=skill.metadata.id,
            name=skill.metadata.name,
            version=skill.metadata.version,
            description=skill.metadata.description,
            category=skill.metadata.category,
            module_path="",  # Will be set for dynamic skills
            metadata_json=skill.metadata.model_dump_json(),
            enabled=skill.metadata.enabled,
            is_builtin=is_builtin,
            author=skill.metadata.author,
        )

        db.add(skill_def)
        await db.commit()

        # Add to loaded skills and metadata cache
        self._loaded_skills[skill.metadata.id] = skill
        self._skill_metadata_cache[skill.metadata.id] = skill.metadata
        self._skill_load_levels[skill.metadata.id] = SkillLoadLevel.INSTRUCTIONS

        logger.info(
            "skill_registered",
            skill_id=skill.metadata.id,
            is_builtin=is_builtin,
        )

    async def sync_builtin_skills(self, db: AsyncSession):
        """Sync builtin skills to database.

        This ensures all builtin skills have database records.

        Args:
            db: Database session
        """
        for skill_id, skill_class in self._builtin_skills.items():
            # Check if skill already exists in database
            result = await db.execute(select(SkillDefinition).where(SkillDefinition.id == skill_id))
            existing = result.scalar_one_or_none()

            if not existing:
                # Create new database record
                skill = skill_class()
                await self.register_skill(db, skill, is_builtin=True)
                logger.info("builtin_skill_synced", skill_id=skill_id)
            else:
                # Update metadata if needed
                skill = skill_class()
                existing.metadata_json = skill.metadata.model_dump_json()
                existing.version = skill.metadata.version
                existing.description = skill.metadata.description
                existing.name = skill.metadata.name
                existing.category = skill.metadata.category
                existing.author = skill.metadata.author
                await db.commit()


# Global singleton
skill_registry = SkillRegistry()
