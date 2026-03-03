"""Builtin skills for HyperAgent."""

from app.agents.skills.builtin.agentic_search_skill import AgenticSearchSkill
from app.agents.skills.builtin.app_builder_skill import AppBuilderSkill
from app.agents.skills.builtin.code_generation_skill import CodeGenerationSkill
from app.agents.skills.builtin.data_analysis_skill import DataAnalysisSkill
from app.agents.skills.builtin.deep_research_skill import DeepResearchSkill
from app.agents.skills.builtin.image_generation_skill import ImageGenerationSkill
from app.agents.skills.builtin.slide_generation_skill import SlideGenerationSkill
from app.agents.skills.builtin.task_planning_skill import TaskPlanningSkill
from app.agents.skills.builtin.web_research_skill import WebResearchSkill

__all__ = [
    "AgenticSearchSkill",
    "WebResearchSkill",
    "DataAnalysisSkill",
    "ImageGenerationSkill",
    "CodeGenerationSkill",
    "TaskPlanningSkill",
    "AppBuilderSkill",
    "SlideGenerationSkill",
    "DeepResearchSkill",
]
