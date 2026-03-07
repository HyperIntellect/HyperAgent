"""Tests for query complexity classification."""
import pytest
from app.agents.classifier import classify_query


@pytest.mark.parametrize("query,mode,expected", [
    ("Hello, how are you?", None, "simple"),
    ("What is Python?", None, "simple"),
    ("Thanks!", None, "simple"),
    ("Build me an app", "app", "simple"),
    ("Generate an image of a cat", "image", "simple"),
    ("Create slides about AI", "slide", "simple"),
    ("Analyze this CSV", "data", "simple"),
    ("First create a backend API, then build a frontend, and deploy it", None, "complex"),
    ("Step 1: design the schema. Step 2: implement the API", None, "complex"),
    ("Build a complete weather dashboard with backend and frontend", None, "complex"),
    ("Plan and implement a user authentication system", None, "complex"),
    ("Research AI developments and write a comprehensive report", "research", "simple"),
    ("Do something", None, "simple"),
])
def test_classify_query(query, mode, expected):
    result = classify_query(query, mode=mode, skills=[])
    assert result == expected, f"Expected {expected} for query={query!r}, mode={mode}"


def test_classify_explicit_skills_is_simple():
    result = classify_query("complex multi-step task", mode=None, skills=["image_generation"])
    assert result == "simple"


def test_classify_short_followup_is_simple():
    result = classify_query("yes", mode=None, skills=[])
    assert result == "simple"
