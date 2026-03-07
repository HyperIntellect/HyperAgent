# Agent Evaluation Framework

## Overview

HyperAgent includes a comprehensive evaluation framework for testing the multi-agent system. The framework covers three key areas:

1. **Routing Accuracy** - Does the supervisor route queries to the correct agent?
2. **Tool Selection** - Does the agent select appropriate tools and skills?
3. **Response Quality** - Are responses helpful, accurate, and safe?

## Architecture

```
backend/evals/
├── __init__.py              # Module exports
├── conftest.py              # Pytest fixtures with mock LLMs
├── datasets/
│   ├── routing.json         # 31 routing test cases (incl. adversarial)
│   ├── tool_selection.json  # 23 tool selection test cases (all 7 skills)
│   └── response_quality.json # 12 response quality test cases
├── mocks/
│   ├── __init__.py
│   └── mock_llm.py          # MockChatModel and MockRouterLLM
├── evaluators/
│   ├── __init__.py
│   ├── routing_evaluator.py      # Routing accuracy evaluator
│   ├── tool_selection_evaluator.py # Tool selection evaluator
│   └── response_quality_evaluator.py # LLM-as-judge evaluator
├── test_routing.py          # Routing test suite
├── test_tool_selection.py   # Tool selection test suite
└── test_response_quality.py # Response quality test suite
```

## Running Evaluations

```bash
# Run all evaluations
make eval

# Run individual eval suites
make eval-routing   # Routing accuracy tests
make eval-tools     # Tool/skill selection tests
make eval-quality   # Response quality tests

# Run with LangSmith tracking
make eval-langsmith
```

## Evaluation Types

### 1. Routing Accuracy

Tests whether the supervisor correctly routes queries to the appropriate agent (task or research).

**Test Categories:**
- General conversation → Task Agent
- Image generation → Task Agent (uses `image_generation` skill)
- Slide/presentation generation → Task Agent (uses `slide_generation` skill / `generate_slides` tool)
- Code tasks → Task Agent (uses `code_generation` skill)
- Writing tasks → Task Agent (handled directly by LLM)
- Browser automation → Task Agent (uses browser tools)
- Data analytics → Task Agent (uses `data_analysis` skill)
- Comprehensive research → Research Agent

**Success Criteria:** ≥ 90% accuracy

**Example Test Case:**
```json
{
  "id": "research_comprehensive_paper",
  "input": "Write a comprehensive 20-page research paper on quantum computing with citations",
  "expected_agent": "research",
  "category": "research"
}
```

### 2. Tool Selection

Tests whether agents select the correct tools and skills for given tasks.

**Test Categories:**
- Skill invocation (image_generation, code_generation, etc.)
- Direct tool usage (web_search, execute_code, browser_navigate)
- No-tool responses (simple greetings, basic questions)

**Success Criteria:** ≥ 85% accuracy

**Example Test Case:**
```json
{
  "id": "skill_image_generate",
  "input": "Create a beautiful picture of mountains at sunset",
  "expected_skill": "image_generation",
  "expected_tool": null,
  "category": "image"
}
```

### 3. Response Quality

Uses the LLM-as-judge pattern to evaluate response quality based on configurable criteria.

**Evaluation Criteria:**
- Helpfulness
- Accuracy
- Clarity
- Safety (refusal of harmful requests)
- Code quality
- Professional tone

**Success Criteria:** ≥ 0.7 average score

**Example Test Case:**
```json
{
  "id": "code_quality",
  "input": "Write a Python function to check if a string is a palindrome",
  "criteria": {
    "correctness": "Function should correctly identify palindromes",
    "code_quality": "Code should be clean, readable, and follow Python conventions",
    "completeness": "Should handle edge cases like empty strings"
  },
  "category": "code"
}
```

## Mock LLM System

The framework uses deterministic mock LLMs for reproducible testing without API calls.

### MockChatModel

Pattern-based mock that returns pre-configured responses:

```python
from evals.mocks import MockChatModel, MockLLMConfig, MockResponse

config = MockLLMConfig(
    responses=[
        MockResponse(
            pattern=r"generate.*image",
            response="I'll generate that image.",
            tool_calls=[{
                "name": "invoke_skill",
                "args": {"skill_id": "image_generation", "params": {...}},
                "id": "call_1"
            }]
        )
    ],
    default_response="I can help with that."
)

mock_llm = MockChatModel(config=config)
```

### MockRouterLLM

Specialized mock for testing the supervisor's routing logic:

```python
from evals.mocks.mock_llm import MockRouterLLM

routing_map = {
    r"comprehensive.*research": "research",
    r"analyze.*csv|excel": "data",
    r"hello|how are you": "chat",
}

mock_router = MockRouterLLM(routing_map=routing_map)
```

## Evaluators

### RoutingEvaluator

```python
from evals.evaluators import RoutingEvaluator

evaluator = RoutingEvaluator()
result = evaluator.evaluate(
    expected_agent="research",
    actual_agent="chat",
    query="Write a comprehensive research paper"
)

# result.score = 0.0 (incorrect routing)
# result.comment = "Expected: research, Got: chat"
```

### ToolSelectionEvaluator

```python
from evals.evaluators import ToolSelectionEvaluator

evaluator = ToolSelectionEvaluator()
result = evaluator.evaluate(
    expected_skill="image_generation",
    expected_tool=None,
    tool_calls=[{
        "name": "invoke_skill",
        "args": {"skill_id": "image_generation"},
        "id": "call_1"
    }]
)

# result.score = 1.0 (correct skill selected)
```

### ResponseQualityEvaluator

Uses LLM-as-judge pattern for quality assessment:

```python
from evals.evaluators import ResponseQualityEvaluator

evaluator = ResponseQualityEvaluator(judge_llm=mock_judge)
result = await evaluator.evaluate_async(
    query="Explain quantum computing",
    response="Quantum computing uses qubits...",
    criteria={
        "accuracy": "Should be technically accurate",
        "clarity": "Should be understandable to beginners"
    }
)

# result.score = 0.85
# result.metadata = {"strengths": [...], "weaknesses": [...]}
```

## LangSmith Integration

The evaluators are compatible with LangSmith for tracking and analysis:

```python
from langsmith.evaluation import evaluate
from evals.evaluators import routing_accuracy_evaluator

# Run evaluation with LangSmith tracking
results = evaluate(
    lambda x: my_agent.run(x["query"]),
    data="routing_test_dataset",
    evaluators=[routing_accuracy_evaluator],
)
```

Enable LangSmith tracking:
```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=your_key
make eval-langsmith
```

## Adding New Test Cases

### Adding Routing Test Cases

Edit `backend/evals/datasets/routing.json`:

```json
{
  "test_cases": [
    {
      "id": "unique_test_id",
      "input": "User query to test",
      "expected_agent": "task|research",
      "category": "general|code|image|research|data",
      "description": "What this test validates"
    }
  ]
}
```

### Adding Tool Selection Test Cases

Edit `backend/evals/datasets/tool_selection.json`:

```json
{
  "test_cases": [
    {
      "id": "unique_test_id",
      "input": "User query to test",
      "expected_skill": "skill_id or null",
      "expected_tool": "tool_name or null",
      "category": "image|code|writing|search|general",
      "description": "What this test validates"
    }
  ]
}
```

### Adding Response Quality Test Cases

Edit `backend/evals/datasets/response_quality.json`:

```json
{
  "test_cases": [
    {
      "id": "unique_test_id",
      "input": "User query to test",
      "criteria": {
        "criterion_name": "Description of what to evaluate"
      },
      "category": "code|writing|safety|general"
    }
  ]
}
```

## Success Thresholds

| Metric | Threshold | Description |
|--------|-----------|-------------|
| Routing Accuracy | ≥ 90% | Correct agent selection |
| Tool Selection | ≥ 85% | Correct tool/skill usage |
| Response Quality | ≥ 0.7 | Average quality score |

## Best Practices

1. **Keep tests deterministic** - Use mock LLMs for reproducible results
2. **Cover edge cases** - Include ambiguous queries and boundary conditions
3. **Balance categories** - Ensure test coverage across all agent types
4. **Update thresholds** - Adjust as the system improves
5. **Run before deployment** - Include evals in CI/CD pipeline
6. **Track trends** - Use LangSmith to monitor quality over time

## Troubleshooting

### Tests Failing Due to Pattern Mismatch

Update patterns in `conftest.py`:

```python
routing_map = {
    r"your.*pattern": "expected_agent",
    # Add more specific patterns first
}
```

### LangSmith Connection Issues

Verify environment variables:
```bash
echo $LANGCHAIN_TRACING_V2
echo $LANGCHAIN_API_KEY
```

### Mock LLM Not Matching

Check pattern specificity - more specific patterns should come before generic ones in the routing map.
