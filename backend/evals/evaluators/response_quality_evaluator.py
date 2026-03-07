"""Response quality evaluator using LLM-as-judge pattern."""

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from evals.evaluators.base import EvaluationResult

# System prompt for the LLM judge
JUDGE_SYSTEM_PROMPT = """You are an expert evaluator assessing AI assistant responses.

Your task is to evaluate a response based on specific criteria and provide a score from 0.0 to 1.0.

Scoring guidelines:
- 1.0: Excellent - Fully meets all criteria with high quality
- 0.8: Good - Meets most criteria well with minor issues
- 0.6: Acceptable - Meets basic criteria but has notable gaps
- 0.4: Below expectations - Has significant issues
- 0.2: Poor - Fails to meet most criteria
- 0.0: Unacceptable - Completely fails or is harmful

Respond with a JSON object containing:
- "score": A number between 0.0 and 1.0
- "reasoning": Brief explanation of the score
- "strengths": List of positive aspects
- "weaknesses": List of areas for improvement

Respond ONLY with the JSON object."""


class ResponseQualityEvaluator:
    """Evaluator for response quality using LLM-as-judge.

    Uses a separate LLM to evaluate response quality based on
    configurable criteria like helpfulness, accuracy, and safety.
    """

    def __init__(self, judge_llm: Any = None):
        """Initialize the response quality evaluator.

        Args:
            judge_llm: LLM to use as judge. If None, uses default.
        """
        self.name = "response_quality"
        self.judge_llm = judge_llm

    def _get_judge_llm(self):
        """Get or create the judge LLM."""
        if self.judge_llm is not None:
            return self.judge_llm

        # Import here to avoid circular imports
        try:
            from app.ai.llm import llm_service
            from app.ai.model_tiers import ModelTier

            return llm_service.get_llm_for_tier(ModelTier.PRO)
        except ImportError:
            raise ValueError(
                "No judge LLM provided and could not import default. "
                "Please provide a judge_llm parameter."
            )

    async def evaluate_async(
        self,
        query: str,
        response: str,
        criteria: dict[str, str],
    ) -> EvaluationResult:
        """Evaluate response quality asynchronously.

        Args:
            query: The original user query
            response: The AI response to evaluate
            criteria: Dict of criterion name -> description

        Returns:
            EvaluationResult with quality score
        """
        import json

        judge = self._get_judge_llm()

        # Build the evaluation prompt
        criteria_text = "\n".join(
            f"- {name}: {description}" for name, description in criteria.items()
        )

        eval_prompt = f"""Evaluate the following AI response based on these criteria:

{criteria_text}

User Query: {query}

AI Response: {response}

Evaluate the response quality."""

        try:
            result = await judge.ainvoke(
                [
                    SystemMessage(content=JUDGE_SYSTEM_PROMPT),
                    HumanMessage(content=eval_prompt),
                ]
            )

            # Parse the JSON response
            content = result.content
            if isinstance(content, str):
                # Clean up potential markdown formatting
                if content.startswith("```"):
                    lines = content.split("\n")
                    content = "\n".join(
                        line for line in lines if not line.startswith("```")
                    ).strip()

                parsed = json.loads(content)
                score = float(parsed.get("score", 0.5))
                reasoning = parsed.get("reasoning", "")
                strengths = parsed.get("strengths", [])
                weaknesses = parsed.get("weaknesses", [])
            else:
                score = 0.5
                reasoning = "Could not parse judge response"
                strengths = []
                weaknesses = []

            return EvaluationResult(
                key=self.name,
                score=score,
                comment=reasoning,
                metadata={
                    "strengths": strengths,
                    "weaknesses": weaknesses,
                    "criteria": criteria,
                },
            )

        except Exception as e:
            return EvaluationResult(
                key=self.name,
                score=0.0,
                comment=f"Evaluation error: {str(e)}",
                metadata={"error": str(e)},
            )

    def evaluate(
        self,
        query: str,
        response: str,
        criteria: dict[str, str],
    ) -> EvaluationResult:
        """Evaluate response quality synchronously.

        Works safely whether or not an event loop is already running.

        Args:
            query: The original user query
            response: The AI response to evaluate
            criteria: Dict of criterion name -> description

        Returns:
            EvaluationResult with quality score
        """
        return _run_async(self.evaluate_async(query, response, criteria))

    async def evaluate_batch_async(
        self,
        results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Evaluate a batch of responses asynchronously.

        Args:
            results: List of dicts with 'query', 'response', and 'criteria'

        Returns:
            Summary statistics including average score and breakdowns
        """
        import asyncio

        if not results:
            return {"average_score": 0.0, "total": 0}

        # Run evaluations concurrently
        tasks = [
            self.evaluate_async(
                query=r["query"],
                response=r["response"],
                criteria=r["criteria"],
            )
            for r in results
        ]
        evaluations = await asyncio.gather(*tasks)

        # Calculate statistics
        scores = [e.score for e in evaluations]
        average_score = sum(scores) / len(scores) if scores else 0.0

        # Category breakdown
        category_stats: dict[str, list[float]] = {}
        for result, evaluation in zip(results, evaluations):
            category = result.get("category", "unknown")
            if category not in category_stats:
                category_stats[category] = []
            category_stats[category].append(evaluation.score)

        category_averages = {
            cat: sum(scores) / len(scores) if scores else 0.0
            for cat, scores in category_stats.items()
        }

        return {
            "average_score": average_score,
            "total": len(evaluations),
            "min_score": min(scores) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0,
            "category_averages": category_averages,
            "evaluations": evaluations,
        }


def _run_async(coro):
    """Run an async coroutine synchronously, safe even inside a running event loop."""
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop — safe to use asyncio.run()
        return asyncio.run(coro)

    # Already inside a running loop — run in a separate thread
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, coro)
        return future.result()


def response_quality_evaluator(
    run: Any,
    example: Any,
) -> EvaluationResult:
    """LangSmith-compatible evaluator function for response quality.

    Args:
        run: LangSmith run object with outputs
        example: LangSmith example with expected outputs

    Returns:
        EvaluationResult for LangSmith
    """
    query = example.inputs.get("query", "")
    response = run.outputs.get("response", "")
    criteria = example.outputs.get("criteria", {})

    evaluator = ResponseQualityEvaluator()
    return _run_async(evaluator.evaluate_async(query, response, criteria))
