"""
This file discusses the ways in which we can evaluate the performance of the agents.
"""

from typing import Dict, List, Literal, Type
import time
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from tqdm import tqdm

from agents.v1.agent import get_game_graph_v1, get_sample_llms_v1
from agents.v2.agent import get_game_graph_v2, get_sample_llms_v2

from langchain_core.runnables.config import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


class GameResult(BaseModel):
    topic: str
    correct_guess: bool
    num_questions: int
    error: str | None
    total_time: float
    messages: List[str]


class EvaluationMetrics(BaseModel):
    success_rate: float
    avg_questions_when_correct: float
    avg_time_per_game: float
    error_rate: float


def _get_llm(
    prompt: ChatPromptTemplate,
    structured_output: Type[BaseModel],
    model_name: str = "gemini-1.5-flash",
):
    """
    Simple function to get a LLM for a given prompt.
    Args:
        prompt: The prompt to use.
        structured_output: The structured output to use.
        model_name: The model to use.
    Returns:
        A LLM with structured output and retry logic.
    """
    if not prompt:
        raise ValueError("Prompt is required")

    if "gemini" in model_name:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=1,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
    elif "gpt" in model_name:
        llm = ChatOpenAI(model=model_name, temperature=1)
    elif "claude" in model_name:
        llm = ChatAnthropic(model=model_name, temperature=1)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return prompt | llm.with_structured_output(structured_output).with_retry(
        retry_if_exception_type=(Exception,),
        wait_exponential_jitter=True,
        stop_after_attempt=2,
    )


def _print_metrics(metrics: EvaluationMetrics):
    print(f"Success Rate: {metrics.success_rate:.2%}")
    print(f"Avg Questions When Correct: {metrics.avg_questions_when_correct:.1f}")
    print(f"Avg Time per Game: {metrics.avg_time_per_game:.2f}s")
    print(f"Error Rate: {metrics.error_rate:.2%}")


class TwentyQuestionsEvaluator:
    def __init__(
        self,
        test_topics: List[str],
        max_questions: int = 20,
        num_runs: int = 1,
        config: RunnableConfig = None,
        agent_version: Literal["v1", "v2"] = "v1",
    ):
        self.test_topics = test_topics
        self.max_questions = max_questions
        self.num_runs = num_runs
        self.results: List[GameResult] = []
        self.config = config
        self.agent_version = agent_version

    def evaluate_prompt_combination(
        self,
    ) -> List[GameResult]:
        """Evaluate a specific prompt and LLM combination."""
        results = []

        for topic in tqdm(self.test_topics * self.num_runs):
            start_time = time.time()
            result = self._run_single_game(topic, self.config)
            result.total_time = time.time() - start_time
            results.append(result)

            # Add small delay to avoid rate limiting
            time.sleep(1)

        return results

    def _run_single_game(
        self,
        topic: str,
        config: RunnableConfig,
    ) -> GameResult:
        """Run a single game of 20 questions."""
        if self.agent_version == "v1":
            graph = get_game_graph_v1()
        else:
            graph = get_game_graph_v2()

        config["configurable"]["topic"] = topic

        try:
            events = graph.stream(
                {
                    "question_count": 0,
                    "messages": [],
                },
                config,
            )

            messages = []
            final_event = None
            for event in events:
                if "messages" in event:
                    messages.extend([m.content for m in event["messages"]])
                # print(event)
                final_event = event

            return GameResult(
                topic=topic,
                correct_guess=final_event.get("correct_guess", False),
                num_questions=final_event.get("question_count", self.max_questions),
                error=final_event.get("error"),
                total_time=0,  # Will be set later
                messages=messages,
            )

        except Exception as e:

            return GameResult(
                topic=topic,
                correct_guess=False,
                num_questions=0,
                error=str(e),
                total_time=0,
                messages=[],
            )

    def _compute_metrics(self, results: List[GameResult]) -> EvaluationMetrics:
        """Compute metrics for a set of game results."""
        total_games = len(results)
        successful_games = [r for r in results if r.correct_guess]
        error_games = [r for r in results if r.error is not None and len(r.error) > 0]

        return EvaluationMetrics(
            success_rate=len(successful_games) / total_games,
            avg_questions_when_correct=(
                sum(r.num_questions for r in successful_games) / len(successful_games)
                if successful_games
                else 0
            ),
            avg_time_per_game=sum(r.total_time for r in results) / total_games,
            error_rate=len(error_games) / total_games,
        )

    def run_evaluation(self) -> EvaluationMetrics:
        """Run evaluation and compute metrics."""

        self.results = self.evaluate_prompt_combination()
        metrics = self._compute_metrics(self.results)

        return metrics


def main_v1(test_topics: List[str]):

    host_llm, guesser_llm = get_sample_llms_v1()
    config = RunnableConfig(
        configurable={
            "host_llm": host_llm,
            "guesser_llm": guesser_llm,
            "max_questions": 20,
        },
        recursion_limit=50,
    )

    evaluator = TwentyQuestionsEvaluator(
        test_topics=test_topics,
        max_questions=20,
        num_runs=1,
        agent_version="v1",
        config=config,
    )
    metrics: EvaluationMetrics = evaluator.run_evaluation()

    print("\nEvaluation Results:")
    print("==================")
    _print_metrics(metrics)
    print("==================")


def main_v2(test_topics: List[str]):
    

    base_llm = ChatOpenAI(model="gpt-4o-mini", temperature=1)
    host_llm, guesser_recommender_llm, guesser_evaluator_llm = get_sample_llms_v2(
        base_llm
    )

    config = RunnableConfig(
        configurable={
            "host_llm": host_llm,
            "guesser_recommender_llm": guesser_recommender_llm,
            "guesser_evaluator_llm": guesser_evaluator_llm,
            "max_questions": 20,
        },
        # we can add recursion limit as a fallback
        recursion_limit=50,
    )

    evaluator = TwentyQuestionsEvaluator(
        test_topics=test_topics,
        max_questions=20,
        num_runs=1,  # Run each topic 1 time
        config=config,
        agent_version="v2",
    )

    metrics: EvaluationMetrics = evaluator.run_evaluation()

    print("\nEvaluation Results:")
    print("==================")
    _print_metrics(metrics)
    print("==================")


if __name__ == "__main__":
    # Load test topics from file
    with open("evals/topics.txt", "r") as f:
        test_topics = [line.strip() for line in f.readlines()]
    test_topics = test_topics[:1]  # for testing
    main_v1(test_topics)
    main_v2(test_topics)
