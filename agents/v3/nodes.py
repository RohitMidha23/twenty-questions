import random
import time

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END

from agents.v3.models import (
    GuesserQuestion,
    QuestionGenerator,
    QuestionEvaluation,
    RecommenderDecision,
)
from agents.v3.state import GameState


def get_random_topic():
    topics = [
        "apple",
        "banana",
        "cherry",
        "dog",
        "cat",
        "car",
        "house",
        "tree",
        "flower",
        "book",
    ]
    return random.choice(topics)


def host_node(state: GameState, config: RunnableConfig) -> GameState:
    """
    Host node that takes in the guesser's question and answers it.
    Host can either choose to end the game or continue the game.
    Host messages are added to the conversation history as HumanMessages.

    We do a deterministic check in the host node to see if the guesser's question is correct. We currently use hard string matching for this.
    Args:
        state (GameState): Current state of the game.
        config (RunnableConfig): Runtime configuration arguments.

    Returns:
        GameState: Updated state after the node is executed.
    """
    guesser_question = state.get("guesser_question")
    question_count = state.get("question_count")

    configuration = config.get("configurable", {})
    max_questions = configuration.get("max_questions")
    host_llm = configuration.get("host_llm")

    if question_count < max_questions:
        next = "guesser"
    else:
        return {
            "next": END,
            "error": "Max questions reached!",
            "correct_guess": False,
        }
    if question_count == 0:
        # At the very start, check if the topic is provided
        topic = configuration.get("topic")
        if topic is None:
            # if the topic is not provided, then choose a random topic
            topic = get_random_topic()
        # Else, use the provided topic
        return {
            "next": next,
            "question_count": 0,
            "topic": topic,
        }

    topic = state.get("topic")
    # At the other steps, answer the guesser's question
    if guesser_question:
        print("Topic: ", topic)
        print("Guesser question: ", guesser_question.question)
        if topic in guesser_question.question:
            print("Correct guess!")
            return {
                "next": END,
                "messages": [HumanMessage(content="Correct guess!")],
                "correct_guess": True,
            }
        # if it is not a correct guess, then ask the host the question
        # time.sleep(1)  # for avoiding rate limiting
        host_response = host_llm.invoke(
            {"topic": topic, "question": guesser_question.question}
        )
        return {
            "next": next,
            "host_response": host_response,
            "messages": [HumanMessage(content=host_response.response)],
        }
    else:
        return {
            "next": END,
            "error": "Guesser's question is not provided!",
        }


def guesser_node(state: GameState, config: RunnableConfig) -> GameState:
    """
    Enhanced guesser node with binary search approach
    """
    configuration = config.get("configurable", {})
    recommender_llm = configuration.get("recommender_llm")
    question_generator_llm = configuration.get("question_generator_llm")
    evaluator_llm = configuration.get("evaluator_llm")

    # Step 1: Get recommendation
    recommender_output: RecommenderDecision = recommender_llm.invoke(
        {
            "messages": state.get("messages"),
            "candidates": state.get("candidates"),
        }
    )

    if recommender_output.decision == "guess":
        # Make a guess based on highest confidence candidate - if the confidence score is greater than 90%, then make a guess
        best_candidate_index = recommender_output.confidence_scores.index(
            max(recommender_output.confidence_scores)
        )
        best_candidate = recommender_output.possible_candidates[best_candidate_index]
        if recommender_output.confidence_scores[best_candidate_index] > 0.9:
            return {
                "guesser_question": GuesserQuestion(question=f"Is it a {best_candidate}?"),
                "messages": [AIMessage(content=f"Is it a {best_candidate}?")],
                "question_count": state.get("question_count") + 1,
                "candidates": recommender_output.possible_candidates,
            }
    
    # Step 2: Generate binary search question
    question_output: QuestionGenerator = question_generator_llm.invoke(
        {
            "candidates": recommender_output.possible_candidates,
            "messages": state.get("messages"),
            "feedback": "",
        }
    )

    # Step 3: Evaluate question
    evaluation: QuestionEvaluation = evaluator_llm.invoke(
        {
            "candidates": recommender_output.possible_candidates,
            "question": question_output.question,
            "expected_elimination": question_output.expected_elimination,
            "expected_retention": question_output.expected_retention,
            "messages": state.get("messages"),
        }
    )

    if not evaluation.is_good_question:
        # Regenerate question with feedback
        question_output = question_generator_llm.invoke(
            {
                "candidates": recommender_output.possible_candidates,
                "messages": state.get("messages"),
                "feedback": evaluation.suggested_improvement,
            }
        )

    return {
        "guesser_question": GuesserQuestion(question=question_output.question),
        "messages": [AIMessage(content=question_output.question)],
        "question_count": state.get("question_count") + 1,
        "candidates": recommender_output.possible_candidates,
    }


def should_continue(state: GameState) -> str:
    return state.get("next")
