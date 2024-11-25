import random
import time

from agents.v1.state import GameState
from agents.v1.models import GuesserQuestion

from langgraph.graph import END
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.config import RunnableConfig


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


def host_node_v1(state: GameState, config: RunnableConfig) -> GameState:
    """
    Host node that takes in the guesser's question and answers it.
    Host can either choose to end the game or continue the game.
    Host messages are added to the conversation history as HumanMessages.

    In v1, the host also checks if the guesser's question is correct - this is a non deterministic check and reduces some amount of reliability.
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
    topic = configuration.get("topic")

    if question_count < max_questions:
        next = "guesser"
    else:
        return {
            "next": END,
            "error": "Max questions reached!",
            "correct_guess": False,
        }

    if question_count == 0 and topic is None:
        # At the very start, choose a random topic if one is not provided
        topic = get_random_topic()
        return {
            "next": next,
            "question_count": 0,
            "topic": topic,
        }
    elif question_count == 0:
        # if the topic is provided, but the question count is 0, then we need to ask the guesser's question
        return {
            "next": "guesser",
            "topic": topic,
            "question_count": 0,
        }

    # At the other steps, answer the guesser's question
    if guesser_question:
        time.sleep(1)  # for avoiding rate limiting
        host_response = host_llm.invoke(
            {"topic": topic, "question": guesser_question.question}
        )
    else:
        return {
            "next": END,
            "error": "Guesser's question is not provided!",
        }
    if host_response.correct_guess:
        print("Correct guess!")
        return {
            "next": END,
            "host_response": host_response,
            "messages": [HumanMessage(content=host_response.response)],
            "correct_guess": True,
        }
    return {
        "next": next,
        "host_response": host_response,
        "messages": [HumanMessage(content=host_response.response)],
    }


def guesser_node_v1(state: GameState, config: RunnableConfig) -> GameState:
    """
    Guesser node that takes in the host's response and asks a question.
    Guesser messages are added to the conversation history as AIMessages.
    Args:
        state (GameState): Current state of the game.
        config (RunnableConfig): Runtime configuration arguments.

    Returns:
        GameState: Updated state after the node is executed.
    """
    question_count = state.get("question_count")
    configuration = config.get("configurable", {})
    max_questions = configuration.get("max_questions")
    guesser_llm = configuration.get("guesser_llm")

    remaining_questions = max_questions - question_count
    question: GuesserQuestion = guesser_llm.invoke(
        {
            "messages": state.get("messages"),
            "question_count": remaining_questions,
        }
    )
    return {
        "guesser_question": question,
        "messages": [AIMessage(content=question.question)],
        "question_count": question_count + 1,
    }


def should_continue(state: GameState) -> str:
    return state.get("next")
