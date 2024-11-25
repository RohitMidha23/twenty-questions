import random
import time

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END

from agents.v2.models import GuesserQuestion, PossibleGuesses, GuessOrQuestion
from agents.v2.state import GameState


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
        if topic in guesser_question.question:
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
    Guesser node that takes in the host's response and asks a question.
    Guesser messages are added to the conversation history as AIMessages.

    In this version, we introduce a recommender and evaluator. The recommender comes up with possible guesses and questions and the evaluator chooses the best one.

    Note: The evaluator is not perfect and can sometimes come up with both a guess and a question. In that case, we retry with a stricter prompt.

    Another version of this is to have the recommender and evaluator as two separate agents collaborating.
    We can even model the current interaction as a sub-graph in langgraph.
    Args:
        state (GameState): Current state of the game.
        config (RunnableConfig): Runtime configuration arguments.

    Returns:
        GameState: Updated state after the node is executed.
    """
    question_count = state.get("question_count")
    configuration = config.get("configurable", {})
    max_questions = configuration.get("max_questions")
    recommender_llm = configuration.get("guesser_recommender_llm")
    evaluator_llm = configuration.get("guesser_evaluator_llm")

    remaining_questions = max_questions - question_count
    recommender_output: PossibleGuesses = recommender_llm.invoke(
        {"messages": state.get("messages")}
    )

    evaluator_output: GuessOrQuestion = evaluator_llm.invoke(
        {
            "guesses": recommender_output.guesses,
            "questions": recommender_output.questions,
            "messages": state.get("messages"),
            "question_count": remaining_questions,
            "input": "Come up with either a guess or question based on the analysis.",
        }
    )
    # Convert evaluator output to guesser question
    if evaluator_output.choice == "guess":
        question = GuesserQuestion(question=f"Is it a {evaluator_output.guess}?")
    else:
        question = GuesserQuestion(question=evaluator_output.question)

    return {
        "guesser_question": question,
        "messages": [AIMessage(content=question.question)],
        "question_count": question_count + 1,
    }


def should_continue(state: GameState) -> str:
    return state.get("next")
