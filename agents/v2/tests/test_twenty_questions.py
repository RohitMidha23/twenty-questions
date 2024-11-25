from unittest.mock import Mock
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END

from agents.v2.models import (
    GuesserQuestion,
    HostResponse,
    PossibleGuesses,
    GuessOrQuestion,
    YesNoResponse,
)
from agents.v2.nodes import host_node, guesser_node
from agents.v2.state import GameState

# Test data
HOST_RESPONSES = {
    "Is it an animal?": HostResponse(response=YesNoResponse.YES),
    "Is it a dog?": HostResponse(response=YesNoResponse.YES),
}

RECOMMENDER_OUTPUT = PossibleGuesses(
    guesses=["dog", "cat", "bird"],
    questions=["Is it a mammal?", "Does it bark?", "Is it a pet?"],
)

EVALUATOR_OUTPUTS = {
    "question": GuessOrQuestion(
        choice="question", question="Is it a mammal?", guess="", analysis=""
    ),
    "guess": GuessOrQuestion(choice="guess", question="", guess="dog", analysis=""),
}


def test_host_node_initial_state(mock_config):
    """Test the host node's initial state (choosing random topic)"""
    state = GameState(question_count=0, messages=[])
    updated_state = host_node(state, mock_config)

    assert updated_state["question_count"] == 0
    assert type(updated_state["topic"]) == str
    assert len(updated_state["topic"]) > 0
    assert updated_state["next"] == "guesser"


def test_host_node_with_provided_topic(mock_config):
    """Test the host node when topic is provided in config"""
    mock_config["configurable"]["topic"] = "dog"

    state = GameState(question_count=0, messages=[])
    updated_state = host_node(state, mock_config)

    assert updated_state["topic"] == "dog"
    assert updated_state["next"] == "guesser"


def test_host_node_correct_guess(mock_config):
    """Test the host node when guesser makes correct guess"""

    state = GameState(
        question_count=1,
        guesser_question=GuesserQuestion(question="Is it a dog?"),
        messages=[AIMessage(content="Is it a dog?")],
        topic="dog",
    )

    updated_state = host_node(state, mock_config)

    assert updated_state["next"] == END
    assert updated_state["correct_guess"] == True
    assert len(updated_state["messages"]) == 1
    assert isinstance(updated_state["messages"][0], HumanMessage)
    assert updated_state["messages"][0].content == "Correct guess!"


def test_host_node_max_questions_reached(mock_config):
    """Test the host node when max questions limit is reached"""
    mock_config["configurable"].update({"topic": "dog", "max_questions": 20})

    state = GameState(
        question_count=20,
        guesser_question=GuesserQuestion(question="Is it a mammal?"),
        messages=[AIMessage(content="Is it a mammal?")],
    )

    updated_state = host_node(state, mock_config)

    assert updated_state["next"] == END
    assert updated_state["correct_guess"] == False
    assert updated_state["error"] == "Max questions reached!"


def test_guesser_node_asking_question(mock_config):
    """Test the guesser node when evaluator decides to ask a question"""
    mock_config["configurable"].update(
        {
            "max_questions": 20,
            "guesser_recommender_llm": Mock(),
            "guesser_evaluator_llm": Mock(),
        }
    )

    # Setup mock returns
    mock_config["configurable"][
        "guesser_recommender_llm"
    ].invoke.return_value = RECOMMENDER_OUTPUT
    mock_config["configurable"]["guesser_evaluator_llm"].invoke.return_value = (
        EVALUATOR_OUTPUTS["question"]
    )

    state = GameState(question_count=0, messages=[HumanMessage(content="Let's play!")])

    updated_state = guesser_node(state, mock_config)

    assert updated_state["question_count"] == 1
    assert updated_state["guesser_question"].question == "Is it a mammal?"
    assert len(updated_state["messages"]) == 1
    assert isinstance(updated_state["messages"][0], AIMessage)


def test_guesser_node_making_guess(mock_config):
    """Test the guesser node when evaluator decides to make a guess"""
    mock_config["configurable"].update(
        {
            "max_questions": 20,
            "guesser_recommender_llm": Mock(),
            "guesser_evaluator_llm": Mock(),
        }
    )

    # Setup mock returns
    mock_config["configurable"][
        "guesser_recommender_llm"
    ].invoke.return_value = RECOMMENDER_OUTPUT
    mock_config["configurable"]["guesser_evaluator_llm"].invoke.return_value = (
        EVALUATOR_OUTPUTS["guess"]
    )

    state = GameState(
        question_count=5,
        messages=[
            AIMessage(content="Is it a mammal?"),
            HumanMessage(content="Yes"),
            AIMessage(content="Does it bark?"),
            HumanMessage(content="Yes"),
        ],
    )

    updated_state = guesser_node(state, mock_config)

    assert updated_state["question_count"] == 6
    assert updated_state["guesser_question"].question == "Is it a dog?"
    assert len(updated_state["messages"]) == 1
    assert isinstance(updated_state["messages"][0], AIMessage)


def test_host_node_regular_question(mock_config):
    """Test the host node handling a regular question (not a guess)"""
    mock_config["configurable"].update(
        {"topic": "dog", "max_questions": 20, "host_llm": Mock()}
    )

    mock_config["configurable"]["host_llm"].invoke.return_value = HOST_RESPONSES[
        "Is it an animal?"
    ]

    state = GameState(
        question_count=1,
        guesser_question=GuesserQuestion(question="Is it an animal?"),
        messages=[AIMessage(content="Is it an animal?")],
        topic="dog",
    )

    updated_state = host_node(state, mock_config)

    assert updated_state["next"] == "guesser"
    assert updated_state["host_response"] == HOST_RESPONSES["Is it an animal?"]
    assert len(updated_state["messages"]) == 1
    assert isinstance(updated_state["messages"][0], HumanMessage)
    assert updated_state["messages"][0].content == "Yes"
