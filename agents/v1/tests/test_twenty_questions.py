from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END

from agents.v1.models import GuesserQuestion, HostResponse_v3, YesNoResponse
from agents.v1.nodes import guesser_node_v1, host_node_v1
from agents.v1.state import GameState

HOST_RESPONSES = {
    "Is it an animal?": HostResponse_v3(
        response=YesNoResponse.YES,
        correct_guess=False,
        analysis="A dog is an animal but the guesser has not yet guessed correctly.",
    ),
    "Is it a dog?": HostResponse_v3(
        response=YesNoResponse.YES,
        correct_guess=True,
        analysis="The guesser has correctly guessed that it is a dog.",
    ),
}

GUESSER_QUESTIONS = {"initial": "Is it an animal?", "follow_up": "Is it a dog?"}


def test_host_node_initial_response(mock_config):
    """Test the host node's initial response (choosing random topic)"""
    question = "Is it an animal?"
    mock_config["configurable"]["host_llm"].invoke.return_value = HOST_RESPONSES[question]

    state = GameState(question_count=0, messages=[AIMessage(content=question)])
    updated_state = host_node_v1(state, mock_config)
    
    assert updated_state["question_count"] == 0
    assert type(updated_state["topic"]) == str and len(updated_state["topic"]) > 0
    assert updated_state["next"] == "guesser"

def test_host_node_correct_guess(mock_config):
    """Test the host node when guesser makes correct guess"""
    question = "Is it a dog?"
    mock_config["configurable"].update({
        "topic": "dog",
        "max_questions": 20
    })
    mock_config["configurable"]["host_llm"].invoke.return_value = HOST_RESPONSES[question]

    state = GameState(
        question_count=1,
        guesser_question=GuesserQuestion(question=question),
        messages=[AIMessage(content=question)]
    )
    
    updated_state = host_node_v1(state, mock_config)
    
    assert updated_state["next"] == END
    assert updated_state["correct_guess"] == True
    assert len(updated_state["messages"]) == 1
    assert isinstance(updated_state["messages"][0], HumanMessage)

def test_host_node_max_questions_reached(mock_config):
    """Test the host node when max questions limit is reached"""
    mock_config["configurable"].update({
        "topic": "dog",
        "max_questions": 20
    })
    
    state = GameState(
        question_count=20,
        guesser_question=GuesserQuestion(question="Is it a dog?"),
        messages=[AIMessage(content="Is it a dog?")]
    )
    
    updated_state = host_node_v1(state, mock_config)
    
    assert updated_state["next"] == END
    assert updated_state["correct_guess"] == False
    assert updated_state["error"] == "Max questions reached!"

def test_host_node_missing_question(mock_config):
    """Test the host node when guesser question is missing"""
    mock_config["configurable"].update({
        "topic": "dog",
        "max_questions": 20
    })
    
    state = GameState(
        question_count=1,
        messages=[AIMessage(content="Is it a dog?")]
    )
    
    updated_state = host_node_v1(state, mock_config)
    
    assert updated_state["next"] == END
    assert updated_state["error"] == "Guesser's question is not provided!"

def test_guesser_node(mock_config):
    """Test the guesser node's question generation"""
    mock_config["configurable"].update({
        "max_questions": 20
    })
    
    expected_question = GuesserQuestion(question=GUESSER_QUESTIONS["initial"])
    mock_config["configurable"]["guesser_llm"].invoke.return_value = expected_question
    
    state = GameState(
        question_count=0,
        messages=[HumanMessage(content="Let's play 20 questions!")]
    )
    
    updated_state = guesser_node_v1(state, mock_config)
    
    assert updated_state["question_count"] == 1
    assert updated_state["guesser_question"] == expected_question
    assert len(updated_state["messages"]) == 1
    assert isinstance(updated_state["messages"][0], AIMessage)
    assert updated_state["messages"][0].content == GUESSER_QUESTIONS["initial"]

    

