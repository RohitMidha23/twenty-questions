from unittest.mock import Mock
import pytest
from langchain_core.messages import AIMessage, HumanMessage


@pytest.fixture
def sample_messages():
    """Fixture providing sample message history"""
    return [
        AIMessage(content="Is it an animal?"),
        HumanMessage(content="Yes"),
        AIMessage(content="Is it a mammal?"),
        HumanMessage(content="Yes"),
        AIMessage(content="Does it bark?"),
        HumanMessage(content="Yes"),
    ]


@pytest.fixture
def mock_config():
    """Create mock config with LLMs"""
    return {
        "configurable": {
            "host_llm": Mock(),
            "guesser_recommender_llm": Mock(),
            "guesser_evaluator_llm": Mock(),
            "max_questions": 20,
        }
    }
