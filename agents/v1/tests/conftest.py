from unittest.mock import Mock
import pytest
from langchain_core.messages import AIMessage, HumanMessage

@pytest.fixture
def sample_messages():
    """Fixture providing sample message history"""
    return [
        AIMessage(content="Is it an animal?"),
        HumanMessage(content="yes"),
        AIMessage(content="Is it a dog?"),
        HumanMessage(content="yes")
    ] 


@pytest.fixture
def mock_config():
    """Create mock config with LLMs"""
    mock_host_llm = Mock()
    mock_guesser_llm = Mock()
    return {
        "configurable": {
            "host_llm": mock_host_llm,
            "guesser_llm": mock_guesser_llm,
            "max_questions": 20,
        }
    }
