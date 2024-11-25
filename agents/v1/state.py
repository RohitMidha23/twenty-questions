from langgraph.graph import MessagesState
from pydantic import BaseModel


class GameState(MessagesState):
    question_count: int = 0
    topic: str = ""  # this is the topic that the guesser is trying to guess
    next: str = ""  # this is the next node to be executed
    host_response: BaseModel = None
    guesser_question: BaseModel = None
    correct_guess: bool = False  # useful for evaluation
    error: str = ""
