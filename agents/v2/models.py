from typing import List, Optional, Literal

from enum import Enum
from pydantic import BaseModel, Field


class YesNoResponse(str, Enum):
    YES = "Yes"
    NO = "No"


class HostResponse(BaseModel):
    response: YesNoResponse = Field(
        ...,
        description="Host's answer to the Guesser's question. Yes if the question is about the topic, No otherwise.",
    )


class PossibleGuesses(BaseModel):
    """
    Recommender uses the previous questions and history to come up with possible guesses or questions.
    """

    guesses: List[str] = Field(
        ..., description="Possible guesses at the topic, limited to 5."
    )
    questions: List[str] = Field(
        ..., description="Possible questions to the host, limited to 5."
    )


class GuessOrQuestion(BaseModel):
    """
    Evaluator's choice of whether to guess or ask a question.
    """

    choice: Literal["guess", "question"] = Field(
        ..., description="Either 'guess' or 'question'."
    )
    guess: Optional[str] = Field(
        ..., description="Guess at the topic if choice is 'guess'."
    )
    question: Optional[str] = Field(
        ..., description="Question to the host if choice is 'question'."
    )
    analysis: Optional[str] = Field(
        ...,
        description="Explanation on the basis of analysis, as to why the guess or question was chosen to be asked.",
    )


class GuesserQuestion(BaseModel):
    question: str = Field(..., description="Guesser's question to the Host.")
