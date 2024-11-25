from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Optional


class YesNoResponse(Enum):
    YES = "yes"
    NO = "no"


class Topic(BaseModel):
    topic: str = Field(..., description="The topic that the host has chosen")


class GuesserQuestion(BaseModel):
    question: str = Field(..., description="Question asked by the guesser")


class HostResponse_v1(BaseModel):
    response: YesNoResponse = Field(
        ...,
        description="Host's answer to the Guesser's question. Yes if the question is about the topic, No otherwise.",
    )
    correct_guess: bool = Field(
        ..., description="Indicates if the Guesser's guess is correct."
    )


HostResponse_v2 = HostResponse_v1


class HostResponse_v3(BaseModel):
    response: YesNoResponse = Field(
        ...,
        description="Host's answer to the Guesser's question. Yes if the question is about the topic, No otherwise.",
    )
    correct_guess: bool = Field(
        ..., description="Indicates if the Guesser's guess is correct."
    )
    analysis: str = Field(..., description="Explanation of why it is a correct guess.")
