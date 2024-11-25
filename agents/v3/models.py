from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from enum import Enum


class YesNoResponse(str, Enum):
    YES = "Yes"
    NO = "No"


class HostResponse(BaseModel):
    response: YesNoResponse = Field(
        ...,
        description="Host's answer to the Guesser's question. Yes if the question is about the topic, No otherwise.",
    )


class GuesserQuestion(BaseModel):
    question: str = Field(..., description="Guesser's question to the Host.")


class RecommenderDecision(BaseModel):
    """Recommender decides whether to guess or question and provides possible candidates"""

    decision: Literal["guess", "question"] = Field(
        ..., description="Whether to make a guess or ask a question"
    )
    possible_candidates: List[str] = Field(
        ..., description="Current list of possible candidates for the answer"
    )
    confidence_scores: Optional[dict[str, float]] = Field(
        default=None, description="Confidence scores for each candidate"
    )
    reasoning: str = Field(
        ..., description="Reasoning behind the decision and candidate list"
    )


class QuestionGenerator(BaseModel):
    """Generates a question that aims to split the candidate pool"""

    question: str = Field(
        ...,
        description="Question that should help eliminate roughly half of candidates",
    )
    expected_elimination: List[str] = Field(
        ..., description="Candidates that would be eliminated by a NO answer"
    )
    expected_retention: List[str] = Field(
        ..., description="Candidates that would be retained by a YES answer"
    )


class QuestionEvaluation(BaseModel):
    """Evaluator's assessment of the proposed question"""

    is_good_question: bool = Field(
        ...,
        description="Whether the question is effective for eliminating candidates and asking the host",
    )
    reasoning: str = Field(..., description="Reasoning behind the evaluation")
    suggested_improvement: Optional[str] = Field(
        None, description="If question isn't good, suggestion for improvement"
    )
