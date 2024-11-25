from langchain_core.runnables.config import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from dotenv import load_dotenv

from agents.v3.nodes import guesser_node, host_node, should_continue
from agents.v3.state import GameState
from agents.v3.prompts import (
    HOST_PROMPT,
    RECOMMENDER_PROMPT,
    QUESTION_GENERATOR_PROMPT,
    EVALUATOR_PROMPT
)
from agents.v3.models import (
    HostResponse,
    RecommenderDecision,
    QuestionGenerator,
    QuestionEvaluation
)

load_dotenv()

def get_game_graph_v3() -> CompiledStateGraph:
    """Create the game graph with binary search approach"""
    graph = StateGraph(GameState)
    
    graph.add_node("host", host_node)
    graph.add_node("guesser", guesser_node)
    
    graph.add_edge(START, "host")
    graph.add_conditional_edges("host", should_continue)
    graph.add_edge("guesser", "host")
    
    return graph.compile()

def get_sample_llms_v3(llm):
    """Initialize LLMs with appropriate prompts and structured outputs"""
    
    # Host LLM
    host_llm = HOST_PROMPT | llm.with_structured_output(HostResponse).with_retry(
        retry_if_exception_type=(Exception,),
        wait_exponential_jitter=True,
        stop_after_attempt=2,
    )
    
    # Recommender LLM - decides whether to guess or question
    recommender_llm = RECOMMENDER_PROMPT | llm.with_structured_output(
        RecommenderDecision
    ).with_retry(
        retry_if_exception_type=(Exception,),
        wait_exponential_jitter=True,
        stop_after_attempt=2,
    )
    
    # Question Generator LLM - creates binary search style questions
    question_generator_llm = QUESTION_GENERATOR_PROMPT | llm.with_structured_output(
        QuestionGenerator
    ).with_retry(
        retry_if_exception_type=(Exception,),
        wait_exponential_jitter=True,
        stop_after_attempt=2,
    )
    
    # Evaluator LLM - assesses question quality
    evaluator_llm = EVALUATOR_PROMPT | llm.with_structured_output(
        QuestionEvaluation
    ).with_retry(
        retry_if_exception_type=(Exception,),
        wait_exponential_jitter=True,
        stop_after_attempt=2,
    )
    
    return host_llm, recommender_llm, question_generator_llm, evaluator_llm

def main():
    base_llm = ChatOpenAI(model="gpt-4", temperature=0.7)
    
    host_llm, recommender_llm, question_generator_llm, evaluator_llm = get_sample_llms_v3(base_llm)

    graph = get_game_graph_v3()
    
    config = RunnableConfig(
        configurable={
            "host_llm": host_llm,
            "recommender_llm": recommender_llm,
            "question_generator_llm": question_generator_llm,
            "evaluator_llm": evaluator_llm,
            "max_questions": 20,
            
        },
        recursion_limit=50,
    )

    # Stream events
    events = graph.stream(
        {
            "question_count": 0,
            "messages": [],
        },
        config,
    )
    
    # Process events
    for event in events:
        print(event)
        print("----")

if __name__ == "__main__":
    main()
