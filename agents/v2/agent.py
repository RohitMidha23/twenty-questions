from langchain_core.runnables.config import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph


from agents.v2.nodes import guesser_node, host_node, should_continue
from agents.v2.state import GameState
from agents.v2.prompts import (
    HOST_PROMPT_v1,
    GUESSER_RECOMMENDER_PROMPT_v1,
    GUESSER_EVALUATOR_PROMPT_v2,
)
from agents.v2.models import HostResponse, PossibleGuesses, GuessOrQuestion

from dotenv import load_dotenv

load_dotenv()


def get_game_graph_v2() -> CompiledStateGraph:
    graph = StateGraph(GameState)
    graph.add_node("host", host_node)
    graph.add_node("guesser", guesser_node)
    graph.add_edge(START, "host")
    graph.add_conditional_edges("host", should_continue)
    graph.add_edge("guesser", "host")
    return graph.compile()


def get_sample_llms_v2(llm):

    host_llm = HOST_PROMPT_v1 | llm.with_structured_output(HostResponse).with_retry(
        retry_if_exception_type=(Exception,),
        wait_exponential_jitter=True,
        stop_after_attempt=2,
    )
    guesser_recommender_llm = (
        GUESSER_RECOMMENDER_PROMPT_v1 | llm.with_structured_output(PossibleGuesses)
    ).with_retry(
        retry_if_exception_type=(Exception,),
        wait_exponential_jitter=True,
        stop_after_attempt=2,
    )
    guesser_evaluator_llm = GUESSER_EVALUATOR_PROMPT_v2 | llm.with_structured_output(
        GuessOrQuestion
    ).with_retry(
        retry_if_exception_type=(Exception,),
        wait_exponential_jitter=True,
        stop_after_attempt=2,
    )
    return host_llm, guesser_recommender_llm, guesser_evaluator_llm


def main():
    base_llm = ChatOpenAI(model="gpt-4o-mini", temperature=1)
    host_llm, guesser_recommender_llm, guesser_evaluator_llm = get_sample_llms_v2(base_llm)
    graph = get_game_graph_v2()
    config = RunnableConfig(
        configurable={
            "host_llm": host_llm,
            "guesser_recommender_llm": guesser_recommender_llm,
            "guesser_evaluator_llm": guesser_evaluator_llm,
            "max_questions": 20,
        },
        # we can add recursion limit as a fallback
        recursion_limit=50,
    )

    events = graph.stream(
        {
            "question_count": 0,
            "messages": [],
        },
        config,
    )
    for event in events:
        print(event)
        print("----")


if __name__ == "__main__":
    main()
