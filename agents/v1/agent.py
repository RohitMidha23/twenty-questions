from langchain_core.runnables.config import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents.v1.nodes import guesser_node_v1, host_node_v1, should_continue
from agents.v1.prompts import GUESSER_PROMPT_v1, HOST_PROMPT_v1
from agents.v1.state import GameState
from agents.v1.models import GuesserQuestion, HostResponse_v1

from dotenv import load_dotenv

load_dotenv()


def get_game_graph_v1() -> CompiledStateGraph:
    graph = StateGraph(GameState)
    graph.add_node("host", host_node_v1)
    graph.add_node("guesser", guesser_node_v1)
    graph.add_edge(START, "host")
    graph.add_conditional_edges("host", should_continue)
    graph.add_edge("guesser", "host")
    return graph.compile()


def get_sample_llms_v1():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=1)
    host_llm = HOST_PROMPT_v1 | llm.with_structured_output(HostResponse_v1)
    guesser_llm = GUESSER_PROMPT_v1 | llm.with_structured_output(GuesserQuestion)
    return host_llm, guesser_llm


def main():
    host_llm, guesser_llm = get_sample_llms_v1()

    graph = get_game_graph_v1()

    config = RunnableConfig(
        configurable={
            "host_llm": host_llm,
            "guesser_llm": guesser_llm,
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
