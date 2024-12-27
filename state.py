# state.py

from typing import Annotated, List, Optional
from typing_extensions import TypedDict # type: ignore
from datetime import datetime as Datetime
from langgraph.graph.message import add_messages # type: ignore

# Define Search Query Class
class SearchQuery(TypedDict):
    queries: List[str]
    start_date: Optional[Datetime]
    end_date: Optional[Datetime]

# Define Paper Class
class Paper(TypedDict):
    title: str
    summary: str
    link: Optional[str]
    id: Optional[str]
    published: Optional[Datetime]
    updated: Optional[Datetime]
    authors: Optional[List[str]]

# Define Arxiv Paper Class
class ArxivPaper(TypedDict):
    paper: Paper
    keywords: Optional[List[str]]
    sources: Optional[List[Paper]]
    relevant: Optional[bool]

# Define Refine Agent State
class RefineAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    mode: str
    done: bool
    iterations: int

# Define Search Agent State
class SearchAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    search_query: Optional[SearchQuery]
    papers: Optional[List[ArxivPaper]]
    papers_id: Optional[List[str]]
    iterations: int
    done: bool

# Define the Main State
class State(TypedDict):
    refine_agent: RefineAgentState
    search_agent: SearchAgentState
    end: bool

# Initialize Agent State
def initialize_agent_state(mode: str) -> State:
    refine_agent_state = RefineAgentState(
        {
            "messages": [],
            "mode": mode,
            "done": False,
            "iterations": 0,
        }
    )

    search_agent_state = SearchAgentState(
        {
            "messages": [],
            "search_query": None,
            "papers": [],
            "papers_id": [],
            "iterations": 0,
            "done": False,
        }
    )

    agent_state = State(
        {
            "refine_agent": refine_agent_state,
            "search_agent": search_agent_state,
            "end": False,
        }
    )

    return agent_state
