import os
from dotenv import load_dotenv # type: ignore

from rich.console import Console # type: ignore
from rich.markdown import Markdown # type: ignore
from rich.prompt import Confirm # type: ignore

from langchain_openai import ChatOpenAI # type: ignore
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage # type: ignore

from typing import Annotated, List, Optional

from typing_extensions import TypedDict # type: ignore

from langgraph.graph import StateGraph, START, END # type: ignore
from langgraph.graph.message import add_messages # type: ignore
from langgraph.checkpoint.memory import MemorySaver # type: ignore

from IPython.display import Image, display # type: ignore
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles # type: ignore

import io
from PIL import Image # type: ignore

from datetime import date as Date

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

DEFAULT_MODE = 'human'
DEFAULT_CONFIG = {"configurable": {"thread_id": "1"}}

class SearchQuery(TypedDict):
    query: str
    queries: Optional[List[str]]
    start_date: Optional[Date]
    end_date: Optional[Date]

class Paper(TypedDict):
    title: str
    summary: str
    link: Optional[str]
    date: Optional[Date]
    authors: Optional[List[str]]

class ArxivPaper(TypedDict):
    paper: Paper
    keywords: Optional[List[str]]
    sources: Optional[List[Paper]]

class RefineAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    mode: str
    done: bool
    iterations: int

class SearchAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    search_query: Optional[SearchQuery]
    papers: Optional[List[ArxivPaper]]
    done: bool

class State(TypedDict):
    refine_agent: RefineAgentState
    search_agent: SearchAgentState
    end: bool

def refine_agent(state: State, llm: ChatOpenAI, console: Console):
    print("RefineAgent")
    system_prompt = """
    If the query is not about a research topic ignore the following.
    You are a research assistant helping the user refine their research query. Your task is to identify areas of ambiguity or insufficient detail in the user's query and ask clarifying questions to help them focus their research. Here's how you should approach it:

    1. Identify the key aspects of the research query that are unclear or vague.
    2. Suggest specific details that need clarification, such as:
    - The exact research question or hypothesis.
    - The context or scope of the research (e.g., geographic region, time period, population).
    - The type of research approach (e.g., qualitative, quantitative, or theoretical).
    - Any other missing information that could help define the research direction more clearly.
    3. Ask open-ended questions to prompt the user to provide more specific details.
    4. Make sure to keep the user's original research intent intact while helping them narrow down their focus.

    Your goal is not to find papers. Only to refine the query with the user.

    If you feel that the refinement process is over or that the user has nothing else to add, simply return 'done'.
    """

    system_message = SystemMessage(system_prompt)
    while True:

        user_input = console.input("[bold blue]Ask your question (type 'exit' to quit, 'again' to generate a new answer, 'discard' to discard your previous input): [/bold blue]")
        
        if user_input.lower() == "exit":
            console.print("[bold green]Goodbye![/bold green]")
            state["end"] = True
            return state
        elif user_input.lower() == "discard":
            state["refine_agent"]["messages"].pop()
            state["refine_agent"]["messages"].pop()

        else:
            try:
                if user_input.lower() == "again":
                    state["refine_agent"]["messages"].pop()
                else:
                    user_message = HumanMessage(content=user_input)
                    state["refine_agent"]["messages"].append(user_message)
                
                if state["refine_agent"]["iterations"] == 3:
                    if Confirm.ask("\n[yellow]End query refinement?[/yellow]"):
                        new_state = RefineAgentState(
                            {
                                "messages": state["refine_agent"]["messages"],
                                "mode": state["refine_agent"]["mode"],
                                "done": True,
                                "iterations": 0,
                            }
                        )
                        return {"refine_agent": new_state, "search_agent": state["search_agent"]}
                    else:
                        state["refine_agent"]["iterations"] = 0
                
                response = None                    
                while response is None or not check_answer_quality(llm, system_message, response):
                    print("Refining...")
                    response = llm.invoke(state["refine_agent"]["messages"] + [system_message])
                    response_message = response.content

                if "done" in response_message.lower():
                    new_state = RefineAgentState(
                        {
                            "messages": state["refine_agent"]["messages"],
                            "mode": state["refine_agent"]["mode"],
                            "done": True,
                            "iterations": 0,
                        }
                    )
                    return {"refine_agent": new_state, "search_agent": state["search_agent"]}
                
                state["refine_agent"]["messages"].append(response)

                console.print(Markdown(response_message))

                state["refine_agent"]["iterations"] += 1
            
            except Exception as e:
                console.print(f"[bold red]An error occurred: {e}[/bold red]")

def build_query(state: State, llm: ChatOpenAI) -> State:
    print("BuildQueryAgent")

    system_prompt = """
    You are an AI agent specialized in crafting SearchQuery objects from conversations between a user and an LLM.
    The SearchQuery object is defined as follows:
    class SearchQuery(TypedDict):
        query: str # The main query
        queries: Optional[List[str]] # Additional augmented / expanded queries
        start_date: Optional[Date] # The oldest date allowed during our researches in YYYY-MM-DD format, if mentioned (i.e., Papers should not be older)
        end_date: Optional[Date] # The most recent date allowed during our researched in YYYY-MM-DD format, if mentioned (i.e., Papers should not be more recent)
    
    Format your response as a JSON object with these fields:
        - query (str): The main query ;
        - queries (Optional[List[str]]): Additional augmented / expanded queries ;
        - start_date (Optional[str]): The oldest date allowed during our researches in YYYY-MM-DD format, if mentioned (i.e., Papers should not be older) ;
        - end_date (Optional[str]): The most recent date allowed during our researched in YYYY-MM-DD format, if mentioned (i.e., Papers should not be more recent) ;
    
    Include only fields that are clearly specified in the conversation.
    Your output should be the raw JSON object and not its markdown representation (i.e., ```json ...```)
    """

    system_message = SystemMessage(system_prompt)

    try:
        response = llm.invoke(state["refine_agent"]["messages"] + [system_message])
        raw_object = response.content
        raw_object = raw_object.replace("null", "None").replace("```json", "").replace("```", "")
        search_query: SearchQuery = eval(raw_object)

        if not search_query.get("query"):
            raise ValueError("Search query must contain a main query")
        if search_query.get("start_date"):
            year, month, day = map(int, search_query["start_date"].split("-"))
            search_query["start_date"] = Date(year, month, day)
        if search_query.get("end_date"):
            year, month, day = map(int, search_query["end_date"].split("-"))
            search_query["end_date"] = Date(year, month, day)

        state["search_agent"]["search_query"] = search_query
    except Exception as e:
        #TODO
        print(e)

    return state

def check_answer_quality(llm: ChatOpenAI, initial_system_message: SystemMessage, llm_output: AIMessage):
    print("Checking Answer")
    system_prompt = f"""
    Determine if this output: {llm_output.content} aligns with the initial SystemMessage: {initial_system_message.content}.
    If it aligns, return 'continue', otherwise return 'redo'.
    """

    system_message = SystemMessage(system_prompt)

    response = llm.invoke([system_message])

    if "continue" in response.content.lower():
        return True
    
    return False
    
    
def search_agent(state: State, llm: ChatOpenAI):
    #TODO
    print("SearchAgent")
    print(state["search_agent"])
    return state

def build_answer(state: State, llm: ChatOpenAI):
    #TODO
    print("BuildAnswerAgent")
    return state

def build_graph(llm: ChatOpenAI, memory: MemorySaver, console: Console):
    workflow = StateGraph(State)

    workflow.add_node("RefineAgent", lambda s: refine_agent(s, llm, console))
    workflow.add_node("BuildQueryAgent", lambda s: build_query(s, llm))

    workflow.add_node("SearchAgent", lambda s: search_agent(s, llm))

    workflow.add_node("BuildAnswerAgent", lambda s: build_answer(s, llm))

    workflow.add_edge(START, "RefineAgent")

    workflow.add_conditional_edges(
        "RefineAgent", 
        lambda s: END if s["end"] else "BuildQueryAgent",
        {
            "BuildQueryAgent": "BuildQueryAgent",
            END: END,
        }
    )

    workflow.add_edge("BuildQueryAgent", "SearchAgent")

    workflow.add_edge("SearchAgent", "BuildAnswerAgent")

    workflow.add_edge("BuildAnswerAgent", END)

    return workflow.compile(checkpointer=memory)
    
def show_graph(workflow, console: Console):
    try:
        # Assuming workflow.get_graph().draw_mermaid_png() returns a PNG image in bytes
        image_bytes = workflow.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)

        # Create an image object from the byte data
        image = Image.open(io.BytesIO(image_bytes))

        # Display the image
        image.show()
    except Exception as e:
        console.print(f"Encountered error {e}.")
    
def initialize_agent_state(mode: str):
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
            "papers": None,
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
def main():
    load_dotenv()
    
    console = Console()

    llm = ChatOpenAI(model=OPENAI_MODEL)

    # mode = console.input("[bold blue]Do you want to work alongside the LLM during its research? (type 'auto' for the LLM to work alone, otherwise type 'human'): [/bold blue]")
    # while mode not in ["auto", "human"]:
    #     console.print("[bold red]Invalid input. Please try again.[/bold red]")
    #     mode = console.input("[bold blue]Do you want to work alongside the LLM during its research? (type 'auto' for the LLM to work alone, otherwise type 'human'): [/bold blue]")
    
    # TODO: For now the mode is 'human' by defeault
    mode = DEFAULT_MODE

    workflow = build_graph(llm, MemorySaver(), console)

    config = DEFAULT_CONFIG

    state = initialize_agent_state(mode)
    
    if Confirm.ask("\n[yellow]Display the graph?[/yellow]"):
        show_graph(workflow, console)
    try: 
        workflow.invoke(state, config, stream_mode="values")
    except Exception as e:
        console.print(f"[bold red]An error occurred: {e}[/bold red]")
    return

if __name__ == "__main__":
    main()