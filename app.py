import os
from dotenv import load_dotenv # type: ignore

from rich.console import Console # type: ignore
from rich.markdown import Markdown # type: ignore
from rich.prompt import Confirm # type: ignore


from langchain_openai import ChatOpenAI # type: ignore
from langchain_core.messages import HumanMessage, SystemMessage # type: ignore


from typing import Annotated

from typing_extensions import TypedDict # type: ignore

from langgraph.graph import StateGraph, START, END # type: ignore
from langgraph.graph.message import add_messages # type: ignore
from langgraph.checkpoint.memory import MemorySaver # type: ignore
from langgraph.prebuilt import ToolNode # type: ignore
from langchain_core.tools import tool # type: ignore


from IPython.display import Image, display # type: ignore
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles # type: ignore

import io
from PIL import Image # type: ignore

class State(TypedDict):
    messages: Annotated[list, add_messages]
    mode: str
    done: bool

def refine_auto(state: State, llm: ChatOpenAI):
    #TODO
    return state

def route_refine(state: State, llm: ChatOpenAI):
    routing_prompt = """
    You are assisting with a research topic. Your task is to determine whether the user’s query has been well understood and whether there is an agreement between the user and the assistant. Specifically, you should:

    1. Analyze the user's query and identify the key topic or question being asked.
    2. Check whether the assistant has correctly captured the main points of the query.
    3. Confirm if the user has expressed clear expectations regarding the research direction or question. 
    4. Respond with an assessment of whether the research topic is understood and agreed upon.

    If the query has nothing to do with research, return "refine".
    If the user query is clear and the assistant’s response aligns with the query, respond with "done". 
    If there is a lack of clarity or ambiguity in the user's query, or if the assistant’s response does not align with the query, respond with "refine."
    """
    routing_message = SystemMessage(routing_prompt)

    response = llm.invoke(state["messages"] + [routing_message])
    
    if not len(state["messages"]):
        return "chatbot"
    
    if state["mode"] == "human":
        if "done" in response.content.lower():
            return "terminate_refine"
        else:
            return "chatbot"
    elif state["mode"] == "auto":
        if "done" in response.content.lower():
            return "terminate_refine"
        else:
            return "refine_auto"
    else:
        raise ValueError("Invalid mode for Agent State")    

def chatbot(state: State, llm: ChatOpenAI, console: Console):
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
    """

    system_message = SystemMessage(system_prompt)


    user_input = console.input("[bold blue]Ask your question (type 'exit' to quit): [/bold blue]")
    
    if user_input.lower() == "exit":
        console.print("[bold green]Goodbye![/bold green]")
    else:
        try:
            user_message = HumanMessage(content=user_input)
            state["messages"].append(user_message)
            
            response = llm.invoke(state["messages"] + [system_message])
            response_message = response.content
            state["messages"].append(response)
            console.print(Markdown(response_message))
        
        except Exception as e:
            console.print(f"[bold red]An error occurred: {e}[/bold red]")
    return state

def terminate_refine(state: State, llm: ChatOpenAI):
    #TODO
    return state

def search(state: State, llm: ChatOpenAI):
    #TODO
    return state

@tool
def arxiv_search():
    #TODO
    """
    Tool used to browse arXiv.
    """
    return


def route_search(state: State, llm: ChatOpenAI):
    #TODO
    return "terminate_search"

def terminate_search(state: State, llm: ChatOpenAI):
    #TODO
    return state

def build_graph(llm: ChatOpenAI, memory: MemorySaver, tools, console: Console):
    workflow = StateGraph(State)
    
    workflow.add_node("route_refine", lambda s: s)
    workflow.add_node("refine_auto", lambda s: refine_auto(s, llm))
    workflow.add_node("chatbot", lambda s: chatbot(s, llm, console))
    workflow.add_node("terminate_refine", lambda s: terminate_refine(s, llm))

    workflow.add_node("route_search", lambda s: s)
    workflow.add_node("search", lambda s: search(s, llm))

    workflow.add_node("terminate_search", lambda s: terminate_search(s, llm))

    tool_node = ToolNode(tools)
    workflow.add_node("tools", tool_node)


    workflow.add_edge(START, "route_refine")

    workflow.add_conditional_edges(
        "route_refine",
        lambda s: route_refine(s, llm),
        {
            "chatbot": "chatbot",
            "refine_auto": "refine_auto",
            "terminate_refine": "terminate_refine",
        }
    )

    workflow.add_edge("chatbot", "route_refine")
    workflow.add_edge("refine_auto", "route_refine")
    workflow.add_edge("terminate_refine", "search")

    workflow.add_conditional_edges(
        "route_search",
        lambda s: route_refine(s, llm),
        {
            "search": "search",
            "tools": "tools",
            "terminate_search": "terminate_search",
        }
    )

    workflow.add_edge("search", "route_search")
    workflow.add_edge("tools", "search")

    workflow.add_edge("terminate_search", END)

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
    

def main():
    load_dotenv()
    
    console = Console()

    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    tools = [arxiv_search]
    llm = ChatOpenAI(model=OPENAI_MODEL)

    mode = console.input("[bold blue]Do you want to work alongside the LLM during its research? (type 'auto' for the LLM to work alone, otherwise type 'human'): [/bold blue]")
    while mode not in ["auto", "human"]:
        console.print("[bold red]Invalid input. Please try again.[/bold red]")
        mode = console.input("[bold blue]Do you want to work alongside the LLM during its research? (type 'auto' for the LLM to work alone, otherwise type 'human'): [/bold blue]")
    
    workflow = build_graph(llm, MemorySaver(), tools, console)

    config = {"configurable": {"thread_id": "1"}}

    state = {"messages": [], "mode": mode}
    if Confirm.ask("\n[yellow]Display the graph?[/yellow]"):
        show_graph(workflow, console)
    
    try: 
        workflow.invoke(state, config, stream_mode="values")
    except Exception as e:
        console.print(f"[bold red]An error occurred: {e}[/bold red]")
    
    return

if __name__ == "__main__":
    main()