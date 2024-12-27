import os
from typing import List, Optional

import requests
from dotenv import load_dotenv # type: ignore

from rich.console import Console # type: ignore
from rich.markdown import Markdown # type: ignore
from rich.prompt import Confirm # type: ignore

from langchain_openai import ChatOpenAI # type: ignore
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage # type: ignore

from typing_extensions import TypedDict # type: ignore


import xml.etree.ElementTree as ET

from state import State, SearchAgentState, RefineAgentState, ArxivPaper, Paper, SearchQuery, initialize_agent_state # type: ignore

from langgraph.graph import StateGraph, START, END # type: ignore
from langgraph.graph.message import add_messages # type: ignore
from langgraph.checkpoint.memory import MemorySaver # type: ignore

from IPython.display import Image, display # type: ignore
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles # type: ignore

import io
from PIL import Image # type: ignore

from datetime import datetime as Datetime

import requests
import re
import PyPDF2
from io import BytesIO


OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_MODE = 'human'
DEFAULT_CONFIG = {"configurable": {"thread_id": "1"}}


class ArxivAgent:
    def __init__(self, llm: ChatOpenAI, console: Console, mode: str, stream_mode: str = "values", memory: MemorySaver = MemorySaver(), config = DEFAULT_CONFIG):
        self.llm = llm
        self.console = console
        
        self.stream_mode = stream_mode
        self.memory = memory
        self.config = config
        self.state = initialize_agent_state(mode)

        self._build_graph()

        if Confirm.ask("\n[yellow]Display the graph?[/yellow]"):
            self._show_graph()
        
        self.namespace = {'ns': 'http://www.w3.org/2005/Atom'}
        self.base_url = 'http://export.arxiv.org/api/query?'

    def _build_graph(self) -> None:
        workflow = StateGraph(State)
        workflow.add_node("RefineAgent", lambda s: self._refine_agent(s))
        workflow.add_node("BuildQueryAgent", lambda s: self._build_query(s))
        workflow.add_node("QueryArxiv", lambda s: self._query_arxiv(s))
        workflow.add_node("ReviewPapers", lambda s: self._review_papers(s))
        workflow.add_node("SourcesExtraction", lambda s: self._extract_sources(s))
        workflow.add_node("BuildAnswerAgent", lambda s: self._build_answer(s))

        workflow.add_edge(START, "RefineAgent")

        workflow.add_conditional_edges(
            "RefineAgent", 
            lambda s: END if s["end"] else "BuildQueryAgent",
            {
                "BuildQueryAgent": "BuildQueryAgent",
                END: END,
            }
        )

        workflow.add_edge("BuildQueryAgent", "QueryArxiv")
        workflow.add_edge("QueryArxiv", "ReviewPapers")

        workflow.add_conditional_edges(
            "ReviewPapers", 
            lambda s: "BuildAnswerAgent" if s["search_agent"]["done"] else "SourcesExtraction",
            {
                "BuildAnswerAgent": "BuildAnswerAgent",
                "SourcesExtraction": "SourcesExtraction",
            }
        )

        workflow.add_edge("SourcesExtraction", "QueryArxiv")

        workflow.add_edge("BuildAnswerAgent", END)

        self.workflow = workflow.compile(checkpointer=self.memory)

        return
    
    def run(self) -> None:
        self.console.print(Markdown(f"# Refining Query"))
        self.workflow.invoke(self.state, config = self.config, stream_mode = self.stream_mode)
    
    def _show_graph(self) -> None:
        try:
            image_bytes = self.workflow.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)
            Image.open(io.BytesIO(image_bytes)).show()
        except Exception as e:
            self.console.print(f"[bold red]An unexpected error occurred: {e}[/bold red]")
        return
    
    def _check_answer_quality(self, initial_system_message: SystemMessage, llm_output: AIMessage) -> bool:
        try:
            system_prompt = (
                f"Determine if this output: {llm_output.content} aligns with the "
                f"initial SystemMessage: {initial_system_message.content}. "
                f"If it aligns, return 'continue', otherwise return 'redo'."
            )
            system_message = SystemMessage(system_prompt)
            response = self.llm.invoke([system_message])
            return "continue" in response.content.lower()
        except Exception as e:
            self.console.print(f"[bold red]An unexpected error occurred: {e}[/bold red]")
        return False
    
    def _refine_agent(self, state: State) -> State:
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
            user_input = self.console.input("[bold blue]Ask your question (type 'exit' to quit, 'again' to generate a new answer, 'discard' to discard your previous input, 'next' to start the search phase): [/bold blue]\n")
            
            if user_input.lower() == "exit":
                self.console.print("[bold green]Goodbye![/bold green]")
                return {**state, "end": True}
            elif user_input.lower() == "discard":
                state["refine_agent"]["messages"].pop()
                state["refine_agent"]["messages"].pop()
            else:
                try:
                    if user_input.lower() == "again":
                        state["refine_agent"]["messages"].pop()
                    elif user_input.lower() == "next":
                        self.console.print(Markdown(f"# Building Query"))
                        return {**state, "refine_agent": {**state["refine_agent"], "done": True}}
                    else:
                        user_message = HumanMessage(content=user_input)
                        state["refine_agent"]["messages"].append(user_message)
                    
                    response = None
                    counter = 0                    
                    while response is None or not self._check_answer_quality(system_message, response):
                        response = self.llm.invoke(state["refine_agent"]["messages"] + [system_message])
                        response_message = response.content
                        
                        if counter > 3: 
                            break
                        
                        counter += 1

                    if "done" in response_message.lower():
                        self.console.print(Markdown(f"# Building Query"))
                        return {**state, "refine_agent": {**state["refine_agent"], "done": True}}
                    
                    state["refine_agent"]["messages"].append(response)

                    self.console.print(Markdown(response_message))

                    state["refine_agent"]["iterations"] += 1
                
                except Exception as e:
                    self.console.print(f"[bold red]An error occurred: {e}[/bold red]")
    
    def _build_query(self, state: State) -> State:
        system_prompt = """
        You are an AI agent specialized in crafting SearchQuery objects from conversations between a user and an LLM.
        The SearchQuery object is defined as follows:
        class SearchQuery(TypedDict):
            queries: List[str] # main query and additional augmented / expanded queries
            start_date: Optional[Datetime] # The oldest date allowed during our researches in YYYY-MM-DD format, if mentioned (i.e., Papers should not be older)
            end_date: Optional[Datetime] # The most recent date allowed during our researched in YYYY-MM-DD format, if mentioned (i.e., Papers should not be more recent)
        
        Format your response as a JSON object with these fields:
            - queries List[str]: Main query and additional augmented / expanded queries ;
            - start_date (Optional[str]): The oldest date allowed during our researches in YYYY-MM-DD format, if mentioned (i.e., Papers should not be older) ;
            - end_date (Optional[str]): The most recent date allowed during our researched in YYYY-MM-DD format, if mentioned (i.e., Papers should not be more recent) ;
        
        Include only fields that are clearly specified in the conversation.
        Your output should be the raw JSON object and not its markdown representation (i.e., ```json ...```)
        The generated queries should be formulated to maximize their relevance during research phase.
        """

        system_message = SystemMessage(system_prompt)

        try:
            response = self.llm.invoke(state["refine_agent"]["messages"] + [system_message])
            raw_object = response.content
            raw_object = raw_object.replace("null", "None").replace("```json", "").replace("```", "")
            search_query: SearchQuery = eval(raw_object)

            if not search_query.get("queries"):
                raise ValueError("Search query must contain queries")
            if search_query.get("start_date"):
                year, month, day = map(int, search_query["start_date"].split("-"))
                search_query["start_date"] = Datetime(year, month, day)
            if search_query.get("end_date"):
                year, month, day = map(int, search_query["end_date"].split("-"))
                search_query["end_date"] = Datetime(year, month, day)

            state["search_agent"]["search_query"] = search_query
        except Exception as e:
            self.console.print(f"[bold red]An unexpected error occurred: {e}[/bold red]")
        
        self.console.print(Markdown(f"# Browsing Through arXiv"))

        return state

    def _query_arxiv(self, state: State, max_results: int = 5) -> State:
        def _fetch_papers(search_query: str):
            """Fetch papers from arXiv API based on a specific query."""
            try:
                response = requests.get(self.base_url + search_query)
                response.raise_for_status()
                root = ET.fromstring(response.text)

                for entry in root.findall('ns:entry', self.namespace):
                    paper = {
                        "title": entry.find('ns:title', self.namespace).text.strip(),
                        "summary": entry.find('ns:summary', self.namespace).text.strip(),
                        "link": entry.find('ns:link', self.namespace).get('href'),
                        "published": Datetime.strptime(
                            entry.find('ns:published', self.namespace).text.strip(),
                            "%Y-%m-%dT%H:%M:%SZ"
                        ),
                        "updated": Datetime.strptime(
                            entry.find('ns:updated', self.namespace).text.strip(),
                            "%Y-%m-%dT%H:%M:%SZ"
                        ),
                        "authors": [
                            author.find('ns:name', self.namespace).text.strip()
                            for author in entry.findall('ns:author', self.namespace)
                        ],
                    }

                    paper_id_match = re.findall(r'\d{4}\.\d{5}', paper['link'])
                    if not paper_id_match:
                        self.console.print(f"[bold yellow]Paper ID not found for {paper['title']}[/bold yellow]")
                        continue

                    paper["id"] = paper_id_match[0]

                    if paper["id"] in state["search_agent"]["papers_id"]:
                        continue  # Skip duplicates
                    
                    arxiv_paper = {
                        "paper": paper,
                        "relevant": None,
                        "sources": None,
                        "keywords": None
                    }

                    state["search_agent"]["papers"].append(arxiv_paper)
                    state["search_agent"]["papers_id"].append(paper["id"])

            except requests.exceptions.RequestException as e:
                self.console.print(f"[bold red]Request error: {e}[/bold red]")
            except ET.ParseError as e:
                self.console.print(f"[bold red]XML Parse error: {e}[/bold red]")
            except Exception as e:
                self.console.print(f"[bold red]An unexpected error occurred: {e}[/bold red]")

        # Main Logic
        if state["search_agent"]["papers_id"]:
            # Fetch papers based on existing relevant papers
            for paper in state["search_agent"]["papers"]:
                if paper.get("relevant", None):
                    id_list = ",".join(paper["sources"] or [])
                    search_query = f'id_list={id_list}'
                    _fetch_papers(search_query)
        else:
            # Fetch papers based on initial queries
            for query in state["search_agent"]["search_query"].get("queries", []):
                search_query = f'search_query=all:{query}&start=0&max_results={max_results}'
                _fetch_papers(search_query)

        return state
    
    def _review_papers(self, state: State) -> State:
        def _is_paper_relevant(paper: ArxivPaper, search_query: SearchQuery) -> bool:
            try:
                system_prompt = f"""
                Your goal is to determine whether a given paper is relevant to a search query.
                The search query is: {search_query}

                You should return 'relevant' if the paper is relevant, and 'irrelevant' otherwise.
                """

                paper_prompt = f"""
                Title: 
                {paper['paper']['title']}

                Summary:
                {paper['paper']['summary']}
                """

                system_message = SystemMessage(system_prompt)
                paper_message = SystemMessage(paper_prompt)

                response = self.llm.invoke([system_message, paper_message])
                return "irrelevant" not in response.content.lower()
            
            except Exception as e:
                self.console.print(f"[bold red]Error while evaluating relevance for {paper['paper']['title']}: {e}[/bold red]")
                return False

        self.console.print("[bold cyan]Reviewing Papers for Relevance...[/bold cyan]")
        state["search_agent"]["done"] = True

        for paper in state["search_agent"]["papers"]:
            if paper.get("relevant") is not None:
                continue

            is_relevant = _is_paper_relevant(paper, state["search_agent"]["search_query"])

            if is_relevant:
                paper["relevant"] = True
                state["search_agent"]["done"] = False
                self.console.print(f"[bold green]✔ {paper['paper']['title']} deemed relevant.[/bold green]")
            else:
                paper["relevant"] = False
                self.console.print(f"[bold yellow]✘ {paper['paper']['title']} deemed irrelevant.[/bold yellow]")
        

        state["search_agent"]["iterations"] += 1
        
        # TODO: Improve routing logic
        if state["search_agent"]["iterations"] == 3:
            state["search_agent"]["done"] = True

        return state
    
    def _extract_sources(self, state: State) -> State:
        def _fetch_pdf(url: str) -> Optional[BytesIO]:
            try:
                response = requests.get(url)
                response.raise_for_status()
                return BytesIO(response.content)
            except requests.exceptions.RequestException as e:
                self.console.print(f"[bold red]Failed to fetch PDF from {url}: {e}[/bold red]")
                return None

        def _extract_text_from_pdf(pdf_file: BytesIO) -> str:
            try:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                return ''.join(page.extract_text() or '' for page in pdf_reader.pages)
            except Exception as e:
                self.console.print(f"[bold red]Failed to extract text from PDF: {e}[/bold red]")
                return ''

        def _extract_references(text: str, paper_id: str, existing_ids: List[str]) -> List[str]:
            pattern = r'arXiv:\d{4}\.\d{5}'
            references = {
                ref[6:] for ref in re.findall(pattern, text)
                if ref[6:] != paper_id and ref[6:] not in existing_ids
            }
            return list(references)

        # Main Logic
        self.console.print("[bold cyan]Extracting sources from relevant papers...[/bold cyan]")
        for paper in state["search_agent"]["papers"]:
            if not paper.get("relevant", False):
                continue  # Skip irrelevant papers

            pdf_url = paper["paper"]["link"].replace("/abs/", "/pdf/")
            self.console.print(f"[bold blue]Processing PDF: {pdf_url}[/bold blue]")

            pdf_file = _fetch_pdf(pdf_url)
            if not pdf_file:
                continue  # Skip if PDF couldn't be fetched

            text = _extract_text_from_pdf(pdf_file)
            if not text:
                continue  # Skip if text couldn't be extracted

            sources = _extract_references(
                text,
                paper["paper"]["id"],
                state["search_agent"]["papers_id"]
            )

            paper["sources"] = sources
            self.console.print(
                f"[bold green]✔ Extracted {len(sources)} source(s) from {paper['paper']['title']}[/bold green]"
            )

        return state

    def _build_answer(self, state: State) -> State:
        self.console.print(Markdown(f"# Building Answer"))
        def _aggregate_relevant_papers(papers: List[ArxivPaper]) -> List[Paper]:
            return [
                paper["paper"]
                for paper in papers
                if paper.get("relevant", False)
            ]

        self.console.print("[bold cyan]Building final answer from relevant papers...[/bold cyan]")
        relevant_papers = _aggregate_relevant_papers(state["search_agent"]["papers"])

        if not relevant_papers:
            self.console.print("[bold yellow]No relevant papers found to build an answer.[/bold yellow]")
            return state

        self.console.print("[bold green]✔ Final Answer Compiled Successfully![/bold green]")
        self.console.print(Markdown("# Papers"))

        for paper in relevant_papers:
            self.console.print(Markdown(f"## {paper.get('title', 'No title found :(')}"))
            self.console.print("[bold cyan]Summary:[/bold cyan]")
            self.console.print(Markdown(f"{paper.get('summary', 'No summary found :(')}"))
            self.console.print("[bold cyan]Link:[/bold cyan]")
            self.console.print(Markdown(f"{paper.get('link', 'No link found :(')}"))
            self.console.print("[bold cyan]Author(s):[/bold cyan]")
            self.console.print(Markdown(f"{', '.join(paper.get('authors', 'No author found :('))}"))

        return state

        

def main():
    load_dotenv()
    
    console = Console()

    llm = ChatOpenAI(model=OPENAI_MODEL)
    
    # TODO: For now the mode is 'human' by default
    mode = DEFAULT_MODE

    agent = ArxivAgent(llm, console, mode)
    try: 
        agent.run()
    except Exception as e:
        console.print(f"[bold red]An error occurred: {e}[/bold red]")
    return

if __name__ == "__main__":
    main()
