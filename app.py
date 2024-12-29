import os
import re
import json
import io
from io import BytesIO
from typing import List, Optional
from PIL import Image # type: ignore
import PyPDF2 # type: ignore
import requests # type: ignore
from dotenv import load_dotenv # type: ignore
from datetime import datetime as Datetime

import xml.etree.ElementTree as ET

from rich.console import Console # type: ignore
from rich.markdown import Markdown # type: ignore
from rich.prompt import Confirm # type: ignore

from langchain_openai import ChatOpenAI # type: ignore
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage # type: ignore
from langchain_community.tools.tavily_search import TavilySearchResults # type: ignore

from langgraph.graph import StateGraph, START, END # type: ignore
from langgraph.checkpoint.memory import MemorySaver # type: ignore

from langchain_core.runnables.graph import MermaidDrawMethod # type: ignore
from langchain_core.messages import ToolMessage # type: ignore

from state import State, ArxivPaper, Paper, SearchQuery, initialize_agent_state # type: ignore

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_MODE = 'human'
DEFAULT_CONFIG = {"configurable": {"thread_id": "1"}}

class BasicToolNode:
    
    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs["refine_agent"].get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )

        return {"refine_agent": {**inputs["refine_agent"], "messages": inputs["refine_agent"]["messages"] + outputs }}

class ArxivAgent:
    def __init__(self, llm: ChatOpenAI, console: Console, mode: str, stream_mode: str = "values", memory: MemorySaver = MemorySaver(), config = DEFAULT_CONFIG):
        
        self.tools = [TavilySearchResults(max_results=5)]
        self.llm = llm.bind_tools(self.tools)
        
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
        workflow.add_node("Refine", lambda s: self._refine_agent(s))
        workflow.add_node("Browse", lambda s: self._browse_agent(s))
        workflow.add_node("BuildQuery", lambda s: self._build_query(s))
        workflow.add_node("QueryArxiv", lambda s: self._query_arxiv(s))
        workflow.add_node("ReviewPapers", lambda s: self._review_papers(s))
        workflow.add_node("SourcesExtraction", lambda s: self._extract_sources(s))
        workflow.add_node("BuildAnswer", lambda s: self._build_answer(s))

        tool_node = BasicToolNode(tools=self.tools)
        workflow.add_node("tools", tool_node)

        workflow.add_conditional_edges(
            "Browse",
            lambda s: "Tavily Search Tool" if s["refine_agent"]["messages"][-1].tool_calls else "Build Query",
            {
                "Tavily Search Tool": "tools",
                "Build Query": "BuildQuery",
            }
        )

        workflow.add_edge("tools", "Browse")

        workflow.add_edge(START, "Refine")

        workflow.add_conditional_edges(
            "Refine", 
            lambda s: END if s["end"] else "Browse",
            {
                "Browse": "Browse",
                END: END,
            }
        )

        workflow.add_edge("BuildQuery", "QueryArxiv")
        workflow.add_edge("QueryArxiv", "ReviewPapers")

        workflow.add_conditional_edges(
            "ReviewPapers", 
            lambda s: "Wrap-Up" if s["search_agent"]["done"] else "Extract Sources",
            {
                "Wrap-Up": "BuildAnswer",
                "Extract Sources": "SourcesExtraction",
            }
        )

        workflow.add_edge("SourcesExtraction", "QueryArxiv")

        workflow.add_edge("BuildAnswer", END)

        self.workflow = workflow.compile(checkpointer=self.memory)

        return
    
    def run(self) -> None:
        self.console.print(Markdown("---"))
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
        self.console.print(
            "[bold cyan]Welcome to the Research Query Refinement Assistant!\n"
            "You can interact with me in the following ways:\n"
            "- Enter your query directly to start refining it.\n"
            "- Type 'exit' to leave the session at any time.\n"
            "- After receiving a response, you can type 'again' to retry or 'discard' to remove the last response.\n"
            "- When you're satisfied with the refinement, type 'next' to proceed to the next phase.[/bold cyan]"
        )
        
        self.console.print(Markdown("---"))

        while True:
            user_input = self.console.input("[bold blue]Ask your question: [/bold blue]")
            
            if user_input.lower() == "exit":
                self.console.print("[bold green]Goodbye![/bold green]")
                return {**state, "end": True}
            elif user_input.lower() == "discard":
                if len(state["refine_agent"]["messages"]) > 2:
                    state["refine_agent"]["messages"].pop()
                    state["refine_agent"]["messages"].pop()
            else:
                try:
                    if user_input.lower() == "again":
                        if state["refine_agent"]["messages"] and type(state["refine_agent"]["messages"][-1]) is AIMessage:
                            state["refine_agent"]["messages"].pop()
                    elif user_input.lower() == "next":
                        if state["refine_agent"]["messages"] and type(state["refine_agent"]["messages"][-1]) is AIMessage:
                            state["refine_agent"]["messages"].pop()
                        
                        self.console.print(Markdown("---"))
                        self.console.print(Markdown(f"# Browsing Query"))
                        return {**state, "refine_agent": {**state["refine_agent"], "done": True}}
                    else:
                        user_message = HumanMessage(content=user_input)
                        state["refine_agent"]["messages"].append(user_message)
                    
                    # TODO: Improve logic
                    response = None
                    counter = 0                    
                    while response is None or not self._check_answer_quality(system_message, response):
                        response = self.llm.invoke(state["refine_agent"]["messages"] + [system_message])
                        response_message = response.content
                        if counter > 3: 
                            break
                        
                        counter += 1

                    if "done" in response_message.lower():
                        self.console.print(Markdown("---"))
                        self.console.print(Markdown(f"# Browsing Query"))
                        return {**state, "refine_agent": {**state["refine_agent"], "done": True}}
                    
                    state["refine_agent"]["messages"].append(response)

                    self.console.print(Markdown("## Agent's Response"))
                    self.console.print(Markdown(response_message))
                    self.console.print(Markdown("---"))

                    state["refine_agent"]["iterations"] += 1
                
                except Exception as e:
                    self.console.print(f"[bold red]An error occurred: {e}[/bold red]")
    
    def _browse_agent(self, state: State) -> State:
        
        system_prompt = """
            Given a message history between the user and a llm, your goal is to browse the web and add to the context of the query. It might be relevant to query the web multiple times given the responses you get.
            """
        
        system_message = SystemMessage(system_prompt)
        
        response = self.llm.invoke(state["refine_agent"]["messages"] + [system_message])
        
        state["refine_agent"]["messages"].append(response)
        
        return state
        
    def _build_query(self, state: State) -> State:
        self.console.print(Markdown(f"# Building Query"))
        system_prompt = """
        You are an AI assistant specialized in generating SearchQuery objects for querying research databases like arXiv. A SearchQuery object is structured as follows:

        class SearchQuery(TypedDict):
            goal: str # User's intent
            queries: List[str]  # Main query and additional expanded queries
            start_date: Optional[str]  # The earliest publication date allowed in YYYY-MM-DD format, if specified
            end_date: Optional[str]  # The latest publication date allowed in YYYY-MM-DD format, if specified

        ### Your Task:
        1. Based on the given conversation, generate a **SearchQuery** JSON object containing:
        - **goal (str)**: The global intent of the user. Used later to determine whether a retrieved paper is relevant or not even though it the queries.
        - **queries (List[str])**: The main search query and any additional refined or expanded queries to maximize relevance.
        - **start_date (Optional[str])**: If the user specifies a start date, include it in YYYY-MM-DD format.
        - **end_date (Optional[str])**: If the user specifies an end date, include it in YYYY-MM-DD format.
        
        2. **Adhere to these rules:**
        - Include only fields explicitly mentioned in the conversation. If a field is not discussed, omit it.
        - Ensure queries are relevant and formatted to maximize effectiveness in arXiv searches (e.g., concise, leveraging logical operators or field-specific syntax if applicable).

        3. **Output Format:**
        - Return only the raw JSON object with no additional explanations, code blocks, or markdown formatting.
        - Ensure JSON is syntactically correct and ready for parsing.

        ### Additional Notes:
        - If the user implies multiple possible queries, include them as additional entries in the `queries` list.
        - Avoid including unrelated or overly broad terms.
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

    def _query_arxiv(self, state: State, max_results: int = 3) -> State:
        def _fetch_papers(search_query: str):
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

        if state["search_agent"]["papers_id"]:
            for paper in state["search_agent"]["papers"]:
                if paper.get("relevant", None):
                    id_list = ",".join(paper["sources"] or [])
                    search_query = f'id_list={id_list}'
                    _fetch_papers(search_query)
        else:
            for query in state["search_agent"]["search_query"].get("queries", []):
                search_query = f'search_query=all:{query}&start=0&max_results={max_results}'
                _fetch_papers(search_query)

        return state
    
    def _review_papers(self, state: State) -> State:
        def _is_paper_relevant(paper: ArxivPaper, search_query: SearchQuery) -> int:
            try:
                system_prompt = f"""
                You are a research paper relevance evaluator. Your task is to analyze a research paper and determine its relevance to a specific search query.

                Search Query: {search_query}

                Evaluation criteria:
                - Score from 0 (completely irrelevant) to 10 (highly relevant)
                - Consider both direct and indirect relevance to the query
                - Factor in:
                1. Title relevance
                2. Abstract/summary content match
                3. Methodology alignment (if mentioned)
                4. Potential applications related to the query
                
                Return only a single integer score between 0 and 10. Do not include any explanation or additional text.
                """

                paper_prompt = f"""
                Title: {paper['paper']['title']}
                Summary: {paper['paper']['summary']}

                Based on the above content and the search query, provide a single integer score between 0 and 10.
                """

                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=paper_prompt)
                ]

                response = self.llm.invoke(messages)
                
                cleaned_response = response.content.strip()
                
                score_match = re.search(r'\b([0-9]|10)\b', cleaned_response)
                
                if score_match:
                    score = int(score_match.group(1))
                    return max(0, min(score, 10))
                else:
                    self.console.print(f"[yellow]Warning: Could not parse score for paper '{paper['paper']['title']}'. Defaulting to 0.[/yellow]")
                    return 0

            except ValueError as ve:
                self.console.print(f"[bold yellow]Value error while evaluating '{paper['paper']['title']}': {ve}[/bold yellow]")
                return 0
            except Exception as e:
                self.console.print(f"[bold red]Error while evaluating '{paper['paper']['title']}': {e}[/bold red]")
                return 0

        self.console.print("[bold cyan]Reviewing Papers for Relevance...[/bold cyan]")
        state["search_agent"]["done"] = True

        for paper in state["search_agent"]["papers"]:
            if paper.get("relevant") is not None:
                continue

            score = _is_paper_relevant(paper, state["search_agent"]["search_query"])

            is_relevant = score > 6

            if is_relevant:
                paper["relevant"] = True
                state["search_agent"]["done"] = False
                self.console.print(f"[bold green]✔ {paper['paper']['title']} deemed relevant with a score of {score}.[/bold green]")
            else:
                paper["relevant"] = False
                self.console.print(f"[bold yellow]✘ {paper['paper']['title']} deemed irrelevant with a score of {score}.[/bold yellow]")
        

        state["search_agent"]["iterations"] += 1
        
        # TODO: Improve routing logic
        if state["search_agent"]["iterations"] == 2:
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

        self.console.print("[bold cyan]Extracting sources from relevant papers...[/bold cyan]")
        for paper in state["search_agent"]["papers"]:
            if not paper.get("relevant", False):
                continue

            pdf_url = paper["paper"]["link"].replace("/abs/", "/pdf/")
            self.console.print(f"[bold blue]Processing PDF: {pdf_url}[/bold blue]")

            pdf_file = _fetch_pdf(pdf_url)
            if not pdf_file:
                continue

            text = _extract_text_from_pdf(pdf_file)
            if not text:
                continue

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
        self.console.print(Markdown("---"))
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
            self.console.print(Markdown("---"))

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