import os
from dotenv import load_dotenv # type: ignore

from rich.console import Console # type: ignore
from rich.markdown import Markdown # type: ignore

from langchain_openai import ChatOpenAI # type: ignore
from langchain_core.messages import HumanMessage, SystemMessage # type: ignore


def main():
    load_dotenv()
    
    console = Console()

    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=OPENAI_MODEL)

    system_prompt = "You are an agent designed to find the most relevant papers on arXiv."
    system_message = SystemMessage(content=system_prompt)

    memory = []
    memory.append(system_message)

    while True:
        user_input = console.input("[bold blue]Ask your question (type 'exit' to quit): [/bold blue]")

        if user_input.lower() == "exit":
            console.print("[bold green]Goodbye![/bold green]")
            break
        
        try:
            memory.append(HumanMessage(content=user_input))

            response = llm.invoke(memory)
            memory.append(response)

            console.print(Markdown(response.content))
        
        except Exception as e:
            console.print(f"[bold red]An error occurred: {e}[/bold red]")
    
    return

if __name__ == "__main__":
    main()