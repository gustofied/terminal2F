from pathlib import Path
import typer

from .agent import Agent
from .runners import load
from .tools import tools as tool_schemas
from . import control_tower

app = typer.Typer(add_completion=False)

def _extract_text(response) -> str:
    msg = response.choices[0].message
    return (getattr(msg, "content", "") or "").strip()

@app.command()
def run(
    file: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to a UTF-8 text file containing the prompt.",
    )
):
    prompt = file.read_text(encoding="utf-8")

    control_tower.init()
    agent = Agent(tools=tool_schemas, name="agentA", instance_id="agentA")
    run_agent = load("regular")

    response = run_agent(agent, prompt)
    typer.echo(_extract_text(response))

@app.command(name="main")
def run_main():
    from .main import main as real_main
    real_main()
