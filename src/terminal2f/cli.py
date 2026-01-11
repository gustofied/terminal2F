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

@app.callback(invoke_without_command=True)
def main(file: Path = typer.Argument(...)):
    prompt = file.read_text(encoding="utf-8")

    control_tower.init()

    agent = Agent(tools=tool_schemas, name="agentA", instance_id="agentA")
    run_agent = load("regular")

    response = run_agent(agent, prompt)
    typer.echo(_extract_text(response))
