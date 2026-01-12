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
    file: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    runner: str = typer.Option("regular", help="Runner module (e.g. regular)."),
):
    """Run a single prompt from a file and exit."""
    prompt = file.read_text(encoding="utf-8")

    control_tower.init()
    agent = Agent(tools=tool_schemas, name="agentA", instance_id="agentA")
    response = load(runner)(agent, prompt)
    typer.echo(_extract_text(response))


@app.command()
def chat(
    runner: str = typer.Option("regular", help="Runner module (e.g. regular)."),
):
    """Interactive chat. Commands: /q quit, /c clear."""
    control_tower.init()
    agent = Agent(tools=tool_schemas, name="agentA", instance_id="agentA")
    run_agent = load(runner)

    state_key = f"_{runner}_runner_state"

    typer.echo(f"t2f chat | runner={runner}")
    typer.echo("Commands: /q quit, /c clear\n")

    while True:
        try:
            user_input = input("❯ ").strip()
            if not user_input:
                continue
            if user_input in ("/q", "exit", "quit"):
                break
            if user_input == "/c":
                agent.__dict__.pop(state_key, None)
                typer.echo("⏺ cleared\n")
                continue

            response = run_agent(agent, user_input)
            text = _extract_text(response)
            if text:
                typer.echo(text)
                typer.echo()

        except (KeyboardInterrupt, EOFError):
            break


@app.command(name="main")
def run_main(
    runner: str = typer.Option("regular", help="Runner module to use for main()."),
):
    """Run the scripted main loop."""
    from . import main as main_mod
    main_mod.load = load  # optional: keeps imports consistent if you refactor later
    main_mod.main()       # runs terminal2f/main.py:main()


if __name__ == "__main__":
    app()
