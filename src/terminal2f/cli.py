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
    prompt = file.read_text(encoding="utf-8")

    control_tower.init()
    agent = Agent(tools=tool_schemas, name="agentA", instance_id="agentA")

    response = load(runner)(agent, prompt)
    typer.echo(_extract_text(response))


@app.command()
def chat(
    runner: str = typer.Option("regular", help="Runner module (e.g. regular)."),
):
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
                state = agent.__dict__.setdefault(
                    state_key,
                    {
                        "instance_id": agent.instance_id,
                        "agent_name": agent.name,
                        "turn_idx": 0,
                        "messages": [{"role": "system", "content": agent.system_message}],
                    },
                )

                # Allocate a unique turn index for the clear marker (so nothing collides).
                state["turn_idx"] += 1
                turn_idx = state["turn_idx"]

                episode_idx = state.setdefault("episode_idx", 0) + 1
                state["episode_idx"] = episode_idx

                control_tower.on_event(agent.name, agent.instance_id, turn_idx, f"⏺ cleared (episode={episode_idx})")

                # Reset context only; DO NOT reset turn_idx.
                state["messages"] = [{"role": "system", "content": agent.system_message}]

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
    from .main import main as real_main
    real_main()


if __name__ == "__main__":
    app()
