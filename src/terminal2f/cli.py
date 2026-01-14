from pathlib import Path
import os
import re
import typer
import uuid

from rich.console import Console
from rich.markup import escape

from .agent import Agent
from .runners import load
from .tools import tools as installed_tools
from . import control_tower

app = typer.Typer(add_completion=False)


def _term_width(default: int = 80) -> int:
    try:
        return min(os.get_terminal_size().columns, default)
    except Exception:
        return default


def _render_bold_md(text: str) -> str:
    out = []
    last = 0
    for m in re.finditer(r"\*\*(.+?)\*\*", text):
        out.append(escape(text[last:m.start()]))
        out.append(f"[bold]{escape(m.group(1))}[/bold]")
        last = m.end()
    out.append(escape(text[last:]))
    return "".join(out)


def _preview_block(s: str, max_len: int) -> str:
    s = s or ""
    return s[:max_len] + ("..." if len(s) > max_len else "")


def _preview_tool_result(result: str) -> str:
    lines = (result or "").splitlines() or [""]
    first = _preview_block(lines[0], 60)
    if len(lines) > 1:
        return f"{first} ... +{len(lines) - 1} lines"
    return first


class TerminalUI:
    def __init__(self):
        self.console = Console()

    def separator(self):
        self.console.print("─" * _term_width(), style="dim")

    def on_event(self, text: str):
        self.console.print(f"\n[yellow]⏺[/] {escape(text)}", markup=True)

    def on_assistant_text(self, text: str):
        self.console.print(f"\n[cyan]⏺[/] {_render_bold_md(text)}", markup=True)

    def on_tool_call(self, name: str, params: dict):
        preview = ""
        if params:
            first_val = next(iter(params.values()))
            preview = _preview_block(str(first_val), 50)

        self.console.print(
            f"\n[green]⏺ {name.capitalize()}[/]([dim]{escape(preview)}[/])",
            markup=True,
        )

    def on_tool_result(self, _name: str, result: str):
        self.console.print(
            f"  [dim]⎿  {escape(_preview_tool_result(result))}[/]",
            markup=True,
        )


@app.command()
def run(
    file: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    runner: str = typer.Option("regular", help="Runner module (e.g. regular)."),
):
    prompt = file.read_text(encoding="utf-8")

    control_tower.init()
    episode_id = uuid.uuid4().hex[:8]
    bench_step = 0

    agent = Agent(tools=installed_tools, name="agentA", instance_id="agentA")
    run_agent = load(runner)
    mem = run_agent.new_memory(agent)

    ui = TerminalUI()
    ui.console.print(
        f"[bold]t2f run[/] | [dim]{agent.model}[/] | [dim]{os.getcwd()}[/]\n"
    )

    bench_step += 1
    control_tower.set_step(bench_step)
    run_agent(agent, prompt, episode_id=episode_id, step=bench_step, memory=mem, ui=ui)
    ui.console.print()


@app.command()
def chat(
    runner: str = typer.Option("regular", help="Runner module (e.g. regular)."),
):
    control_tower.init()
    episode_id = uuid.uuid4().hex[:8]
    bench_step = 0

    agent = Agent(tools=installed_tools, name="agentA", instance_id="agentA")
    run_agent = load(runner)
    mem = run_agent.new_memory(agent)

    ui = TerminalUI()
    ui.console.print(
        f"[bold]t2f chat[/] | [dim]{agent.model}[/] | [dim]{os.getcwd()}[/]"
    )
    ui.console.print("[dim]Commands:[/] /q quit, /c clear\n")

    while True:
        try:
            ui.separator()
            user_input = ui.console.input("[bold blue]❯[/] ").strip()
            ui.separator()

            if not user_input:
                continue
            if user_input in ("/q", "exit", "quit"):
                break

            if user_input == "/c":
                bench_step += 1
                control_tower.set_step(bench_step)

                run_agent.reset(agent, mem)
                control_tower.on_event(
                    episode_id,
                    agent.name,
                    agent.instance_id,
                    bench_step,
                    "⏺ cleared",
                )
                ui.console.print("[green]⏺ Cleared conversation[/]\n")
                continue

            bench_step += 1
            control_tower.set_step(bench_step)
            run_agent(
                agent,
                user_input,
                episode_id=episode_id,
                step=bench_step,
                memory=mem,
                ui=ui,
            )
            ui.console.print()

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
