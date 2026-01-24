from __future__ import annotations

from pathlib import Path
import os
import re
import time
import typer

from rich.console import Console
from rich.markup import escape

from .agent import Agent
from .agent_profiles import get_profile
from .runners import get_runner
from .tools import tools as installed_tools

from .experiments import ExperimentController
from .telemetry_rerun import SegmentContext

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
            f"\n[green]⏺ {escape(name)}[/]([dim]{escape(preview)}[/])",
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
    runner_name: str = typer.Option("loop", help="Runner module (e.g. loop)."),
    profile: str = typer.Option(
        "default", help="Agent profile preset (default/chat_safe/dev_all_tools)."
    ),
    spawn: bool = typer.Option(False, help="Spawn rerun viewer."),
    dataset: str = typer.Option("cli_run", help="Dataset name (experiment folder)."),
):
    """
    Run a single prompt file as a single segment (.rrd).
    """
    prompt = file.read_text(encoding="utf-8")

    exp = ExperimentController(dataset_name=dataset, spawn_viewer=spawn)
    seg_name = f"run_{file.stem}"

    with exp.new_segment(seg_name, meta={"mode": "run", "file": file.name, "profile": profile}) as rec:
        ctx = SegmentContext(rec)

        the_profile = get_profile(profile)
        agent = Agent(
            tools_installed=installed_tools,
            profile=the_profile,
            name="agentA",
            instance_id="agentA",
        )

        runner = get_runner(runner_name)
        mem = runner.new_memory(agent)

        ui = TerminalUI()
        ui.console.print(
            f"[bold]t2f run[/] | [dim]{agent.model}[/] | [dim]{os.getcwd()}[/] | "
            f"[dim]profile={the_profile.name}[/] | [dim]recording_id={rec.recording_id}[/]\n"
            f"[dim]rrd={rec.rrd_path}[/]\n"
        )

        runner(agent, prompt, memory=mem, ui=ui, ctx=ctx)
        ui.console.print()


@app.command()
def chat(
    runner_name: str = typer.Option("loop", help="Runner module (e.g. loop)."),
    profile: str = typer.Option(
        "default", help="Agent profile preset (default/chat_safe/dev_all_tools)."
    ),
    spawn: bool = typer.Option(False, help="Spawn rerun viewer."),
    dataset: str = typer.Option("cli_chat", help="Dataset name (experiment folder)."),
):
    """
    Interactive chat session. One segment (.rrd) per session.
    """
    exp = ExperimentController(dataset_name=dataset, spawn_viewer=spawn)

    seg_name = f"chat_{int(time.time())}"

    with exp.new_segment(seg_name, meta={"mode": "chat", "profile": profile}) as rec:
        ctx = SegmentContext(rec)

        the_profile = get_profile(profile)
        agent = Agent(
            tools_installed=installed_tools,
            profile=the_profile,
            name="agentA",
            instance_id="agentA",
        )

        runner = get_runner(runner_name)
        mem = runner.new_memory(agent)

        ui = TerminalUI()
        ui.console.print(
            f"[bold]t2f chat[/] | [dim]{agent.model}[/] | [dim]{os.getcwd()}[/] | "
            f"[dim]profile={the_profile.name}[/] | [dim]recording_id={rec.recording_id}[/]\n"
            f"[dim]rrd={rec.rrd_path}[/]"
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
                    runner.reset(agent, mem)
                    rec.event("⏺ cleared", step=ctx.step())
                    ui.console.print("[green]⏺ Cleared conversation[/]\n")
                    continue

                runner(agent, user_input, memory=mem, ui=ui, ctx=ctx)
                ui.console.print()

            except (KeyboardInterrupt, EOFError):
                break


@app.command(name="main")
def run_main():
    """
    Run the repo's experiment harness (terminal2f/main.py).
    This restores: `t2f main`
    """
    from .main import main as real_main
    real_main()


if __name__ == "__main__":
    app()
