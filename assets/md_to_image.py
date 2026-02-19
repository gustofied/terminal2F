"""
Render markdown files as styled HTML, SVG, or PNG.

Output goes to assets/docs/<project>/

Usage:
    uv run --with rich assets/md_to_image.py rl grpo_example.md
    uv run --with rich assets/md_to_image.py rl myfile.md --section "## Training"
    uv run --with rich assets/md_to_image.py rl --all
    uv run --with rich assets/md_to_image.py --index
"""

import argparse
import re
import sys
import tomllib
from pathlib import Path

ASSETS_DIR = Path(__file__).parent
DOCS_DIR = ASSETS_DIR / "docs"

DEFAULT_STYLE = {
    "colors": {
        "h1": "#cb4b16",
        "h2": "#268bd2",
        "h3": "#2aa198",
        "h4": "#859900",
        "code": "#d33682",
        "code_block": "#586e75",
        "link": "#268bd2",
        "bullet": "#cb4b16",
        "bold": "#073642",
        "italic": "#586e75",
        "hr": "#93a1a1",
        "table_header": "#268bd2",
        "table_cell": "#073642",
        "table_border": "#93a1a1",
        "background": "#ffffff",
        "text": "#1e1e1e",
        "code_bg": "#f5f5f0",
        "selection": "#d0e8ff",
    },
    "font": {
        "family": "'Fira Code', 'Cascadia Code', 'JetBrains Mono', 'Menlo', monospace",
        "size": "14px",
        "line_height": "1.35",
    },
    "layout": {
        "width": 100,
        "max_width": "960px",
        "padding": "0.5em 2em",
    },
    "code_theme": "solarized-light",
}


def load_project_style(project_dir: Path) -> dict:
    """Load style.toml from project dir, merged over defaults."""
    style = _deep_copy(DEFAULT_STYLE)
    style_path = project_dir / "style.toml"
    if style_path.exists():
        overrides = tomllib.loads(style_path.read_text())
        _deep_merge(style, overrides)
    return style


def _deep_copy(d: dict) -> dict:
    return {k: _deep_copy(v) if isinstance(v, dict) else v for k, v in d.items()}


def _deep_merge(base: dict, overrides: dict):
    for k, v in overrides.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def clean_text(text: str) -> str:
    """Apply style rules: no em dashes, no smart quotes."""
    text = text.replace("\u2014", "-")
    text = text.replace("\u2013", "-")
    text = text.replace("\u2012", "-")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    return text


def extract_section(text: str, section: str) -> str:
    """Extract a specific section from markdown text."""
    lines = text.split("\n")
    level = section.count("#")
    prefix = "#" * level + " "
    capturing = False
    captured = []
    for line in lines:
        if line.strip().startswith(section.strip()):
            capturing = True
            captured.append(line)
        elif capturing and line.startswith(prefix) and line.strip() != section.strip():
            break
        elif capturing:
            captured.append(line)
    result = "\n".join(captured)
    if not result.strip():
        print(f"Section '{section}' not found")
        sys.exit(1)
    return result


def make_stem(md_path: str, section: str | None) -> str:
    stem = Path(md_path).stem
    if section:
        slug = section.strip().strip("#").strip().lower()
        slug = re.sub(r"[^a-z0-9]+", "_", slug).strip("_")
        stem = f"{stem}_{slug}"
    return stem


def get_theme(style: dict):
    from rich.theme import Theme
    c = style["colors"]
    return Theme({
        "markdown.h1": f"bold {c['h1']}",
        "markdown.h2": f"bold {c['h2']}",
        "markdown.h3": f"bold {c['h3']}",
        "markdown.h4": f"bold {c['h4']}",
        "markdown.code": c["code"],
        "markdown.code_block": c["code_block"],
        "markdown.link": f"{c['link']} underline",
        "markdown.item.bullet": c["bullet"],
        "markdown.item.number": c["bullet"],
        "markdown.bold": f"bold {c['bold']}",
        "markdown.italic": f"italic {c['italic']}",
        "markdown.hr": c["hr"],
        "table.header": f"bold {c['table_header']}",
        "table.cell": c["table_cell"],
        "table.border": c["table_border"],
    })


def get_page_style(style: dict) -> str:
    c = style["colors"]
    f = style["font"]
    l = style["layout"]
    return f"""
    body {{
        background: {c['background']};
        color: {c['text']};
        font-family: {f['family']};
        font-size: {f['size']};
        line-height: {f['line_height']};
        padding: {l['padding']};
        max-width: {l['max_width']};
        margin: 0 auto;
    }}
    pre {{
        white-space: pre-wrap;
        word-wrap: break-word;
        margin: 0;
    }}
    ::selection {{
        background: {c['selection']};
        color: {c['text']};
    }}
    a {{ color: {c['link']}; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
"""


def render_to_svg(text: str, output_dir: Path, stem: str, style: dict) -> str:
    from rich.console import Console
    from rich.markdown import Markdown

    width = style["layout"]["width"]
    console = Console(record=True, width=width, theme=get_theme(style))
    console.print(Markdown(text, code_theme=style["code_theme"]))
    svg_path = output_dir / f"{stem}.svg"
    console.save_svg(str(svg_path), title=stem)
    return str(svg_path)


def render_to_html(text: str, output_dir: Path, stem: str, style: dict) -> str:
    from rich.console import Console
    from rich.markdown import Markdown

    width = style["layout"]["width"]
    console = Console(record=True, width=width, theme=get_theme(style))
    console.print(Markdown(text, code_theme=style["code_theme"]))
    html_content = console.export_html()

    code_bg = style["colors"]["code_bg"]
    html_content = re.sub(
        r'background-color: #[0-9a-fA-F]{6}',
        f'background-color: {code_bg}',
        html_content,
    )

    page_style = get_page_style(style)
    c = style["colors"]
    full_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>{page_style}
    .back {{ color: {c['hr']}; font-size: 12px; margin-bottom: 0; }}
</style>
</head>
<body>
<p class="back"><a href="index.html">back</a></p>
<pre>{html_content}</pre>
</body>
</html>"""

    html_path = output_dir / f"{stem}.html"
    html_path.write_text(full_html)
    return str(html_path)


def svg_to_png(svg_path: str) -> str:
    import cairosvg
    png_path = svg_path.replace(".svg", ".png")
    cairosvg.svg2png(url=svg_path, write_to=png_path, scale=2)
    return png_path


def _stem_to_title(stem: str) -> str:
    return stem.replace("_", " ").title()


def build_project_index(project_dir: Path) -> str:
    """Build index.html for a single project folder."""
    style = load_project_style(project_dir)
    page_style = get_page_style(style)
    c = style["colors"]

    html_files = {f.stem: f for f in project_dir.glob("*.html") if f.name != "index.html"}
    total = len(html_files)
    project_name = project_dir.name.upper()

    # Check for index.toml with groups
    index_toml = project_dir / "index.toml"
    if index_toml.exists():
        config = tomllib.loads(index_toml.read_text())
        project_name = config.get("title", project_name)
        grouped_stems = set()
        body_parts = []

        for group in config.get("groups", []):
            name = group["name"]
            items = []
            for entry in group.get("docs", []):
                if isinstance(entry, dict):
                    stem = entry["file"]
                    title = entry.get("title", _stem_to_title(stem))
                else:
                    stem = entry
                    title = _stem_to_title(stem)
                if stem in html_files:
                    items.append(f'        <li><a href="{stem}.html">{title}</a></li>')
                    grouped_stems.add(stem)
            if items:
                body_parts.append(
                    f'    <h2>{name}</h2>\n    <ul>\n'
                    + "\n".join(items)
                    + "\n    </ul>"
                )

        # Ungrouped docs
        ungrouped = [s for s in sorted(html_files) if s not in grouped_stems]
        if ungrouped:
            items = [f'        <li><a href="{s}.html">{_stem_to_title(s)}</a></li>' for s in ungrouped]
            body_parts.append(
                '    <h2>Other</h2>\n    <ul>\n'
                + "\n".join(items)
                + "\n    </ul>"
            )

        entries_html = "\n".join(body_parts)
    else:
        items = [f'        <li><a href="{f.name}">{_stem_to_title(f.stem)}</a></li>' for f in sorted(html_files.values(), key=lambda f: f.stem)]
        entries_html = "    <ul>\n" + "\n".join(items) + "\n    </ul>" if items else "    <ul>\n        <li>No docs yet.</li>\n    </ul>"

    index = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>{page_style}
    h1 {{ color: {c['h1']}; border-bottom: 2px solid {c['hr']}; padding-bottom: 0.3em; }}
    h2 {{ color: {c['h2']}; font-size: 14px; margin: 1.2em 0 0.3em 0; }}
    li {{ margin: 0.3em 0; }}
    .count {{ color: {c['hr']}; font-size: 12px; }}
    .back {{ color: {c['hr']}; font-size: 12px; }}
</style>
</head>
<body>
    <p class="back"><a href="../index.html">back</a></p>
    <h1>{project_name}</h1>
    <p class="count">{total} documents</p>
{entries_html}
</body>
</html>"""

    index_path = project_dir / "index.html"
    index_path.write_text(index)
    return str(index_path)


def build_top_index() -> str:
    """Build top-level index.html linking to all project folders."""
    page_style = get_page_style(DEFAULT_STYLE)
    c = DEFAULT_STYLE["colors"]

    project_dirs = sorted(
        d for d in DOCS_DIR.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )

    entries = []
    for d in project_dirs:
        doc_count = len(list(d.glob("*.html"))) - (1 if (d / "index.html").exists() else 0)
        entries.append(
            f'        <li><a href="{d.name}/index.html">{d.name.upper()}</a>'
            f' <span class="count">({doc_count} docs)</span></li>'
        )

    entries_html = "\n".join(entries) if entries else "        <li>No projects yet.</li>"

    index = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>{page_style}
    h1 {{ color: {c['h1']}; border-bottom: 2px solid {c['hr']}; padding-bottom: 0.3em; }}
    li {{ margin: 0.8em 0; font-size: 16px; }}
    .count {{ color: {c['hr']}; font-size: 12px; }}
</style>
</head>
<body>
    <h1>Docs</h1>
    <ul>
{entries_html}
    </ul>
</body>
</html>"""

    index_path = DOCS_DIR / "index.html"
    index_path.write_text(index)
    return str(index_path)


def main():
    parser = argparse.ArgumentParser(description="Render markdown as styled output")
    parser.add_argument("project", nargs="?", help="Project folder name (e.g. rl, terminal2f)")
    parser.add_argument("file", nargs="?", help="Markdown file to render")
    parser.add_argument("--section", help='Section heading to extract')
    parser.add_argument("--width", type=int, default=100, help="Console width (default 100)")
    parser.add_argument("--format", choices=["svg", "html", "png"], default="html", help="Output format")
    parser.add_argument("--index", action="store_true", help="Build all index pages")
    parser.add_argument("--all", action="store_true", help="Render all .md in project and rebuild indexes")
    args = parser.parse_args()

    DOCS_DIR.mkdir(exist_ok=True)

    # Just rebuild indexes
    if args.index and not args.project:
        for d in DOCS_DIR.iterdir():
            if d.is_dir() and not d.name.startswith("."):
                build_project_index(d)
        path = build_top_index()
        print(f"Index: {path}")
        return

    # Render all in a project
    if args.all:
        if not args.project:
            parser.error("--all requires a project name")
        project_dir = DOCS_DIR / args.project
        project_dir.mkdir(exist_ok=True)
        style = load_project_style(project_dir)
        if args.width != 100:
            style["layout"]["width"] = args.width
        md_files = sorted(project_dir.glob("*.md"))
        if not md_files:
            print(f"No .md files in {project_dir}")
            return
        for md_file in md_files:
            text = clean_text(md_file.read_text())
            render_to_html(text, project_dir, md_file.stem, style)
            print(f"  {md_file.name} -> {md_file.stem}.html")
        build_project_index(project_dir)
        build_top_index()
        print(f"Done. Open: {project_dir / 'index.html'}")
        return

    # Render single file
    if not args.project or not args.file:
        parser.error("need: project file.md (or use --all / --index)")

    project_dir = DOCS_DIR / args.project
    project_dir.mkdir(exist_ok=True)
    style = load_project_style(project_dir)
    if args.width != 100:
        style["layout"]["width"] = args.width

    text = Path(args.file).read_text()
    text = clean_text(text)
    if args.section:
        text = extract_section(text, args.section)

    stem = make_stem(args.file, args.section)

    if args.format == "html":
        path = render_to_html(text, project_dir, stem, style)
        print(f"HTML: {path}")
    elif args.format == "svg":
        path = render_to_svg(text, project_dir, stem, style)
        print(f"SVG: {path}")
    elif args.format == "png":
        svg_path = render_to_svg(text, project_dir, stem, style)
        png_path = svg_to_png(svg_path)
        print(f"PNG: {png_path}")


if __name__ == "__main__":
    main()
