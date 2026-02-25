"""
Render markdown files as styled HTML, SVG, or PNG.

Output goes to assets/docs/<project>/

Usage:
    uv run assets/md_to_image.py rl grpo_example.md
    uv run assets/md_to_image.py rl myfile.md --section "## Training"
    uv run assets/md_to_image.py rl --all
    uv run assets/md_to_image.py --index
    uv run --with rich assets/md_to_image.py rl myfile.md --format svg
"""

import argparse
import html as html_mod
import re
import sys
import tomllib
from pathlib import Path

ASSETS_DIR = Path(__file__).parent
DOCS_DIR = ASSETS_DIR / "docs"

EDITORIAL_CSS = """\
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500&display=swap');

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif;
    font-size: 14.5px;
    line-height: 1.72;
    color: #2a2a2a;
    background: #fdfcfa;
    max-width: 800px;
    margin: 0 auto;
    padding: 72px 1rem 120px;
    -webkit-font-smoothing: antialiased;
  }

  a.back {
    font-family: 'JetBrains Mono', monospace; font-size: 9.5px;
    color: #bbb; text-decoration: none; letter-spacing: 0.5px;
  }
  a.back:hover { color: #1a1a1a; }

  .memo-head {
    border-top: 2.5px solid #1a1a1a;
    border-bottom: 0.5px solid #1a1a1a;
    padding: 20px 0 16px;
    margin-bottom: 40px;
  }
  .memo-head h1 {
    font-size: 1.6rem;
    font-weight: normal;
    color: #333;
    line-height: 1.25;
    margin: 0;
  }

  h2 {
    font-size: 1.3rem;
    font-weight: normal;
    color: #333;
    margin: 56px 0 6px;
  }
  h3 {
    font-size: 1.1rem;
    font-weight: normal;
    color: #333;
    margin: 36px 0 4px;
  }
  p { margin: 10px 0; }

  ul, ol { margin: 12px 0 12px 22px; }
  li { margin: 5px 0; color: #333; }
  li::marker { color: #bbb; }

  code {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #1a1a1a;
    background: #f5f4f0;
    padding: 1.5px 5px;
    border-bottom: 1px solid #ddd;
  }
  pre {
    margin: 20px 0;
    padding: 16px 20px;
    background: #f5f4f0;
    border: 0.5px solid #e8e4de;
    overflow-x: auto;
    line-height: 1.5;
  }
  pre code {
    background: none;
    padding: 0;
    border: none;
    font-size: 12px;
  }
  strong { font-weight: 700; color: #1a1a1a; }
  em { color: #555; }
  a { color: #268bd2; text-decoration: none; }
  a:hover { text-decoration: underline; }

  blockquote {
    font-size: 13px;
    color: #888;
    font-style: italic;
    margin: 20px 0 20px 28px;
    padding-left: 16px;
    border-left: 1.5px solid #ddd;
    line-height: 1.6;
  }

  hr {
    border: none;
    text-align: center;
    margin: 56px 0;
    line-height: 0;
  }
  hr::after {
    content: '\\2022\\2009\\2009\\2022\\2009\\2009\\2022';
    font-size: 10px;
    color: #ccc;
    letter-spacing: 4px;
  }

  .colophon {
    border-top: 0.5px solid #ddd;
    margin-top: 64px;
    padding-top: 16px;
  }
  .colophon p {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    color: #ccc;
    letter-spacing: 0.5px;
  }
"""


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


# ── Markdown to HTML ──


def _inline(text: str) -> str:
    """Convert inline markdown (bold, italic, code, links)."""
    # Bold + italic first (before code spans, so **`code`** works)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)

    # Code spans
    parts = []
    last = 0
    for m in re.finditer(r"`([^`]+)`", text):
        parts.append(text[last:m.start()])
        parts.append(f"<code>{html_mod.escape(m.group(1))}</code>")
        last = m.end()
    parts.append(text[last:])
    return "".join(parts)


def _md_to_html(lines: list[str]) -> str:
    """Simple markdown to semantic HTML converter."""
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # Code block
        if line.startswith("```"):
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1  # skip closing ```
            code = html_mod.escape("\n".join(code_lines))
            out.append(f"<pre><code>{code}</code></pre>")
            continue

        # Headings
        if line.startswith("### "):
            out.append(f"<h3>{_inline(line[4:])}</h3>")
            i += 1
            continue
        if line.startswith("## "):
            out.append(f"<h2>{_inline(line[3:])}</h2>")
            i += 1
            continue

        # Horizontal rule
        if line.strip() in ("---", "***", "___"):
            out.append("<hr>")
            i += 1
            continue

        # Unordered list
        if re.match(r"^[\-\*] ", line):
            items = []
            while i < len(lines) and re.match(r"^[\-\*] ", lines[i]):
                items.append(f"<li>{_inline(lines[i][2:])}</li>")
                i += 1
            out.append("<ul>\n" + "\n".join(items) + "\n</ul>")
            continue

        # Ordered list
        if re.match(r"^\d+\. ", line):
            items = []
            while i < len(lines) and re.match(r"^\d+\. ", lines[i]):
                text = re.sub(r"^\d+\. ", "", lines[i])
                items.append(f"<li>{_inline(text)}</li>")
                i += 1
            out.append("<ol>\n" + "\n".join(items) + "\n</ol>")
            continue

        # Blockquote
        if line.startswith("> "):
            bq_lines = []
            while i < len(lines) and lines[i].startswith("> "):
                bq_lines.append(lines[i][2:])
                i += 1
            out.append(f"<blockquote><p>{_inline(' '.join(bq_lines))}</p></blockquote>")
            continue

        # Empty line
        if not line.strip():
            i += 1
            continue

        # Paragraph
        para_lines = []
        while (i < len(lines) and lines[i].strip()
               and not lines[i].startswith("#")
               and not lines[i].startswith("```")
               and not lines[i].strip() in ("---", "***", "___")
               and not re.match(r"^[\-\*] ", lines[i])
               and not re.match(r"^\d+\. ", lines[i])
               and not lines[i].startswith("> ")):
            para_lines.append(lines[i])
            i += 1
        out.append(f"<p>{_inline(' '.join(para_lines))}</p>")

    return "\n\n".join(out)


def render_to_html(text: str, output_dir: Path, stem: str) -> str:
    """Render markdown to editorial-style HTML."""
    lines = text.strip().split("\n")

    # Extract title from first heading
    title = stem.replace("_", " ").title()
    body_lines = lines
    if lines and lines[0].startswith("#"):
        title = lines[0].lstrip("#").strip()
        body_lines = lines[1:]

    body_html = _md_to_html(body_lines)

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html_mod.escape(title)}</title>
<style>
{EDITORIAL_CSS}
</style>
</head>
<body>

<a class="back" href="index.html">&larr; docs</a>

<div class="memo-head">
  <h1>{html_mod.escape(title)}</h1>
</div>

{body_html}

<div class="colophon">
  <p>{html_mod.escape(title).upper()}</p>
</div>

</body>
</html>"""

    html_path = output_dir / f"{stem}.html"
    html_path.write_text(full_html)
    return str(html_path)


# ── Rich (terminal-style) ──


def render_to_rich(text: str, output_dir: Path, stem: str) -> str:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.theme import Theme

    theme = Theme({
        "markdown.h1": "bold #cb4b16",
        "markdown.h2": "bold #268bd2",
        "markdown.h3": "bold #2aa198",
        "markdown.code": "#d33682",
        "markdown.code_block": "#586e75",
        "markdown.link": "#268bd2 underline",
        "markdown.item.bullet": "#cb4b16",
        "markdown.item.number": "#cb4b16",
        "markdown.bold": "bold #073642",
        "markdown.italic": "italic #586e75",
        "markdown.hr": "#93a1a1",
    })
    console = Console(record=True, width=100, theme=theme)
    console.print(Markdown(text, code_theme="solarized-light"))
    html_content = console.export_html()

    html_content = re.sub(
        r'background-color: #[0-9a-fA-F]{6}',
        'background-color: #f5f5f0',
        html_content,
    )

    full_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
    body {{
        background: #ffffff;
        color: #1e1e1e;
        font-family: 'Fira Code', 'Cascadia Code', 'JetBrains Mono', 'Menlo', monospace;
        font-size: 14px;
        line-height: 1.35;
        padding: 0.5em 2em;
        max-width: 960px;
        margin: 0 auto;
    }}
    pre {{
        white-space: pre-wrap;
        word-wrap: break-word;
        margin: 0;
    }}
    ::selection {{
        background: #d0e8ff;
        color: #1e1e1e;
    }}
    a {{ color: #268bd2; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .back {{ color: #93a1a1; font-size: 12px; margin-bottom: 0; }}
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


# ── SVG / PNG (Rich) ──


def render_to_svg(text: str, output_dir: Path, stem: str) -> str:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.theme import Theme

    theme = Theme({
        "markdown.h1": "bold #cb4b16",
        "markdown.h2": "bold #268bd2",
        "markdown.h3": "bold #2aa198",
        "markdown.code": "#d33682",
    })
    console = Console(record=True, width=100, theme=theme)
    console.print(Markdown(text, code_theme="solarized-light"))
    svg_path = output_dir / f"{stem}.svg"
    console.save_svg(str(svg_path), title=stem)
    return str(svg_path)


def svg_to_png(svg_path: str) -> str:
    import cairosvg
    png_path = svg_path.replace(".svg", ".png")
    cairosvg.svg2png(url=svg_path, write_to=png_path, scale=2)
    return png_path


# ── Index pages ──


def _stem_to_title(stem: str) -> str:
    return stem.replace("_", " ").title()


def build_project_index(project_dir: Path) -> str:
    """Build index.html for a single project folder."""
    html_files = {f.stem: f for f in project_dir.glob("*.html") if f.name != "index.html"}
    total = len(html_files)
    project_name = project_dir.name.upper()

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
<html lang="en">
<head>
<meta charset="utf-8">
<title>{project_name}</title>
<style>
{EDITORIAL_CSS}
</style>
</head>
<body>
    <a class="back" href="../index.html">&larr; docs</a>
    <div class="memo-head">
      <h1>{project_name}</h1>
    </div>
    <p style="font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #999;">{total} documents</p>
{entries_html}
</body>
</html>"""

    index_path = project_dir / "index.html"
    index_path.write_text(index)
    return str(index_path)


def build_top_index() -> str:
    """Build top-level index.html linking to all project folders."""
    project_dirs = sorted(
        d for d in DOCS_DIR.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )

    entries = []
    for d in project_dirs:
        doc_count = len(list(d.glob("*.html"))) - (1 if (d / "index.html").exists() else 0)
        entries.append(
            f'        <li><a href="{d.name}/index.html">{d.name.upper()}</a>'
            f' <span style="font-family: \'JetBrains Mono\', monospace; font-size: 10px; color: #999;">({doc_count} docs)</span></li>'
        )

    entries_html = "\n".join(entries) if entries else "        <li>No projects yet.</li>"

    index = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Docs</title>
<style>
{EDITORIAL_CSS}
</style>
</head>
<body>
    <div class="memo-head">
      <h1>Docs</h1>
    </div>
    <ul>
{entries_html}
    </ul>
</body>
</html>"""

    index_path = DOCS_DIR / "index.html"
    index_path.write_text(index)
    return str(index_path)


# ── CLI ──


def main():
    parser = argparse.ArgumentParser(description="Render markdown as styled output")
    parser.add_argument("project", nargs="?", help="Project folder name (e.g. rl)")
    parser.add_argument("file", nargs="?", help="Markdown file to render")
    parser.add_argument("--section", help='Section heading to extract')
    parser.add_argument("--format", choices=["html", "rich", "svg", "png"], default="html", help="Output format (rich/svg/png require rich)")
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
        md_files = sorted(project_dir.glob("*.md"))
        if not md_files:
            print(f"No .md files in {project_dir}")
            return

        # Load handcrafted skip list from index.toml
        skip_stems: set[str] = set()
        index_toml = project_dir / "index.toml"
        if index_toml.exists():
            config = tomllib.loads(index_toml.read_text())
            for group in config.get("groups", []):
                for entry in group.get("docs", []):
                    if isinstance(entry, dict) and entry.get("handcrafted"):
                        skip_stems.add(entry["file"])

        render = render_to_rich if args.format == "rich" else render_to_html
        for md_file in md_files:
            if md_file.stem in skip_stems:
                print(f"  {md_file.name} -> SKIP (handcrafted)")
                continue
            text = clean_text(md_file.read_text())
            render(text, project_dir, md_file.stem)
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

    text = Path(args.file).read_text()
    text = clean_text(text)
    if args.section:
        text = extract_section(text, args.section)

    stem = make_stem(args.file, args.section)

    if args.format == "html":
        path = render_to_html(text, project_dir, stem)
        print(f"HTML: {path}")
    elif args.format == "rich":
        path = render_to_rich(text, project_dir, stem)
        print(f"Rich: {path}")
    elif args.format == "svg":
        path = render_to_svg(text, project_dir, stem)
        print(f"SVG: {path}")
    elif args.format == "png":
        svg_path = render_to_svg(text, project_dir, stem)
        png_path = svg_to_png(svg_path)
        print(f"PNG: {png_path}")


if __name__ == "__main__":
    main()