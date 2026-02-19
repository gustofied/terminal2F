# Docs

Rendered markdown docs, organized by project. Each project gets its own folder with HTML output, an index page, and optional style overrides.

## Structure

```
assets/
  md_to_image.py          -- the renderer
  docs/
    README.md             -- this file
    STYLE.md              -- default theme + style rules
    index.html            -- top-level index (auto-generated)
    rl/                   -- RL training project
      style.toml          -- project-specific style overrides (optional)
      grpo_example.md     -- source markdown
      grpo_example.html   -- rendered output
      index.html          -- project index (auto-generated)
    terminal2f/           -- another project (example)
      ...
```

## How It Works

1. Write markdown files and drop them in `docs/<project>/`
2. Run the renderer to produce styled HTML
3. Open the HTML in a browser, copy-paste into Google Docs for sharing

The renderer uses Rich to convert markdown into styled terminal output, then exports that as HTML with a clean white background and Solarized color accents. Each project can override colors, fonts, and layout via `style.toml`.

## Usage

```bash
# Render a single file
uv run --with rich assets/md_to_image.py rl grpo_example.md

# Render a specific section
uv run --with rich assets/md_to_image.py rl myfile.md --section "## Training"

# Render all .md files in a project
uv run --with rich assets/md_to_image.py rl --all

# Rebuild all index pages
uv run --with rich assets/md_to_image.py --index

# SVG output
uv run --with rich assets/md_to_image.py rl myfile.md --format svg

# PNG output (needs cairosvg)
uv run --with rich,cairosvg assets/md_to_image.py rl myfile.md --format png

# Custom width
uv run --with rich assets/md_to_image.py rl myfile.md --width 120
```

## Adding a New Project

```bash
mkdir assets/docs/myproject
# drop .md files in there
uv run --with rich assets/md_to_image.py myproject --all
```

Optionally add a `style.toml` for custom colors - see STYLE.md.
