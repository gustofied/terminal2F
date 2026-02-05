from __future__ import annotations

import json
from pathlib import Path

import jinja2

TEMPLATE_DIR = Path(__file__).parent / "templates"
DATA_DIR = Path(__file__).parent / "data"


def render_cv(data_file: str = "cv.json", output: str = "cv.html") -> None:
    data = json.loads((DATA_DIR / data_file).read_text())
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(TEMPLATE_DIR))
    template = env.get_template("cv.html.j2")
    Path(output).write_text(template.render(**data))
    print(f"Generated {output}")


if __name__ == "__main__":
    render_cv()
