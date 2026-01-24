import glob as globlib
import os
import re
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_in_repo(user_path: str) -> Path:
    root = _repo_root()

    s = (user_path or ".").strip()
    p = Path(s)
    candidate = p if p.is_absolute() else (root / p)

    resolved = candidate.resolve(strict=False)

    try:
        ok = resolved.is_relative_to(root) 
    except AttributeError:
        try:
            resolved.relative_to(root)
            ok = True
        except ValueError:
            ok = False

    if not ok:
        raise PermissionError("path outside repo")

    return resolved


def read(path: str, offset: int = 0, limit: int | None = None) -> str:
    p = _resolve_in_repo(path)

    offset = max(int(offset or 0), 0)
    lines = open(p, encoding="utf-8", errors="replace").readlines()

    if limit is None:
        limit = len(lines)
    else:
        limit = max(int(limit or 0), 0)

    selected = lines[offset : offset + limit]
    return "".join(f"{offset + idx + 1:4}| {line}" for idx, line in enumerate(selected))


def write(path: str, content: str) -> str:
    p = _resolve_in_repo(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(content or "")
    return "ok"


def edit(path: str, old: str, new: str, all: bool = False) -> str:
    p = _resolve_in_repo(path)
    text = open(p, encoding="utf-8", errors="replace").read()

    if old not in text:
        return "error: old_string not found"

    count = text.count(old)
    if (not all) and count > 1:
        return f"error: old_string appears {count} times, must be unique (use all=true)"

    replacement = text.replace(old, new) if all else text.replace(old, new, 1)
    with open(p, "w", encoding="utf-8") as f:
        f.write(replacement)
    return "ok"


def glob(pat: str, path: str = ".") -> str:
    base = _resolve_in_repo(path)
    pattern = f"{base}/{pat}".replace("//", "/")

    files = globlib.glob(pattern, recursive=True)
    files = sorted(
        files,
        key=lambda f: os.path.getmtime(f) if os.path.isfile(f) else 0,
        reverse=True,
    )
    return "\n".join(files) or "none"


def grep(pat: str, path: str = ".") -> str:
    base = _resolve_in_repo(path)
    rx = re.compile(pat)

    hits: list[str] = []
    for filepath in globlib.glob(f"{base}/**", recursive=True):
        try:
            if not os.path.isfile(filepath):
                continue
            with open(filepath, encoding="utf-8", errors="replace") as f:
                for line_num, line in enumerate(f, 1):
                    if rx.search(line):
                        hits.append(f"{filepath}:{line_num}:{line.rstrip()}")
                        if len(hits) >= 50:
                            return "\n".join(hits)
        except Exception:
            pass

    return "\n".join(hits) or "none"


def bash(cmd: str) -> str:
    return "error: bash tool is disabled (locked)"


names_to_functions = {
    "read": read,
    "write": write,
    "edit": edit,
    "glob": glob,
    "grep": grep,
    "bash": bash,  # currently commented out
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read file with line numbers (repo-only)",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path (repo-only)"},
                    "offset": {"type": "integer", "description": "Line offset (0-based)"},
                    "limit": {"type": "integer", "description": "Max lines to read"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write",
            "description": "Write content to file (repo-only)",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path (repo-only)"},
                    "content": {"type": "string", "description": "File content"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit",
            "description": "Replace old with new in file (repo-only; old must be unique unless all=true)",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path (repo-only)"},
                    "old": {"type": "string", "description": "Old string"},
                    "new": {"type": "string", "description": "New string"},
                    "all": {"type": "boolean", "description": "Replace all occurrences"},
                },
                "required": ["path", "old", "new"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Find files by pattern, sorted by mtime (repo-only)",
            "parameters": {
                "type": "object",
                "properties": {
                    "pat": {"type": "string", "description": "Glob pattern"},
                    "path": {"type": "string", "description": "Base path (repo-only)"},
                },
                "required": ["pat"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search files for regex pattern (repo-only)",
            "parameters": {
                "type": "object",
                "properties": {
                    "pat": {"type": "string", "description": "Regex pattern"},
                    "path": {"type": "string", "description": "Base path (repo-only)"},
                },
                "required": ["pat"],
            },
        },
    },
    #     {
    #     "type": "function",
    #     "function": {
    #         "name": "bash",
    #         "description": "Run shell command",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "cmd": {"type": "string", "description": "Shell command"},
    #             },
    #             "required": ["cmd"],
    #         },
    #     },
    # },
]
