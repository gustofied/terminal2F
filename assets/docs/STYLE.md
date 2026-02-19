# Style Guide

## Text Rules
- No em dashes, en dashes, or figure dashes. Use hyphens (-) instead.
- No smart quotes. Use straight quotes (' and ").
- Keep lines under 80 chars in source markdown for clean rendering.

## Default Theme
- White background with Solarized color accents
- Background: #ffffff
- Text: #1e1e1e
- Headings: orange (#cb4b16) h1, blue (#268bd2) h2, cyan (#2aa198) h3, green (#859900) h4
- Code inline: magenta (#d33682)
- Code blocks: solarized-light theme, #f5f5f0 background
- Tables: blue headers (#268bd2), gray borders (#93a1a1)
- Bullets: orange (#cb4b16)
- Bold: dark (#073642)
- Links: blue (#268bd2)

## Per-Project Overrides
Drop a `style.toml` in any project folder to override defaults:

```toml
code_theme = "monokai"

[colors]
h1 = "#e74c3c"
background = "#fafafa"

[font]
size = "13px"

[layout]
width = 120
```

Only include keys you want to change - everything else stays default.
