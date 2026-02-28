from fastmcp import FastMCP
from fastmcp.server.apps import AppConfig, ResourceCSP
from fastmcp.tools import ToolResult
from mcp import types
from PIL import Image, ImageDraw
import base64
import io
import json

mcp = FastMCP("Drawing Server")

VIEW_URI = "ui://drawing-server/canvas.html"

canvas = Image.new("RGB", (400, 400), "white")
draw_history = []

def _canvas_b64() -> str:
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

@mcp.tool(app=AppConfig(resource_uri=VIEW_URI))
def draw_line(x1: int, y1: int, x2: int, y2: int, color: str = "black", width: int = 2) -> ToolResult:
    """Draw a line on the canvas."""
    draw = ImageDraw.Draw(canvas)
    draw.line([(x1, y1), (x2, y2)], fill=color, width=width)
    draw_history.append({"type": "line", "x1": x1, "y1": y1, "x2": x2, "y2": y2, "color": color})
    return ToolResult(content=[
        types.ImageContent(type="image", data=_canvas_b64(), mimeType="image/png")
    ])

@mcp.tool(app=AppConfig(resource_uri=VIEW_URI))
def draw_circle(cx: int, cy: int, radius: int, color: str = "black", width: int = 2) -> ToolResult:
    """Draw a circle on the canvas."""
    draw = ImageDraw.Draw(canvas)
    draw.ellipse([(cx - radius, cy - radius), (cx + radius, cy + radius)], outline=color, width=width)
    draw_history.append({"type": "circle", "cx": cx, "cy": cy, "r": radius, "color": color})
    return ToolResult(content=[
        types.ImageContent(type="image", data=_canvas_b64(), mimeType="image/png")
    ])

@mcp.tool(app=AppConfig(resource_uri=VIEW_URI))
def screenshot() -> ToolResult:
    """Take a screenshot of the canvas."""
    return ToolResult(content=[
        types.ImageContent(type="image", data=_canvas_b64(), mimeType="image/png")
    ])

@mcp.tool
def clear() -> str:
    """Clear the canvas to white."""
    global canvas
    canvas = Image.new("RGB", (400, 400), "white")
    draw_history.clear()
    return "Canvas cleared"

@mcp.resource(
    VIEW_URI,
    app=AppConfig(csp=ResourceCSP(resource_domains=["https://unpkg.com"])),
)
def canvas_view() -> str:
    """Live canvas viewer."""
    return """\
<!DOCTYPE html>
<html>
<head>
  <meta name="color-scheme" content="light dark">
  <style>
    body { display: flex; justify-content: center; align-items: center;
           height: 420px; width: 420px; margin: 0; background: transparent; }
    img  { width: 400px; height: 400px; border: 1px solid #ccc; }
  </style>
</head>
<body>
  <img id="canvas" alt="Canvas" />
  <script type="module">
    import { App } from
      "https://unpkg.com/@modelcontextprotocol/ext-apps@0.4.0/app-with-deps";

    const app = new App({ name: "Canvas View", version: "1.0.0" });

    app.ontoolresult = ({ content }) => {
      const img = content?.find(c => c.type === 'image');
      if (img) {
        document.getElementById('canvas').src =
          `data:${img.mimeType};base64,${img.data}`;
      }
    };

    await app.connect();
  </script>
</body>
</html>"""

if __name__ == "__main__":
    mcp.run()
