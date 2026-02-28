import asyncio
import base64
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from fastmcp import Client

client = Client("./example_drawing_server.py")

# simple HTML page that auto-refreshes the image
VIEWER_HTML = """\
<!DOCTYPE html>
<html>
<head><title>Canvas</title></head>
<body style="margin:0; display:flex; justify-content:center; align-items:center; height:100vh; background:#111;">
  <img id="canvas" src="house.png" style="border:1px solid #333;" />
  <script>setInterval(() => { document.getElementById('canvas').src = 'house.png?' + Date.now(); }, 300);</script>
</body>
</html>
"""

def save_image(result):
    img_data = base64.b64decode(result.content[0].data)
    with open("house.png", "wb") as f:
        f.write(img_data)

def start_server():
    with open("canvas_viewer.html", "w") as f:
        f.write(VIEWER_HTML)
    server = HTTPServer(("localhost", 8888), SimpleHTTPRequestHandler)
    server.serve_forever()

async def main():
    # start live viewer in background
    threading.Thread(target=start_server, daemon=True).start()
    print("Live viewer: http://localhost:8888/canvas_viewer.html\n")

    async with client:
        steps = [
            ("draw_line",   {"x1": 100, "y1": 300, "x2": 100, "y2": 150}, "left wall"),
            ("draw_line",   {"x1": 300, "y1": 300, "x2": 300, "y2": 150}, "right wall"),
            ("draw_line",   {"x1": 100, "y1": 300, "x2": 300, "y2": 300}, "floor"),
            ("draw_line",   {"x1": 100, "y1": 150, "x2": 200, "y2": 50},  "roof left"),
            ("draw_line",   {"x1": 300, "y1": 150, "x2": 200, "y2": 50},  "roof right"),
            ("draw_circle", {"cx": 200, "cy": 240, "radius": 20, "color": "blue"}, "window"),
        ]

        for tool, args, label in steps:
            result = await client.call_tool(tool, args)
            save_image(result)
            print(f"  {label}")
            await asyncio.sleep(1)

        print("\nDone. Ctrl+C to stop viewer.")
        await asyncio.sleep(999)

asyncio.run(main())
