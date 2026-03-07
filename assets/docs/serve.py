"""Tiny dev server that serves HTML and saves edits via POST."""
import http.server
import os

PORT = 8080
ROOT = os.path.dirname(os.path.abspath(__file__))


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    def do_POST(self):
        path = os.path.join(ROOT, self.path.lstrip("/"))
        if not path.endswith(".html") or not os.path.isfile(path):
            self.send_error(400, "Bad path")
            return
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8")
        with open(path, "w") as f:
            f.write(body)
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(b"saved")
        print(f"  saved {self.path} ({len(body)} chars)")


print(f"serving {ROOT} on http://localhost:{PORT}")
http.server.HTTPServer(("", PORT), Handler).serve_forever()
