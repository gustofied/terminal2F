"""Dev server for docs with inline editing support.
Handles GET (serve files) and POST (save edits).

Usage: python3 serve.py
"""

import os
from http.server import HTTPServer, SimpleHTTPRequestHandler


class EditHandler(SimpleHTTPRequestHandler):
    def do_POST(self):
        path = self.translate_path(self.path)
        if not path.endswith(".html"):
            self.send_error(403, "only .html files can be saved")
            return
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        with open(path, "wb") as f:
            f.write(body)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    server = HTTPServer(("127.0.0.1", 8080), EditHandler)
    print("docs server on http://localhost:8080 (edit + save enabled)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
