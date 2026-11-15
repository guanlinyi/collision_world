"""Loopback-only, read-only numerical replay server; Python standard library."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / 'inspect_predictions/web'


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(302)
            self.send_header('Location', '/inspect_predictions/web/index.html')
            self.end_headers()
            return
        return super().do_GET()

    def list_directory(self, path):
        self.send_error(404)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8882)
    args = parser.parse_args()
    with ThreadingHTTPServer(('127.0.0.1', args.port), partial(Handler, directory=str(ROOT))) as server:
        print(f'Open http://127.0.0.1:{server.server_port}/ ; Ctrl+C stops the server.', flush=True)
        try: server.serve_forever()
        except KeyboardInterrupt: pass


if __name__ == '__main__': main()
