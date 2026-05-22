import http.server
import socketserver
from pathlib import Path

PORT = 8080
ROOT = Path(__file__).resolve().parent

class QuietHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        pass

if __name__ == '__main__':
    import os

    os.chdir(ROOT)
    handler = QuietHTTPRequestHandler
    with socketserver.TCPServer(('localhost', PORT), handler) as httpd:
        print(f'Serving frontend at http://localhost:{PORT}')
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print('\nShutting down server.')
            httpd.server_close()
