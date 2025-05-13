from http.server import HTTPServer, SimpleHTTPRequestHandler
import os

class WasmHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        return super().end_headers()

    def do_GET(self):
        # Set correct MIME type for WASM files
        if self.path.endswith('.wasm'):
            self.send_response(200)
            self.send_header('Content-type', 'application/wasm')
            self.end_headers()
            # Get the absolute path to the project root (two directories up from this script)
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            file_path = os.path.join(project_root, self.path[1:])
            with open(file_path, 'rb') as f:
                self.wfile.write(f.read())
            return
        return super().do_GET()

if __name__ == '__main__':
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    os.chdir(project_root)
    server = HTTPServer(('localhost', 8000), WasmHandler)
    print("Server running at http://localhost:8000")
    print(f"Serving from directory: {project_root}")
    server.serve_forever() 