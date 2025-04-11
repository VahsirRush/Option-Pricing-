import http.server
import socketserver
import os
import webbrowser
import threading
import time

PORT = 8080

class MyHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=os.path.dirname(__file__), **kwargs)

def open_browser():
    """Open the browser after a short delay to ensure the server is running."""
    time.sleep(1.5)
    webbrowser.open(f"http://localhost:{PORT}")

def main():
    """Start the HTTP server and open the browser."""
    # Change to the directory containing the HTML file
    os.chdir(os.path.dirname(__file__))
    
    # Start the browser in a separate thread
    threading.Thread(target=open_browser).start()
    
    # Start the HTTP server
    with socketserver.TCPServer(("", PORT), MyHandler) as httpd:
        print(f"Serving landing page at http://localhost:{PORT}")
        httpd.serve_forever()

if __name__ == "__main__":
    main() 