"""
tabular/dashboard/app.py
------------------------
Lightweight standalone Python Web Server for the Visual Control Dashboard.
Zero external dependencies (uses standard library http.server and subprocess).

Endpoints:
  GET  /              -> Serves index.html
  GET  /styles.css    -> Serves styles.css
  GET  /app.js        -> Serves app.js
  GET  /api/config    -> Reads tabular/config.json
  POST /api/config    -> Saves tabular/config.json
  GET  /api/pipeline  -> Reads tabular/pipeline_order.json
  POST /api/pipeline  -> Saves tabular/pipeline_order.json
  GET  /api/results   -> Reads tabular/outputs/pipeline_results.json
  POST /api/run       -> Starts pipeline execution in background
  GET  /api/status    -> Checks running status, tasks status & metrics
  POST /api/stop      -> Terminates running pipeline execution
"""

import os
import sys
import json
import time
import queue
import threading
import subprocess
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

REPO_ROOT = str(Path(__file__).resolve().parents[2])
DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(REPO_ROOT, "tabular", "config.json")
PIPELINE_PATH = os.path.join(REPO_ROOT, "tabular", "pipeline_order.json")
RESULTS_PATH = os.path.join(REPO_ROOT, "tabular", "outputs", "pipeline_results.json")
RUNNER_SCRIPT = os.path.join(REPO_ROOT, "tabular", "run_pipeline.py")

# Global Process State
current_process = None
process_lock = threading.Lock()
execution_logs = []
process_start_time = None
process_is_running = False
process_exit_code = None


def get_python_executable():
    """Returns the python executable containing torch, checking virtualenvs if needed."""
    try:
        import torch
        return sys.executable
    except ImportError:
        pass

    candidates = [
        # Windows conda paths
        os.path.expanduser(r"~\.conda\envs\few_shot_meta_learning\python.exe"),
        os.path.expanduser(r"~\anaconda3\envs\few_shot_meta_learning\python.exe"),
        os.path.expanduser(r"~\miniconda3\envs\few_shot_meta_learning\python.exe"),
        # macOS / Linux conda paths
        os.path.expanduser("~/.conda/envs/few_shot_meta_learning/bin/python"),
        os.path.expanduser("~/anaconda3/envs/few_shot_meta_learning/bin/python"),
        os.path.expanduser("~/miniconda3/envs/few_shot_meta_learning/bin/python"),
        os.path.expanduser("~/miniforge3/envs/few_shot_meta_learning/bin/python"),
        os.path.expanduser("~/opt/anaconda3/envs/few_shot_meta_learning/bin/python"),
        "/opt/homebrew/Caskroom/miniforge/base/envs/few_shot_meta_learning/bin/python",
        "/opt/homebrew/anaconda3/envs/few_shot_meta_learning/bin/python",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return sys.executable



def log_listener(proc):
    """Reads stdout and stderr from the subprocess and buffers into execution_logs."""
    global process_is_running, process_exit_code
    for line in iter(proc.stdout.readline, ''):
        with process_lock:
            execution_logs.append(line)
    proc.stdout.close()
    proc.wait()
    with process_lock:
        process_is_running = False
        process_exit_code = proc.returncode


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""
    daemon_threads = True
    allow_reuse_address = True


class DashboardHandler(BaseHTTPRequestHandler):
    def _set_headers(self, content_type="application/json", status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(status=200)

    def do_GET(self):
        url_path = self.path.split("?")[0]

        if url_path == "/" or url_path == "/index.html":
            file_path = os.path.join(DASHBOARD_DIR, "index.html")
            self._serve_file(file_path, "text/html; charset=utf-8")
        elif url_path == "/styles.css":
            file_path = os.path.join(DASHBOARD_DIR, "styles.css")
            self._serve_file(file_path, "text/css")
        elif url_path == "/app.js":
            file_path = os.path.join(DASHBOARD_DIR, "app.js")
            self._serve_file(file_path, "application/javascript")
        elif url_path == "/api/config":
            self._serve_json_file(CONFIG_PATH)
        elif url_path == "/api/pipeline":
            self._serve_json_file(PIPELINE_PATH)
        elif url_path == "/api/results":
            self._serve_json_file(RESULTS_PATH, default={"tasks_status": [], "metrics": []})
        elif url_path == "/api/status":
            self._handle_status()
        else:
            self._set_headers("text/plain", 404)
            self.wfile.write(b"Not Found")

    def do_POST(self):
        url_path = self.path.split("?")[0]
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"

        try:
            data = json.loads(body) if body else {}
        except Exception as e:
            self._set_headers(status=400)
            self.wfile.write(json.dumps({"error": f"Invalid JSON: {e}"}).encode("utf-8"))
            return

        if url_path == "/api/config":
            self._save_json_file(CONFIG_PATH, data)
        elif url_path == "/api/pipeline":
            self._save_json_file(PIPELINE_PATH, data)
        elif url_path == "/api/results/clear":
            self._save_json_file(RESULTS_PATH, {"tasks_status": [], "metrics": []})
        elif url_path in ("/api/terminal/clear", "/api/logs/clear"):
            global execution_logs
            with process_lock:
                execution_logs = []
            self._set_headers(status=200)
            self.wfile.write(json.dumps({"status": "cleared", "message": "Terminal logs cleared"}).encode("utf-8"))
        elif url_path == "/api/run":
            self._handle_run(data)
        elif url_path == "/api/stop":
            self._handle_stop()
        else:
            self._set_headers("text/plain", 404)
            self.wfile.write(b"Not Found")

    def _serve_file(self, file_path, content_type):
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                content = f.read()
            self._set_headers(content_type, 200)
            self.wfile.write(content)
        else:
            self._set_headers("text/plain", 404)
            self.wfile.write(f"File not found: {file_path}".encode("utf-8"))

    def _serve_json_file(self, file_path, default=None):
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = json.load(f)
                self._set_headers(status=200)
                self.wfile.write(json.dumps(content, indent=2).encode("utf-8"))
                return
            except Exception:
                pass
        
        if default is not None:
            self._set_headers(status=200)
            self.wfile.write(json.dumps(default, indent=2).encode("utf-8"))
        else:
            self._set_headers(status=404)
            self.wfile.write(json.dumps({"error": f"File not found: {file_path}"}).encode("utf-8"))

    def _save_json_file(self, file_path, data):
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            self._set_headers(status=200)
            self.wfile.write(json.dumps({"status": "success", "message": f"Saved {os.path.basename(file_path)}"}).encode("utf-8"))
        except Exception as e:
            self._set_headers(status=500)
            self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode("utf-8"))

    def _handle_run(self, data):
        global current_process, process_is_running, execution_logs, process_start_time, process_exit_code
        with process_lock:
            if process_is_running:
                self._set_headers(status=400)
                self.wfile.write(json.dumps({"status": "error", "message": "Pipeline is already running"}).encode("utf-8"))
                return

            execution_logs = []
            process_start_time = time.time()
            process_is_running = True
            process_exit_code = None

            py_exec = get_python_executable()
            cmd = [py_exec, "-u", RUNNER_SCRIPT]

            # Optional extra CLI flags
            if data.get("stage") and data.get("stage") != "all":
                cmd.extend(["--stage", data.get("stage")])
            if data.get("architecture"):
                cmd.extend(["--architecture", data.get("architecture")])
            if data.get("model"):
                cmd.extend(["--model", data.get("model")])
            if data.get("dry_run"):
                cmd.append("--dry_run")

            try:
                current_process = subprocess.Popen(
                    cmd,
                    cwd=REPO_ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
                t = threading.Thread(target=log_listener, args=(current_process,), daemon=True)
                t.start()

                self._set_headers(status=200)
                self.wfile.write(json.dumps({
                    "status": "started",
                    "pid": current_process.pid,
                    "command": " ".join(cmd)
                }).encode("utf-8"))
            except Exception as e:
                process_is_running = False
                self._set_headers(status=500)
                self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode("utf-8"))

    def _handle_status(self):
        global process_is_running, execution_logs, process_start_time, process_exit_code
        with process_lock:
            logs_str = "".join(execution_logs)
            running = process_is_running
            exit_code = process_exit_code
            elapsed = time.time() - process_start_time if process_start_time and running else 0

        # Also load latest pipeline_results.json
        results_data = {"tasks_status": [], "metrics": []}
        if os.path.exists(RESULTS_PATH):
            try:
                with open(RESULTS_PATH, "r", encoding="utf-8") as f:
                    results_data = json.load(f)
            except Exception:
                pass

        self._set_headers(status=200)
        self.wfile.write(json.dumps({
            "running": running,
            "exit_code": exit_code,
            "elapsed_seconds": round(elapsed, 1),
            "log_lines_count": len(execution_logs),
            "logs": logs_str,
            "tasks_status": results_data.get("tasks_status", []),
            "metrics": results_data.get("metrics", [])
        }).encode("utf-8"))

    def _handle_stop(self):
        global current_process, process_is_running, process_exit_code, execution_logs
        with process_lock:
            if current_process and (process_is_running or current_process.poll() is None):
                pid = current_process.pid
                try:
                    if sys.platform == "win32":
                        # On Windows, taskkill /F /T kills the entire process tree including child python workers
                        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
                    else:
                        current_process.terminate()
                except Exception as e:
                    try:
                        current_process.kill()
                    except Exception:
                        pass
                
                process_is_running = False
                process_exit_code = -1
                execution_logs.append("\n[PROCESS ABORTED] Pipeline execution stopped by user.\n")
                self._set_headers(status=200)
                self.wfile.write(json.dumps({"status": "stopped", "message": "Pipeline run aborted successfully"}).encode("utf-8"))
            else:
                process_is_running = False
                self._set_headers(status=200)
                self.wfile.write(json.dumps({"status": "idle", "message": "No active pipeline process to stop"}).encode("utf-8"))

    def log_message(self, format, *args):
        # Clean logging
        sys.stderr.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {format % args}\n")
        sys.stderr.flush()


def start_server(port=8050):
    server = ThreadedHTTPServer(("127.0.0.1", port), DashboardHandler)
    print(f"\n=======================================================")
    print(f" TABULAR META-LEARNING CONTROL DASHBOARD RUNNING")
    print(f" URL: http://127.0.0.1:{port}")
    print(f"=======================================================\n")
    try:
        while True:
            try:
                server.serve_forever()
            except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                continue
    except (KeyboardInterrupt, SystemExit):
        print("\nShutting down dashboard server...")
        try:
            server.server_close()
        except Exception:
            pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Start the Tabular Meta-Learning Visual Dashboard.")
    parser.add_argument("--port", type=int, default=8050, help="Port to listen on (default 8050)")
    args = parser.parse_args()
    start_server(port=args.port)

