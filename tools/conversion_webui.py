"""Local web dashboard for tools/convert.py.

Run from the repository root with:
    python tools\\conversion_webui.py
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import uuid
import webbrowser
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent.parent
CONVERTER = ROOT / "tools" / "convert.py"
PAGE = Path(__file__).with_suffix(".html")
SOURCE_EXTENSIONS = {".safetensors", ".ckpt", ".pt", ".pth", ".bin"}
LORA_EXTENSIONS = {".safetensors", ".gguf"}
QUANT_TYPES = {"source", "Q8_0", "Q5_1", "Q5_0", "Q4_1", "Q4_0", "Q8_CR", "Q4_CR_W4A4"}
Q8_TYPES = {"Q8_CR", "Q8_0"}
DEVICES = {"auto", "cpu", "cuda"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT / path).resolve()


def validate_path(value: Any, label: str, extensions: set[str]) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} is required.")
    path = resolve_path(value.strip())
    if not path.is_file():
        raise ValueError(f"{label} does not exist: {path}")
    if path.suffix.lower() not in extensions:
        allowed = ", ".join(sorted(extensions))
        raise ValueError(f"{label} must use one of: {allowed}")
    return path


class ConversionManager:
    def __init__(self) -> None:
        self.jobs: dict[str, dict[str, Any]] = {}
        self.pending: queue.Queue[str] = queue.Queue()
        self.lock = threading.RLock()
        self.active_process: subprocess.Popen[str] | None = None
        self.active_job_id: str | None = None
        self.worker = threading.Thread(target=self._work, daemon=True, name="gguf-conversion-worker")
        self.worker.start()

    def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        source = validate_path(payload.get("source"), "Source checkpoint", SOURCE_EXTENSIONS)
        destination_value = payload.get("destination")
        if not isinstance(destination_value, str) or not destination_value.strip():
            raise ValueError("Destination path is required.")
        destination = resolve_path(destination_value.strip())
        if destination.suffix.lower() != ".gguf":
            raise ValueError("Destination path must end with .gguf.")
        if not destination.parent.is_dir():
            raise ValueError(f"Destination folder does not exist: {destination.parent}")
        if destination.exists():
            raise ValueError(f"Destination already exists: {destination}")

        mode = payload.get("mode", "quant")
        quant_type = payload.get("quant_type", "Q8_CR")
        if mode not in {"quant", "target"}:
            raise ValueError("Conversion mode is invalid.")
        if quant_type not in QUANT_TYPES:
            raise ValueError("Quantization type is invalid.")

        target_size = payload.get("target_size_mb")
        target_q8_type = payload.get("target_size_q8_type", "Q8_CR")
        if mode == "target":
            try:
                target_size = float(target_size)
            except (TypeError, ValueError) as error:
                raise ValueError("Target size must be a number of MiB.") from error
            if target_size <= 0:
                raise ValueError("Target size must be greater than zero.")
            if target_q8_type not in Q8_TYPES:
                raise ValueError("Target-size Q8 baseline is invalid.")

        device = payload.get("device", "auto")
        if device not in DEVICES:
            raise ValueError("Quantization device is invalid.")

        loras: list[tuple[Path, float]] = []
        for index, entry in enumerate(payload.get("loras", []), start=1):
            if not isinstance(entry, dict):
                raise ValueError(f"LoRA {index} is invalid.")
            path_value = entry.get("path", "")
            if not str(path_value).strip():
                continue
            lora_path = validate_path(path_value, f"LoRA {index}", LORA_EXTENSIONS)
            try:
                strength = float(entry.get("strength", 1))
            except (TypeError, ValueError) as error:
                raise ValueError(f"LoRA {index} strength must be a number.") from error
            loras.append((lora_path, strength))

        command = [
            sys.executable,
            str(CONVERTER),
            "--src",
            str(source),
            "--dst",
            str(destination),
            "--quantization-device",
            device,
        ]
        if mode == "target":
            command.extend(
                [
                    "--max-size-mb",
                    str(target_size),
                    "--target-size-q8-type",
                    target_q8_type,
                ]
            )
        elif quant_type != "source":
            command.extend(["--quant-type", quant_type])
        for lora_path, strength in loras:
            command.extend(["--lora", str(lora_path), "--lora-strength", str(strength)])
        if payload.get("streamed"):
            command.append("--streamed")

        job_id = uuid.uuid4().hex[:8]
        job = {
            "id": job_id,
            "created_at": now(),
            "started_at": None,
            "finished_at": None,
            "status": "queued",
            "source": str(source),
            "destination": str(destination),
            "mode": mode,
            "quant_type": quant_type if mode == "quant" else None,
            "target_size_mb": target_size if mode == "target" else None,
            "target_size_q8_type": target_q8_type if mode == "target" else None,
            "device": device,
            "streamed": bool(payload.get("streamed")),
            "loras": [{"path": str(path), "strength": strength} for path, strength in loras],
            "command": command,
            "log": [],
            "error": None,
            "return_code": None,
        }
        with self.lock:
            self.jobs[job_id] = job
        self.pending.put(job_id)
        return self.public(job)

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self.lock:
            job = self._job(job_id)
            if job["status"] == "queued":
                job["status"] = "cancelled"
                job["finished_at"] = now()
                job["log"].append("Cancelled before conversion started.")
            elif job["status"] == "running":
                job["status"] = "cancelling"
                job["log"].append("Cancellation requested.")
                if self.active_job_id == job_id and self.active_process is not None:
                    self.active_process.terminate()
            else:
                raise ValueError("Only queued or running conversions can be cancelled.")
            return self.public(job)

    def list_jobs(self) -> list[dict[str, Any]]:
        with self.lock:
            return [self.public(job) for job in reversed(list(self.jobs.values()))]

    def _job(self, job_id: str) -> dict[str, Any]:
        try:
            return self.jobs[job_id]
        except KeyError as error:
            raise ValueError("Conversion job was not found.") from error

    def public(self, job: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in job.items() if key != "command"}

    def _work(self) -> None:
        while True:
            job_id = self.pending.get()
            with self.lock:
                job = self.jobs.get(job_id)
                if job is None or job["status"] != "queued":
                    self.pending.task_done()
                    continue
                job["status"] = "running"
                job["started_at"] = now()
                job["log"].append("Starting conversion.")

            try:
                environment = os.environ.copy()
                environment["PYTHONUNBUFFERED"] = "1"
                process = subprocess.Popen(
                    job["command"],
                    cwd=ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    env=environment,
                )
                with self.lock:
                    self.active_process = process
                    self.active_job_id = job_id
                assert process.stdout is not None
                for line in process.stdout:
                    with self.lock:
                        job["log"].append(line.rstrip())
                return_code = process.wait()
                with self.lock:
                    job["return_code"] = return_code
                    if job["status"] == "cancelling":
                        job["status"] = "cancelled"
                    elif return_code == 0:
                        job["status"] = "completed"
                    else:
                        job["status"] = "failed"
                        job["error"] = f"Converter exited with code {return_code}."
            except OSError as error:
                with self.lock:
                    job["status"] = "failed"
                    job["error"] = f"Could not start converter: {error}"
                    job["log"].append(job["error"])
            finally:
                with self.lock:
                    job["finished_at"] = now()
                    self.active_process = None
                    self.active_job_id = None
                self.pending.task_done()


MANAGER = ConversionManager()


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "ComfyUI-GGUF Conversion Dashboard"

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send_page()
        elif path == "/api/jobs":
            self._send_json(HTTPStatus.OK, {"jobs": MANAGER.list_jobs()})
        else:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found."})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            payload = self._read_json()
            if path == "/api/jobs":
                self._send_json(HTTPStatus.CREATED, {"job": MANAGER.create(payload)})
            elif path.startswith("/api/jobs/") and path.endswith("/cancel"):
                job_id = path.removeprefix("/api/jobs/").removesuffix("/cancel").strip("/")
                self._send_json(HTTPStatus.OK, {"job": MANAGER.cancel(job_id)})
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found."})
        except ValueError as error:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})

    def _read_json(self) -> dict[str, Any]:
        length_header = self.headers.get("Content-Length", "0")
        try:
            length = int(length_header)
        except ValueError as error:
            raise ValueError("Invalid request length.") from error
        if length <= 0 or length > 131072:
            raise ValueError("Request body must be between 1 and 131072 bytes.")
        try:
            payload = json.loads(self.rfile.read(length))
        except json.JSONDecodeError as error:
            raise ValueError("Request body must be valid JSON.") from error
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object.")
        return payload

    def _send_page(self) -> None:
        try:
            content = PAGE.read_bytes()
        except OSError as error:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"Could not read UI: {error}"})
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        content = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format: str, *args: object) -> None:
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local ComfyUI-GGUF conversion dashboard.")
    parser.add_argument("--port", type=int, default=8189, help="Local TCP port to use (default: 8189).")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the dashboard automatically.")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535.")
    return args


def main() -> None:
    args = parse_args()
    address = ("127.0.0.1", args.port)
    httpd = ThreadingHTTPServer(address, RequestHandler)
    url = f"http://{address[0]}:{address[1]}"
    print(f"ComfyUI-GGUF conversion dashboard: {url}")
    print("Press Ctrl+C to stop the server.")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
