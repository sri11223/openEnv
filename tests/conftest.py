"""Shared pytest fixtures for the OpenEnv test suite."""
from __future__ import annotations

import socket
import subprocess
import sys
import time

import pytest


def _port_open(host: str = "127.0.0.1", port: int = 7860, timeout: float = 0.5) -> bool:
    """Return True if a TCP connection can be made to host:port."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@pytest.fixture(scope="session")
def live_server():
    """Start the FastAPI app on localhost:7860 for the duration of the test session.

    If the port is already open (e.g. a developer has the server running manually,
    or Docker is running on CI) we skip starting a new process and just yield.
    Used by TestInferenceScript so inference.py can connect without an external
    server.
    """
    already_running = _port_open()
    proc = None
    if not already_running:
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", "7860"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # Wait up to 15 s for the server to become ready
        deadline = time.time() + 15
        while time.time() < deadline:
            if _port_open():
                break
            time.sleep(0.3)
        else:
            proc.terminate()
            pytest.fail("live_server: uvicorn did not start within 15 s")

    yield  # tests run here

    if proc is not None:
        proc.terminate()
        proc.wait(timeout=5)
