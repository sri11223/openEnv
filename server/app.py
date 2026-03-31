"""
Entry point for OpenEnv multi-mode deployment.

This module provides the server entry point required by the OpenEnv
framework for multi-mode deployment (Docker, uv run server, Python module).
"""
from __future__ import annotations

import os
import sys

# Ensure the repo root is on the Python path so we can import app.py
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


def main() -> None:
    """Start the OpenEnv incident-response-triage server."""
    import uvicorn  # noqa: PLC0415

    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
