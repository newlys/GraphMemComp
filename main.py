"""Web app and CLI entry for the fire-scene graph memory console."""

from __future__ import annotations

import argparse
import os

# Force hash-based embedding to avoid model download issues
os.environ["GRAPHMEM_FORCE_HASH_EMBEDDING"] = "1"

import uvicorn

import api
from api import APP, get_memory, refresh_graph_image
from cli import run_cli, run_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Fire-scene graph memory console")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seed", action="store_true", help="Insert sample questions with LLM-generated answers")
    parser.add_argument("--question", help="Run a single CLI chat round instead of starting the server")
    args = parser.parse_args()

    get_memory()

    if args.seed:
        run_seed()
        if not args.question:
            return

    if args.question:
        try:
            run_cli(args.question)
        finally:
            if api.MEMORY is not None:
                api.MEMORY.close()
        return

    refresh_graph_image()
    uvicorn.run(APP, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
