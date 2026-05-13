"""Command-line interface for the fire-scene graph memory console."""

from __future__ import annotations

import json

import api
from api import CHAT_MODEL, GRAPH_IMAGE_PATH, get_memory, refresh_graph_image, seed_memory_with_questions


def run_cli(question: str) -> None:
    """Run one chat round from the command line."""
    memory = get_memory()
    context_blocks = memory.retrieve_context(question)
    answer = CHAT_MODEL.answer(question, context_blocks)
    turn = memory.add_interaction(question, answer)
    refresh_graph_image()
    print(
        json.dumps(
            {
                "question": question,
                "answer": answer,
                "stored_turn": turn.id,
                "coarse_id": turn.coarse_id,
                "modality": turn.modality,
                "retrieval": memory.get_retrieval_trace(),
                "events": memory.get_recent_events(),
                "graph": memory.graph_snapshot(),
                "graph_image": str(GRAPH_IMAGE_PATH),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def run_seed(questions: list[str] | None = None) -> None:
    """Insert questions into memory by calling LLM to generate answers."""
    try:
        print(json.dumps(seed_memory_with_questions(questions), ensure_ascii=False, indent=2))
    finally:
        if api.MEMORY is not None:
            api.MEMORY.close()
