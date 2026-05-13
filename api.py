"""FastAPI backend routes for the fire-scene graph memory console."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from frontend import build_app_html
from memory_graph_v2 import GraphMemory, QwenSummarizer, TurnExtractor
from memory_graph import ChatModel
from visualizer import save_graph_figure

APP_DIR = Path(__file__).parent
GRAPH_IMAGE_PATH = APP_DIR / "memory_graph.png"

APP = FastAPI(title="Fire Scene Graph Memory Console")
MEMORY: GraphMemory | None = None
CHAT_MODEL = ChatModel()


class ChatRequest(BaseModel):
    question: str
    image_paths: list[str] = Field(default_factory=list)
    visual_description: str = ""
    saliency_score: float = 0.0


class SeedRequest(BaseModel):
    questions: list[str] | None = None


def build_memory() -> GraphMemory:
    """Create a persistent fire-scene memory instance."""
    return GraphMemory(
        cold_start_threshold=3,
        initial_cluster_threshold=0.32,
        new_event_threshold=0.85,
        entity_merge_threshold=0.65,
        temporal_merge_threshold=0.75,
        summarizer=QwenSummarizer(),
        extractor=TurnExtractor(),  # Use LLM for entity/temporal/causal extraction
        storage_dir=str(APP_DIR / "qdrant_data"),
        decay_half_life_seconds=6 * 3600,
        compress_threshold=0.16,
    )


def get_memory(reset: bool = False) -> GraphMemory:
    """Lazily create the shared memory instance."""
    global MEMORY
    if MEMORY is None or reset:
        MEMORY = build_memory()
    return MEMORY


def refresh_graph_image() -> None:
    """Render a local PNG snapshot of the current coarse graph."""
    memory = get_memory()
    save_graph_figure(memory.export_graph(), output_path=str(GRAPH_IMAGE_PATH))


def seed_memory_with_questions(questions: list[str] | None = None) -> dict:
    """Insert sample fire-scene interactions."""
    memory = get_memory()
    if questions is None:
        questions = [
            "东区三楼机房出现明火，热成像显示北侧墙面温度达到78摄氏度。",
            "三楼北侧烟雾向疏散通道蔓延，2名人员可能受困在302室。",
            "指挥员要求关闭东区电源，并部署两支水枪从南侧楼梯推进。",
        ]

    inserted_ids = []
    for question in questions:
        context = memory.retrieve_context(question)
        answer = CHAT_MODEL.answer(question, context)
        turn = memory.add_interaction(question, answer)
        inserted_ids.append(turn.id)

    refresh_graph_image()
    snapshot = memory.graph_snapshot()
    return {
        "inserted": len(inserted_ids),
        "live_events": snapshot["coarse_event_count"],
        "entity_nodes": snapshot["entity_node_count"],
        "temporal_nodes": snapshot["temporal_node_count"],
        "causal_nodes": snapshot["causal_node_count"],
        "compression_ratio": snapshot["compression_ratio"],
        "turn_ids": inserted_ids,
    }


@APP.get("/", response_class=HTMLResponse)
def index() -> str:
    return build_app_html()


@APP.get("/api/graph")
def graph_snapshot() -> dict:
    memory = get_memory()
    refresh_graph_image()
    snapshot = memory.graph_snapshot()
    # Add full subgraph data for frontend rendering
    subgraphs = {}
    for event_id in memory.coarse_order:
        event = memory.coarse_events[event_id]
        subgraphs[event_id] = {
            "entity_nodes": [node.to_payload() for eid in event.entity_node_ids if (node := memory.entity_nodes.get(eid))],
            "entity_edges": [edge.to_payload() for eid, edge in memory.entity_edges.items() if edge.event_id == event_id],
            "temporal_nodes": [node.to_payload() for tid in event.temporal_node_ids if (node := memory.temporal_nodes.get(tid))],
            "temporal_edges": [edge.to_payload() for eid, edge in memory.temporal_edges.items() if edge.event_id == event_id],
            "causal_nodes": [node.to_payload() for cid in event.causal_node_ids if (node := memory.causal_nodes.get(cid))],
            "causal_edges": [edge.to_payload() for eid, edge in memory.causal_edges.items() if edge.event_id == event_id],
        }
    snapshot["subgraphs"] = subgraphs
    return snapshot


@APP.post("/api/chat")
def chat(payload: ChatRequest) -> dict:
    memory = get_memory()
    context_blocks = memory.retrieve_context(payload.question)
    answer = CHAT_MODEL.answer(payload.question, context_blocks)
    turn = memory.add_interaction(
        payload.question,
        answer,
        image_paths=payload.image_paths,
        visual_description=payload.visual_description,
        saliency_score=payload.saliency_score,
    )
    refresh_graph_image()
    return {
        "question": payload.question,
        "answer": answer,
        "retrieved_context": context_blocks,
        "retrieval": memory.get_retrieval_trace(),
        "changes": memory.get_recent_events(),
        "turn": {
            "id": turn.id,
            "status": turn.status,
            "event_id": turn.event_id,
            "local_summary": turn.local_summary,
            "modality": turn.modality,
            "visual_description": turn.visual_description,
        },
        "node": {
            "id": turn.event_id or turn.id,
            "summary": turn.local_summary,
            "source_ids": [turn.id],
        },
        "graph": memory.graph_snapshot(),
    }


@APP.post("/api/seed")
def seed(payload: SeedRequest | None = None) -> dict:
    questions = payload.questions if payload else None
    return seed_memory_with_questions(questions)


@APP.delete("/api/nodes/{node_id}")
def delete_node(node_id: str) -> dict:
    memory = get_memory()
    success = memory.delete_coarse_node(node_id)
    refresh_graph_image()
    return {"deleted": success, "node_id": node_id, "graph": memory.graph_snapshot()}


@APP.delete("/api/turns/{turn_id}")
def delete_turn(turn_id: str) -> dict:
    memory = get_memory()
    success = memory.delete_turn_by_id(turn_id)
    refresh_graph_image()
    return {"deleted": success, "turn_id": turn_id, "graph": memory.graph_snapshot()}


@APP.get("/api/debug")
def debug_state() -> dict:
    memory = get_memory()
    return {
        "graph": memory.graph_snapshot(),
        "retrieval": memory.get_retrieval_trace(),
        "events": memory.get_recent_events(),
        "graph_image": str(GRAPH_IMAGE_PATH),
    }
