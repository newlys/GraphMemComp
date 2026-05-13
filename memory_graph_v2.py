"""Dual-granularity multi-graph fire-scene memory compression.

Patent-aligned implementation: coarse event nodes + three fine-grained sub-graphs
(entity-relation, temporal topology, causal relation), LLM-driven online fusion,
active-weight-based dynamic compression, and multi-layer retrieval.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
from openai import OpenAI
from qdrant_client import QdrantClient, models


# ============================================================
# Constants
# ============================================================

ENTITY_NODE_TYPES = (
    "person",
    "location",
    "equipment",
    "material",
    "building_area",
    "hazard",
)

TEMPORAL_NODE_TYPES = (
    "time_window",
    "stage",
    "sensor_state",
)

CAUSAL_NODE_TYPES = (
    "state_change",
    "trigger",
    "decision",
    "outcome",
)

VALID_TURN_STATUSES = {"buffered", "assigned"}

CRITICAL_KEYWORDS = (
    "起火", "火势", "烟雾", "爆炸", "坍塌", "受困", "伤亡", "高温",
    "易燃", "化学品", "疏散", "消防", "灭火", "救援", "出口", "楼梯",
    "电梯", "排烟", "水源",
)

ENTITY_RELATION_LABELS = (
    "位于", "包含", "依赖", "影响", "受限于", "通向", "用于", "关联",
)

CAUSAL_RELATION_LABELS = (
    "导致", "引发", "促使", "抑制", "依赖", "使得",
)

TEMPORAL_RELATION_LABELS = (
    "时序后继", "阶段过渡",
)


# ============================================================
# Utility functions
# ============================================================

def now_ts() -> float:
    return time.time()


def normalize_text(text: str) -> str:
    return " ".join((text or "").replace("\n", " ").split())


def _compute_information_density(text: str, question: str = "", answer: str = "") -> float:
    """Compute information density score for a turn.
    
    Returns a value in [0, 1] based on:
    - Entity richness (number of distinct nouns/proper nouns)
    - Keyword hits (fire-scenario critical terms)
    - Semantic novelty (unique token ratio)
    - Structural completeness (has Q&A, location, action, etc.)
    """
    if not text.strip():
        return 0.0
    
    normalized = normalize_text(text)
    tokens = tokenize(normalized)
    unique_tokens = set(tokens)
    
    # Factor 1: Entity richness - estimate by filtering likely entity tokens
    entity_indicators = sum(1 for t in unique_tokens if len(t) >= 2 and t[0].isupper() or any(kw in t for kw in ENTITY_NODE_TYPES))
    entity_score = min(1.0, entity_indicators / 5.0)  # Normalize: 5+ entities = 1.0
    
    # Factor 2: Keyword density - fire scenario critical terms
    keyword_hits = sum(1 for kw in CRITICAL_KEYWORDS if kw in normalized)
    keyword_score = min(1.0, keyword_hits / 4.0)  # Normalize: 4+ keywords = 1.0
    
    # Factor 3: Lexical diversity - unique token ratio
    lexical_diversity = len(unique_tokens) / max(len(tokens), 1)
    
    # Factor 4: Content length bonus (logarithmic scaling)
    char_count = len(normalized.replace(" ", ""))
    length_score = min(1.0, math.log1p(char_count) / math.log1p(200))  # ~200 chars = 1.0
    
    # Factor 5: Structural completeness
    has_question = bool(question.strip())
    has_answer = bool(answer.strip())
    has_location = any(loc in normalized for loc in ["楼", "层", "室", "区", "门", "号", "路", "街"])
    has_action = any(act in normalized for act in ["发现", "报告", "前往", "疏散", "灭火", "救援", "检查", "关闭", "启动"])
    structure_score = (has_question + has_answer + has_location + has_action) / 4.0
    
    # Weighted combination
    density = (
        0.25 * entity_score +
        0.30 * keyword_score +
        0.15 * lexical_diversity +
        0.15 * length_score +
        0.15 * structure_score
    )
    
    return round(min(1.0, max(0.0, density)), 4)


def strip_json_fence(text: str) -> str:
    cleaned = normalize_text(text)
    cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def tokenize(text: str) -> List[str]:
    normalized = normalize_text(text).lower()
    tokens: List[str] = []
    ascii_buffer: List[str] = []
    compact_chars: List[str] = []
    for char in normalized:
        if char.isascii() and char.isalnum():
            ascii_buffer.append(char)
            compact_chars.append(char)
            continue
        if ascii_buffer:
            token = "".join(ascii_buffer)
            if len(token) > 1:
                tokens.append(token)
            ascii_buffer = []
        if "\u4e00" <= char <= "\u9fff":
            tokens.append(char)
            compact_chars.append(char)
    if ascii_buffer:
        token = "".join(ascii_buffer)
        if len(token) > 1:
            tokens.append(token)
    for index in range(max(0, len(compact_chars) - 1)):
        tokens.append(compact_chars[index] + compact_chars[index + 1])
    return tokens


def lexical_overlap(text_a: str, text_b: str) -> float:
    tokens_a = set(tokenize(text_a))
    tokens_b = set(tokenize(text_b))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def shared_keywords(text_a: str, text_b: str, limit: int = 6) -> List[str]:
    stop_tokens = {
        "可以", "这个", "那个", "需要", "进行", "系统", "用户", "问题",
        "回答", "情况", "信息", "现场", "火灾",
    }
    overlap = sorted(set(tokenize(text_a)) & set(tokenize(text_b)))
    return [token for token in overlap if len(token) > 1 and token not in stop_tokens][:limit]


def deterministic_qdrant_id(value: str) -> str:
    return uuid.uuid5(uuid.NAMESPACE_DNS, value).hex


def fire_importance(text: str, node_type: str = "state_change", saliency_score: float = 0.0) -> float:
    """All turns have equal importance, no saliency-based weighting."""
    # Fixed base importance for all nodes
    type_boost = {
        "hazard": 0.15,
        "person": 0.12,
        "location": 0.10,
        "equipment": 0.08,
        "trigger": 0.14,
        "outcome": 0.12,
    }.get(node_type, 0.05)
    score = 0.50 + type_boost  # All nodes start with 0.50 + type boost
    return round(min(1.0, max(0.05, score)), 4)


def classify_modality(image_paths: Sequence[str], visual_description: str) -> str:
    if image_paths and visual_description:
        return "text+image"
    if image_paths:
        return "image"
    return "text"


def build_visual_description(image_paths: Sequence[str], visual_description: str = "") -> str:
    pieces = [normalize_text(visual_description)]
    for path in image_paths:
        name = Path(path).name
        if name:
            pieces.append(f"图像证据:{name}")
    return "；".join([piece for piece in pieces if piece])


# ============================================================
# Data classes
# ============================================================

@dataclass
class UpdateEvent:
    event_type: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=now_ts)


@dataclass
class RawTurn:
    id: str
    question: str
    answer: str
    text: str
    embedding: np.ndarray
    timestamp: float
    valid: bool = True
    status: str = "buffered"
    event_id: Optional[str] = None
    local_summary: str = ""
    candidate_node_ids: List[str] = field(default_factory=list)
    image_paths: List[str] = field(default_factory=list)
    visual_description: str = ""
    modality: str = "text"
    saliency_score: float = 0.0
    information_density: float = 0.0

    def to_payload(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "text": self.text,
            "embedding": self.embedding.astype(float).tolist(),
            "timestamp": self.timestamp,
            "valid": self.valid,
            "status": self.status,
            "event_id": self.event_id,
            "local_summary": self.local_summary,
            "candidate_node_ids": list(self.candidate_node_ids),
            "image_paths": list(self.image_paths),
            "visual_description": self.visual_description,
            "modality": self.modality,
            "saliency_score": self.saliency_score,
            "information_density": self.information_density,
        }

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "RawTurn":
        return cls(
            id=str(payload["id"]),
            question=str(payload.get("question", "")),
            answer=str(payload.get("answer", "")),
            text=str(payload.get("text", "")),
            embedding=np.asarray(payload.get("embedding", []), dtype=np.float32),
            timestamp=float(payload.get("timestamp", now_ts())),
            valid=bool(payload.get("valid", True)),
            status=str(payload.get("status", "buffered")),
            event_id=payload.get("event_id"),
            local_summary=str(payload.get("local_summary", "")),
            candidate_node_ids=[str(item) for item in payload.get("candidate_node_ids", [])],
            image_paths=[str(item) for item in payload.get("image_paths", [])],
            visual_description=str(payload.get("visual_description", "")),
            modality=str(payload.get("modality", "text")),
            saliency_score=float(payload.get("saliency_score", 0.0)),
            information_density=float(payload.get("information_density", 0.0)),
        )


# --- Fine-grained node types for the three sub-graphs ---

@dataclass
class EntityNode:
    """实体-关系图节点：人员、地点、设备、物资、建筑区域、危险源"""
    id: str
    event_id: str
    entity_type: str
    name: str
    summary: str
    embedding: np.ndarray
    frequency: int
    first_seen: float
    last_seen: float
    access_count: int = 0
    source_turn_ids: List[str] = field(default_factory=list)
    importance: float = 0.5
    active_weight: float = 1.0
    pinned: bool = False
    corrected: bool = False
    correction_source: str = ""
    confidence: float = 1.0

    def to_payload(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "event_id": self.event_id,
            "entity_type": self.entity_type,
            "name": self.name,
            "summary": self.summary,
            "embedding": self.embedding.astype(float).tolist(),
            "frequency": self.frequency,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "access_count": self.access_count,
            "source_turn_ids": list(self.source_turn_ids),
            "importance": self.importance,
            "active_weight": self.active_weight,
            "pinned": self.pinned,
            "corrected": self.corrected,
            "correction_source": self.correction_source,
            "confidence": self.confidence,
        }

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "EntityNode":
        text = str(payload.get("name", ""))
        return cls(
            id=str(payload["id"]),
            event_id=str(payload.get("event_id", "")),
            entity_type=str(payload.get("entity_type", "location")),
            name=text,
            summary=str(payload.get("summary", text)),
            embedding=np.asarray(payload.get("embedding", []), dtype=np.float32),
            frequency=int(payload.get("frequency", 1)),
            first_seen=float(payload.get("first_seen", now_ts())),
            last_seen=float(payload.get("last_seen", now_ts())),
            access_count=int(payload.get("access_count", 0)),
            source_turn_ids=[str(item) for item in payload.get("source_turn_ids", [])],
            importance=float(payload.get("importance", fire_importance(text, payload.get("entity_type", "location")))),
            active_weight=float(payload.get("active_weight", 1.0)),
            pinned=bool(payload.get("pinned", False)),
            corrected=bool(payload.get("corrected", False)),
            correction_source=str(payload.get("correction_source", "")),
            confidence=float(payload.get("confidence", 1.0)),
        )


@dataclass
class TemporalNode:
    """时序拓扑图节点：时间窗口、阶段、传感器状态"""
    id: str
    event_id: str
    node_type: str
    stage_name: str
    start_time: float
    end_time: float
    summary: str
    embedding: np.ndarray
    frequency: int
    first_seen: float
    last_seen: float
    access_count: int = 0
    source_turn_ids: List[str] = field(default_factory=list)
    sensor_ref: str = ""
    importance: float = 0.5
    active_weight: float = 1.0
    pinned: bool = False

    def to_payload(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "event_id": self.event_id,
            "node_type": self.node_type,
            "stage_name": self.stage_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "summary": self.summary,
            "embedding": self.embedding.astype(float).tolist(),
            "frequency": self.frequency,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "access_count": self.access_count,
            "source_turn_ids": list(self.source_turn_ids),
            "sensor_ref": self.sensor_ref,
            "importance": self.importance,
            "active_weight": self.active_weight,
            "pinned": self.pinned,
        }

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "TemporalNode":
        start_time_raw = payload.get("start_time", now_ts())
        end_time_raw = payload.get("end_time", now_ts())
        # Handle string time references (e.g., '下午', '上午')
        try:
            start_time = float(start_time_raw) if isinstance(start_time_raw, (int, float, str)) and str(start_time_raw).replace('.','').isdigit() else now_ts()
        except (ValueError, TypeError):
            start_time = now_ts()
        try:
            end_time = float(end_time_raw) if isinstance(end_time_raw, (int, float, str)) and str(end_time_raw).replace('.','').isdigit() else now_ts()
        except (ValueError, TypeError):
            end_time = now_ts()
        
        return cls(
            id=str(payload["id"]),
            event_id=str(payload.get("event_id", "")),
            node_type=str(payload.get("node_type", "time_window")),
            stage_name=str(payload.get("stage_name", "")),
            start_time=start_time,
            end_time=end_time,
            summary=str(payload.get("summary", "")),
            embedding=np.asarray(payload.get("embedding", []), dtype=np.float32),
            frequency=int(payload.get("frequency", 1)),
            first_seen=float(payload.get("first_seen", now_ts())),
            last_seen=float(payload.get("last_seen", now_ts())),
            access_count=int(payload.get("access_count", 0)),
            source_turn_ids=[str(item) for item in payload.get("source_turn_ids", [])],
            sensor_ref=str(payload.get("sensor_ref", "")),
            importance=float(payload.get("importance", 0.5)),
            active_weight=float(payload.get("active_weight", 1.0)),
            pinned=bool(payload.get("pinned", False)),
        )


@dataclass
class CausalNode:
    """因果关系图节点：状态变化、触发条件、决策、结果"""
    id: str
    event_id: str
    node_type: str
    text: str
    embedding: np.ndarray
    frequency: int
    first_seen: float
    last_seen: float
    access_count: int = 0
    source_turn_ids: List[str] = field(default_factory=list)
    importance: float = 0.5
    active_weight: float = 1.0
    pinned: bool = False
    corrected: bool = False  # Whether this node has been corrected/replaced
    correction_source: str = ""  # What caused the correction (e.g., "expert_review")
    confidence: float = 1.0  # Confidence score for this node's validity

    def to_payload(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "event_id": self.event_id,
            "node_type": self.node_type,
            "text": self.text,
            "embedding": self.embedding.astype(float).tolist(),
            "frequency": self.frequency,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "access_count": self.access_count,
            "source_turn_ids": list(self.source_turn_ids),
            "importance": self.importance,
            "active_weight": self.active_weight,
            "pinned": self.pinned,
            "corrected": self.corrected,
            "correction_source": self.correction_source,
            "confidence": self.confidence,
        }

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "CausalNode":
        text = str(payload.get("text", ""))
        return cls(
            id=str(payload["id"]),
            event_id=str(payload.get("event_id", "")),
            node_type=str(payload.get("node_type", "state_change")),
            text=text,
            embedding=np.asarray(payload.get("embedding", []), dtype=np.float32),
            frequency=int(payload.get("frequency", 1)),
            first_seen=float(payload.get("first_seen", now_ts())),
            last_seen=float(payload.get("last_seen", now_ts())),
            access_count=int(payload.get("access_count", 0)),
            source_turn_ids=[str(item) for item in payload.get("source_turn_ids", [])],
            importance=float(payload.get("importance", fire_importance(text, payload.get("node_type", "state_change")))),
            active_weight=float(payload.get("active_weight", 1.0)),
            pinned=bool(payload.get("pinned", False)),
            corrected=bool(payload.get("corrected", False)),
            correction_source=str(payload.get("correction_source", "")),
            confidence=float(payload.get("confidence", 1.0)),
        )


# --- Edge types ---

@dataclass
class EntityEdge:
    id: str
    event_id: str
    source_id: str
    target_id: str
    relation_label: str
    weight: float
    frequency: int
    last_seen: float
    evidence_turn_ids: List[str] = field(default_factory=list)

    def to_payload(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "EntityEdge":
        return cls(
            id=str(payload["id"]),
            event_id=str(payload.get("event_id", "")),
            source_id=str(payload.get("source_id", "")),
            target_id=str(payload.get("target_id", "")),
            relation_label=str(payload.get("relation_label", "关联")),
            weight=float(payload.get("weight", 1.0)),
            frequency=int(payload.get("frequency", 1)),
            last_seen=float(payload.get("last_seen", now_ts())),
            evidence_turn_ids=[str(item) for item in payload.get("evidence_turn_ids", [])],
        )


@dataclass
class TemporalEdge:
    id: str
    event_id: str
    source_id: str
    target_id: str
    relation_label: str
    weight: float
    last_seen: float

    def to_payload(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "TemporalEdge":
        return cls(
            id=str(payload["id"]),
            event_id=str(payload.get("event_id", "")),
            source_id=str(payload.get("source_id", "")),
            target_id=str(payload.get("target_id", "")),
            relation_label=str(payload.get("relation_label", "时序后继")),
            weight=float(payload.get("weight", 1.0)),
            last_seen=float(payload.get("last_seen", now_ts())),
        )


@dataclass
class CausalEdge:
    id: str
    event_id: str
    source_id: str
    target_id: str
    relation_label: str
    weight: float
    frequency: int
    last_seen: float
    evidence_turn_ids: List[str] = field(default_factory=list)

    def to_payload(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "CausalEdge":
        return cls(
            id=str(payload["id"]),
            event_id=str(payload.get("event_id", "")),
            source_id=str(payload.get("source_id", "")),
            target_id=str(payload.get("target_id", "")),
            relation_label=str(payload.get("relation_label", "导致")),
            weight=float(payload.get("weight", 1.0)),
            frequency=int(payload.get("frequency", 1)),
            last_seen=float(payload.get("last_seen", now_ts())),
            evidence_turn_ids=[str(item) for item in payload.get("evidence_turn_ids", [])],
        )


# --- Coarse event node ---

@dataclass
class CoarseEventNode:
    id: str
    title: str
    summary: str
    embedding: np.ndarray
    created_at: float
    updated_at: float
    start_time: float = 0.0
    end_time: float = 0.0
    turn_ids: List[str] = field(default_factory=list)
    entity_node_ids: List[str] = field(default_factory=list)
    temporal_node_ids: List[str] = field(default_factory=list)
    causal_node_ids: List[str] = field(default_factory=list)
    recent_hit_turn_ids: List[str] = field(default_factory=list)
    recent_consistency_scores: List[float] = field(default_factory=list)
    key_points: List[str] = field(default_factory=list)
    node_count: int = 0
    duplicate_ratio: float = 0.0
    last_hit_at: float = 0.0
    reconstruct_count: int = 0
    visual_description: str = ""
    survival_weight: float = 1.0
    compression_ratio: float = 1.0
    importance_coefficient: float = 0.5

    def to_payload(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "summary": self.summary,
            "embedding": self.embedding.astype(float).tolist(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "turn_ids": list(self.turn_ids),
            "entity_node_ids": list(self.entity_node_ids),
            "temporal_node_ids": list(self.temporal_node_ids),
            "causal_node_ids": list(self.causal_node_ids),
            "recent_hit_turn_ids": list(self.recent_hit_turn_ids),
            "recent_consistency_scores": list(self.recent_consistency_scores),
            "key_points": list(self.key_points),
            "node_count": self.node_count,
            "duplicate_ratio": self.duplicate_ratio,
            "last_hit_at": self.last_hit_at,
            "reconstruct_count": self.reconstruct_count,
            "visual_description": self.visual_description,
            "survival_weight": self.survival_weight,
            "compression_ratio": self.compression_ratio,
            "importance_coefficient": self.importance_coefficient,
        }

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "CoarseEventNode":
        return cls(
            id=str(payload["id"]),
            title=str(payload.get("title", "")),
            summary=str(payload.get("summary", "")),
            embedding=np.asarray(payload.get("embedding", []), dtype=np.float32),
            created_at=float(payload.get("created_at", now_ts())),
            updated_at=float(payload.get("updated_at", now_ts())),
            start_time=float(payload.get("start_time", 0.0)),
            end_time=float(payload.get("end_time", 0.0)),
            turn_ids=[str(item) for item in payload.get("turn_ids", [])],
            entity_node_ids=[str(item) for item in payload.get("entity_node_ids", [])],
            temporal_node_ids=[str(item) for item in payload.get("temporal_node_ids", [])],
            causal_node_ids=[str(item) for item in payload.get("causal_node_ids", [])],
            recent_hit_turn_ids=[str(item) for item in payload.get("recent_hit_turn_ids", [])],
            recent_consistency_scores=[float(item) for item in payload.get("recent_consistency_scores", [])],
            key_points=[str(item) for item in payload.get("key_points", [])],
            node_count=int(payload.get("node_count", 0)),
            duplicate_ratio=float(payload.get("duplicate_ratio", 0.0)),
            last_hit_at=float(payload.get("last_hit_at", 0.0)),
            reconstruct_count=int(payload.get("reconstruct_count", 0)),
            visual_description=str(payload.get("visual_description", "")),
            survival_weight=float(payload.get("survival_weight", 1.0)),
            compression_ratio=float(payload.get("compression_ratio", 1.0)),
            importance_coefficient=float(payload.get("importance_coefficient", 0.5)),
        )


# ============================================================
# Summarizer
# ============================================================

class BaseSummarizer:
    def summarize_turn(self, question: str, answer: str, visual_description: str = "") -> str:
        raise NotImplementedError

    def build_event_title(self, turn_texts: Sequence[str], previous_title: Optional[str] = None) -> str:
        raise NotImplementedError

    def build_event_summary(
        self,
        turn_texts: Sequence[str],
        previous_summary: Optional[str] = None,
    ) -> str:
        raise NotImplementedError

    def fuse_summary(
        self,
        previous_summary: str,
        local_summary: str,
        recent_turn_texts: Sequence[str],
    ) -> str:
        raise NotImplementedError

    def fuse_entity(self, previous_summary: str, new_info: str) -> str:
        raise NotImplementedError

    def fuse_temporal(self, previous_summary: str, new_stage: str) -> str:
        raise NotImplementedError

    def fuse_causal(self, previous_summary: str, new_causal_info: str) -> str:
        raise NotImplementedError


class HeuristicSummarizer(BaseSummarizer):
    def __init__(self, max_summary_chars: int = 240) -> None:
        self.max_summary_chars = max_summary_chars

    def summarize_turn(self, question: str, answer: str, visual_description: str = "") -> str:
        text = normalize_text(f"{question} {answer} {visual_description}")
        return self._clip(text, 120)

    def build_event_title(self, turn_texts: Sequence[str], previous_title: Optional[str] = None) -> str:
        joined = " ".join(turn_texts[:3])
        for keyword in CRITICAL_KEYWORDS:
            if keyword in joined:
                return f"{keyword}态势"
        keywords = shared_keywords(joined, joined, limit=3)
        if keywords:
            return " / ".join(keywords)[:18]
        if previous_title:
            return previous_title
        return normalize_text(joined)[:18] or "火场态势"

    def build_event_summary(
        self,
        turn_texts: Sequence[str],
        previous_summary: Optional[str] = None,
    ) -> str:
        pieces = [normalize_text(item) for item in turn_texts if normalize_text(item)]
        if previous_summary:
            pieces.insert(0, normalize_text(previous_summary))
        return self._clip("；".join(dict.fromkeys(pieces[:6])), self.max_summary_chars)

    def fuse_summary(
        self,
        previous_summary: str,
        local_summary: str,
        recent_turn_texts: Sequence[str],
    ) -> str:
        pieces: List[str] = []
        for item in [previous_summary, local_summary, *recent_turn_texts[-3:]]:
            normalized = normalize_text(item)
            if not normalized:
                continue
            if any(lexical_overlap(normalized, existing) > 0.72 for existing in pieces):
                continue
            pieces.append(normalized)
        return self._clip("；".join(pieces), self.max_summary_chars)

    def fuse_entity(self, previous_summary: str, new_info: str) -> str:
        if lexical_overlap(previous_summary, new_info) > 0.70:
            return previous_summary
        return self._clip(f"{previous_summary}；{new_info}", self.max_summary_chars)

    def fuse_temporal(self, previous_summary: str, new_stage: str) -> str:
        return self._clip(f"{previous_summary}；{new_stage}", self.max_summary_chars)

    def fuse_causal(self, previous_summary: str, new_causal_info: str) -> str:
        if lexical_overlap(previous_summary, new_causal_info) > 0.70:
            return previous_summary
        return self._clip(f"{previous_summary}；{new_causal_info}", self.max_summary_chars)

    def _clip(self, text: str, limit: int) -> str:
        normalized = normalize_text(text)
        return normalized[:limit] if len(normalized) > limit else normalized


class QwenSummarizer(BaseSummarizer):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "qwen-turbo",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        max_summary_chars: int = 260,
    ) -> None:
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.model = model
        self.max_summary_chars = max_summary_chars
        self._client = (
            OpenAI(api_key=self.api_key, base_url=base_url) if self.api_key else None
        )
        self._fallback = HeuristicSummarizer(max_summary_chars=max_summary_chars)

    def summarize_turn(self, question: str, answer: str, visual_description: str = "") -> str:
        prompt = (
            "请将以下火灾现场交互压缩为一条局部记忆摘要。"
            "保留起火点、位置、人员、设备、危险源、时间、图像证据和处置动作；"
            "控制在90字以内，不要泛化。\n"
            f"输入:{question}\n回答:{answer}\n视觉描述:{visual_description or '无'}"
        )
        return self._complete_text(prompt, max_tokens=140) or self._fallback.summarize_turn(
            question, answer, visual_description,
        )

    def build_event_title(self, turn_texts: Sequence[str], previous_title: Optional[str] = None) -> str:
        prompt = (
            "为以下火灾场景记忆生成一个粗粒度事件标题。"
            "要求8到14个汉字，突出宏观态势，不要解释。\n"
            f"已有标题:{previous_title or '无'}\n片段:\n- " + "\n- ".join(turn_texts[:5])
        )
        return (self._complete_text(prompt, max_tokens=40) or self._fallback.build_event_title(turn_texts))[:18]

    def build_event_summary(
        self,
        turn_texts: Sequence[str],
        previous_summary: Optional[str] = None,
    ) -> str:
        prompt = (
            "根据以下同一粗粒度事件下的火灾记忆，生成结构化摘要。"
            "保留灾情演化、关键实体、空间位置、图像证据和处置结果；控制在160字以内。\n"
            f"已有摘要:{previous_summary or '无'}\n片段:\n- " + "\n- ".join(turn_texts[:8])
        )
        return self._complete_text(prompt, max_tokens=220) or self._fallback.build_event_summary(
            turn_texts, previous_summary,
        )

    def fuse_summary(
        self,
        previous_summary: str,
        local_summary: str,
        recent_turn_texts: Sequence[str],
    ) -> str:
        prompt = (
            "请把已有粗粒度事件摘要与新的局部摘要融合为火灾态势记忆。"
            "删除重复和过时表述，保留高重要性事件、地点、人员、设备、危险源、视觉证据和处置动作；"
            "最终不超过160字。\n"
            f"已有摘要:{previous_summary or '无'}\n当前摘要:{local_summary}\n近期片段:\n- "
            + "\n- ".join(recent_turn_texts[-4:])
        )
        return self._complete_text(prompt, max_tokens=220) or self._fallback.fuse_summary(
            previous_summary, local_summary, recent_turn_texts,
        )

    def fuse_entity(self, previous_summary: str, new_info: str) -> str:
        prompt = (
            "请将已有实体描述与新信息融合，删除重复，保留核心语义，不超过120字。\n"
            f"已有:{previous_summary}\n新增:{new_info}"
        )
        return self._complete_text(prompt, max_tokens=140) or self._fallback.fuse_entity(previous_summary, new_info)

    def fuse_temporal(self, previous_summary: str, new_stage: str) -> str:
        prompt = (
            "请将已有阶段描述与新阶段融合，保持时间顺序，不超过120字。\n"
            f"已有:{previous_summary}\n新增:{new_stage}"
        )
        return self._complete_text(prompt, max_tokens=140) or self._fallback.fuse_temporal(previous_summary, new_stage)

    def fuse_causal(self, previous_summary: str, new_causal_info: str) -> str:
        prompt = (
            "请将已有因果描述与新因果信息融合，保留关键因果链，不超过120字。\n"
            f"已有:{previous_summary}\n新增:{new_causal_info}"
        )
        return self._complete_text(prompt, max_tokens=140) or self._fallback.fuse_causal(previous_summary, new_causal_info)

    def _complete_text(self, prompt: str, max_tokens: int) -> str:
        if self._client is None:
            return ""
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=max_tokens,
            )
            text = normalize_text(response.choices[0].message.content or "")
            return text[: self.max_summary_chars] if len(text) > self.max_summary_chars else text
        except Exception:
            return ""


# ============================================================
# Turn extractor
# ============================================================

@dataclass
class ExtractedRelation:
    source_text: str
    target_text: str
    relation_label: str


@dataclass
class InternalFact:
    text: str
    info_type: str


@dataclass
class ExtractedTurn:
    entities: Dict[str, List[str]]
    temporal_info: Dict[str, Any]
    causal_info: Dict[str, Any]
    relations: List[ExtractedRelation] = field(default_factory=list)
    internal_facts: List[InternalFact] = field(default_factory=list)


class TurnExtractor:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "qwen-turbo",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ) -> None:
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.model = model
        self._client = (
            OpenAI(api_key=self.api_key, base_url=base_url) if self.api_key else None
        )

    def extract(
        self,
        question: str,
        answer: str,
        previous_turn: Optional[RawTurn] = None,
        visual_description: str = "",
        existing_info: Optional[Dict[str, Any]] = None,
    ) -> ExtractedTurn:
        if self._client is None:
            return self._heuristic_extract(question, answer, visual_description)
        
        # Detect if this is a correction/clarification text
        is_correction = any(keyword in question.lower() + answer.lower() for keyword in 
            ["经专家评定", "经调查", "实际是", "真实原因是", "并非", "不是", "纠正", "澄清", "其实"])
        
        causal_instruction = ""
        if is_correction:
            causal_instruction = (
                "3. causal: 因果信息列表 [{causal_type, cause, effect}] - **重要：这是修正/澄清信息，只提取新文本中明确提到的因果关系！**\n"
                "   - 如果文本提到'并非XXX导致'，说明XXX是旧原因，不要提取\n"
                "   - 如果文本提到'实际是XXX'、'真实原因是XXX'，这才是新原因，必须提取\n"
                "   - 例如: 新文本说'并非扔烟蒂，实际是故意点燃'，应该提取: [{causal_type:'direct_cause', cause:'故意点燃', effect:'起火'}]\n"
                "   - 不要提取旧文本中提到的原因（如'扔烟蒂'），只提取新文本中的原因\n"
            )
        else:
            causal_instruction = (
                "3. causal: 因果信息列表 [{causal_type, cause, effect}] - 必须有明确的cause和effect\n"
                "   例如: [{causal_type:'direct_cause', cause:'汽油倒在床前', effect:'烟蒂引发起火'}]\n"
            )
        
        prompt = (
            "从火灾现场交互中抽取四类细粒度信息，只返回严格JSON。\n"
            "1. entities: 关键实体 {entity_type: [实体文本]}，类型: person, location, equipment, material, building_area, hazard\n"
            "2. temporal: 时序信息列表 [{stage_name, start_time_ref, summary}] - 如果有多个时间点，返回多个对象\n"
            "   例如: [{stage_name:'起火', start_time_ref:'上午', summary:'房屋发生火灾'}, {stage_name:'伤亡', start_time_ref:'下午', summary:'女主人和小孩死亡'}]\n"
            + causal_instruction +
            "4. relations: 实体关系 [{source_text, target_text, relation_label}]，从对话中自由提取明确的关系标签\n"
            "   - 根据原文语境提取具体关系，例如:'扔在床前'、'导致起火'、'正在扑救'、'位于三楼'等\n"
            "   - 关系标签应该是原文中体现的动作、位置、状态或因果，不要泛化为'关联'\n"
            "   - 只提取有意义的、对理解决策有帮助的关系\n"
            "请完整提取对话中的所有相关信息，后续会有审查者负责去重和冲突检测。\n"
            '输出格式: {"entities":{...}, "temporal":[...], "causal":[...], "relations":[...]}\n'
            f"输入:{question}\n回答:{answer}\n视觉描述:{visual_description or '无'}"
        )
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=900,
            )
            raw_content = response.choices[0].message.content or ""
            print(f"\n========== LLM提取结果 ==========")
            print(raw_content)
            print(f"================================\n")
            payload = json.loads(strip_json_fence(raw_content))
            return self._payload_to_extracted(payload)
        except Exception as e:
            print(f"\n========== LLM提取失败，回退到启发式 ==========")
            print(f"Error: {e}")
            print(f"================================\n")
            return self._heuristic_extract(question, answer, visual_description)

    def _payload_to_extracted(self, payload: Dict[str, Any]) -> ExtractedTurn:
        entities: Dict[str, List[str]] = {}
        for entity_type in ENTITY_NODE_TYPES:
            items = payload.get("entities", {}).get(entity_type, [])
            entities[entity_type] = [normalize_text(item) for item in items if normalize_text(item)][:8]

        temporal_info = payload.get("temporal", {})
        causal_info = payload.get("causal", {})

        relations = [
            ExtractedRelation(
                source_text=normalize_text(item.get("source_text", "")),
                target_text=normalize_text(item.get("target_text", "")),
                relation_label=normalize_text(item.get("relation_label", "关联")),
            )
            for item in payload.get("relations", [])
            if normalize_text(item.get("source_text", ""))
            and normalize_text(item.get("target_text", ""))
        ]

        facts = [
            InternalFact(text=item, info_type=entity_type)
            for entity_type, items in entities.items()
            for item in items
        ]

        return ExtractedTurn(
            entities=entities,
            temporal_info=temporal_info,
            causal_info=causal_info,
            relations=relations,
            internal_facts=facts,
        )

    def _heuristic_extract(self, question: str, answer: str = "", visual_description: str = "") -> ExtractedTurn:
        merged = normalize_text(f"{question} {answer} {visual_description}")
        chunks = [
            normalize_text(part)
            for part in re.split(r"[。；;，,\n]", merged)
            if normalize_text(part)
        ]
        entities: Dict[str, List[str]] = {t: [] for t in ENTITY_NODE_TYPES}
        for chunk in chunks[:18]:
            entity_type = self._classify_entity(chunk)
            entities[entity_type].append(chunk[:48])

        temporal_info = {
            "stage_name": self._infer_stage(chunks),
            "start_time_ref": "",
            "summary": chunks[0] if chunks else "",
            "sensor_ref": "",
        }
        causal_info = self._extract_causal_info(chunks)

        relations = self._infer_relations([
            InternalFact(text=item, info_type=et)
            for et, items in entities.items() for item in items
        ])

        facts = [
            InternalFact(text=item, info_type=et)
            for et, items in entities.items() for item in items
        ]

        return ExtractedTurn(
            entities=entities,
            temporal_info=temporal_info,
            causal_info=causal_info,
            relations=relations,
            internal_facts=facts,
        )

    def _classify_entity(self, chunk: str) -> str:
        if any(key in chunk for key in ("受困", "人员", "消防员", "指挥员", "队员", "伤员")):
            return "person"
        if any(key in chunk for key in ("东", "西", "南", "北", "楼", "层", "区", "室", "出口", "通道", "仓库", "坐标")):
            return "location"
        if any(key in chunk for key in ("灭火器", "水枪", "喷淋", "排烟", "防火门", "摄像头", "热成像", "泵")):
            return "equipment"
        if any(key in chunk for key in ("易燃", "化学品", "燃气", "爆炸", "高温", "有毒", "坍塌")):
            return "hazard"
        if any(key in chunk for key in ("建筑", "机房", "厂房", "仓库", "走廊", "楼梯间")):
            return "building_area"
        return "material"

    def _extract_causal_info(self, chunks: Sequence[str]) -> Dict[str, str]:
        text = " ".join(chunks[:6])
        # 显式因果关系
        for pattern in [
            ("因.*导致", "direct_cause"),
            ("由于.*使", "direct_cause"),
            ("因为.*所以", "direct_cause"),
            (".*导致.*", "state_change"),
            ("由于.*", "condition_cause"),
            (".*原因是.*", "explanation"),
        ]:
            import re as _re
            if _re.search(pattern[0], text):
                # 提取原因和结果
                if "因" in text and "导致" in text:
                    parts = text.split("导致", 1)
                    return {
                        "causal_type": pattern[1],
                        "text": text,
                        "cause": parts[0].replace("因", "").strip()[:60],
                        "effect": parts[1].strip()[:60],
                    }
                elif "由于" in text and ("使" in text or "成功" in text):
                    parts = text.split("，", 1)
                    return {
                        "causal_type": pattern[1],
                        "text": text,
                        "cause": parts[0].replace("由于", "").strip()[:60],
                        "effect": parts[1].strip()[:60] if len(parts) > 1 else "",
                    }
        # 隐式状态变化
        if any(k in text for k in ("温度", "降至", "升至", "恢复", "改善", "完成")):
            return {
                "causal_type": "state_change",
                "text": text,
                "cause": "",
                "effect": chunks[-1] if chunks else "",
            }
        return {
            "causal_type": "implicit",
            "text": chunks[-1] if chunks else "",
            "cause": "",
            "effect": "",
        }

    def _infer_stage(self, chunks: Sequence[str]) -> str:
        text = " ".join(chunks[:5])
        # 按时间词判断阶段
        if any(k in text for k in ("上午", "下午", "中午", "凌晨", "清晨", "傍晚", "夜间", "深夜")):
            if any(k in text for k in ("起火", "火灾", "燃烧", "着火")):
                return "起火发现"
        # 初期预警阶段
        if any(k in text for k in ("报警器", "触发", "烟感", "热成像", "首次")):
            return "初期预警"
        # 火势蔓延阶段
        if any(k in text for k in ("蔓延", "扩散", "升温", "温度升", "浓烟")):
            return "火势蔓延"
        # 指挥部署阶段
        if any(k in text for k in ("指挥员", "到达", "部署", "关闭电源", "命令")):
            return "指挥部署"
        # 救援行动阶段
        if any(k in text for k in ("受困", "救援", "救出", "应急广播", "掩护")):
            return "救援行动"
        # 灭火行动阶段
        if any(k in text for k in ("水源", "消防栓", "排烟", "水枪", "灭火")):
            return "灭火行动"
        # 善后处理阶段
        if any(k in text for k in ("扑灭", "清理", "清点", "降至", "控制", "死亡", "伤亡")):
            return "善后处理"
        # 风险监测阶段
        if any(k in text for k in ("复燃", "监控", "待命", "泄漏", "风险")):
            return "风险监测"
        # 人员疏散阶段
        if any(k in text for k in ("疏散", "撤离", "逃生")):
            return "人员疏散"
        # 起火发现阶段
        if any(k in text for k in ("起火", "火灾", "燃烧", "着火", "烟蒂", "点燃")):
            return "起火发现"
        return "现场处置"

    def _infer_relations(self, facts: Sequence[InternalFact]) -> List[ExtractedRelation]:
        relations: List[ExtractedRelation] = []
        locations = [fact for fact in facts if fact.info_type == "location"]
        for fact in facts:
            if fact.info_type != "location" and locations:
                relations.append(
                    ExtractedRelation(
                        source_text=fact.text,
                        target_text=locations[0].text,
                        relation_label="位于",
                    )
                )
        return relations[:10]


# ============================================================
# 审查者：去重 + 冲突消解
# ============================================================

class CausalReviewer:
    """Reviews newly extracted causal/temporal/entity nodes against existing nodes.
    Removes duplicates and resolves conflicts.
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "qwen-turbo",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ) -> None:
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.base_url = base_url
        self._client: Optional[Any] = None
        if self.api_key:
            try:
                from openai import OpenAI as _OpenAI
                self._client = _OpenAI(api_key=self.api_key, base_url=self.base_url)
            except Exception:
                self._client = None

    def review_causal(
        self,
        new_causal_items: List[Dict[str, Any]],
        existing_causal_items: List[Dict[str, Any]],
        new_text: str = "",
    ) -> Tuple[List[Dict[str, Any]], List[UpdateEvent], bool]:
        """Review new causal items against existing ones.
        Returns (new_causal_chain, corrections, should_replace_all).
        If should_replace_all is True, new_causal_chain is the complete replacement.
        """
        if not self._client or not new_causal_items:
            return new_causal_items, [], False
        
        # Even if no existing items, still return for logging
        if not existing_causal_items:
            print(f"\n========== 审查者：无已有因果节点，全部保留 ==========")
            return new_causal_items, [], False
        
        prompt = (
            "你是火灾调查信息审查专家。请对比【新文本】和【已有的因果信息】，\n"
            "判断新文本是否是对已有因果的**修正/澄清**。\n\n"
            "如果是修正（如'经专家评定，实际是XXX'、'并非XXX，而是XXX'）：\n"
            "- 返回 is_correction: true\n"
            "- 根据新文本，输出**完整的、正确的新因果链**（不要包含旧信息）\n"
            "- **重要**：每个因果项必须有明确的 cause 和 effect，且它们应该是不同的内容！\n"
            "- **格式**：new_causal_chain: 因果列表 [{causal_type, cause, effect}]，其中 cause 是原因，effect 是结果\n"
            "- **示例**：如果新文本说'王某故意点燃汽油导致火灾'，应该输出：\n"
            '  [{"causal_type": "direct_cause", "cause": "王某故意用打火机点燃地上的油", "effect": "引发火灾"}]\n'
            "- 不要输出 cause 和 effect 相同的项！\n\n"
            "如果只是补充新信息（不是修正）：\n"
            "- 返回 is_correction: false\n"
            "- new_causal_chain: 合并新旧后的完整因果链\n\n"
            "【已有因果信息】\n"
        )
        for i, item in enumerate(existing_causal_items[:10]):
            prompt += f"{i+1}. [{item.get('causal_type','')}] 原因:{item.get('cause','')} → 结果:{item.get('effect','')}\n"
        
        prompt += (
            "\n【新文本】\n" + new_text + "\n\n"
            "请返回JSON，格式：\n"
            '{\n'
            '  "is_correction": true/false,\n'
            '  "reason": "简要说明为什么是/不是修正",\n'
            '  "new_causal_chain": [\n'
            '    {"causal_type": "direct_cause", "cause": "xxx", "effect": "yyy"},\n'
            '    ...\n'
            '  ]\n'
            '}\n'
            "只返回JSON，不要其他内容。"
        )
        
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=800,
            )
            raw = response.choices[0].message.content or "{}"
            print(f"\n========== 审查者因果审查结果 ==========")
            print(raw)
            print(f"========================================\n")
            
            import json
            result = json.loads(raw) if raw.strip().startswith("{") else {}
            
            is_correction = result.get("is_correction", False)
            new_chain = result.get("new_causal_chain", [])
            
            corrections = []
            if is_correction:
                corrections.append(
                    UpdateEvent(
                        event_type="causal_full_replacement",
                        message=f"审查者全量替换因果链：{result.get('reason', '')}",
                        details={
                            "old_count": len(existing_causal_items),
                            "new_count": len(new_chain),
                        },
                    )
                )
            
            return new_chain if new_chain else new_causal_items, corrections, is_correction
        
        except Exception as e:
            print(f"审查者因果审查失败: {e}")
            return new_causal_items, [], False

    def review_temporal(
        self,
        new_temporal_items: List[Dict[str, Any]],
        existing_temporal_items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Review temporal items, remove duplicates."""
        if not new_temporal_items or not existing_temporal_items:
            return new_temporal_items
        
        filtered = []
        existing_stages = {item.get("stage_name", "").lower() for item in existing_temporal_items}
        
        for item in new_temporal_items:
            stage = item.get("stage_name", "").lower()
            if stage not in existing_stages:
                filtered.append(item)
        
        return filtered

    def review_entities(
        self,
        new_entities: Dict[str, List[str]],
        existing_entities: Dict[str, List[str]],
    ) -> Dict[str, List[str]]:
        """Review entities, remove duplicates."""
        filtered: Dict[str, List[str]] = {}
        for etype, names in new_entities.items():
            existing = {n.lower() for n in existing_entities.get(etype, [])}
            filtered[etype] = [n for n in names if n.lower() not in existing]
        return filtered


# ============================================================
# Embedding model
# ============================================================

class EmbeddingModel:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None
        self._dim = 384
        if os.getenv("GRAPHMEM_FORCE_HASH_EMBEDDING") == "1":
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
            probe = np.asarray(self._model.encode("dimension probe"), dtype=np.float32)
            self._dim = int(probe.shape[0])
        except Exception:
            self._model = None

    @property
    def dimension(self) -> int:
        return self._dim

    def encode(self, text: str) -> np.ndarray:
        if self._model is not None:
            vector = np.asarray(self._model.encode(text), dtype=np.float32)
        else:
            vector = self._hash_embed(text)
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def encode_multimodal(
        self,
        text: str,
        visual_description: str = "",
        saliency_score: float = 0.0,
    ) -> np.ndarray:
        text_vector = self.encode(text)
        if not visual_description:
            return text_vector
        visual_vector = self.encode(visual_description)
        visual_weight = 0.28 + min(0.32, saliency_score * 0.32)
        fused = (1.0 - visual_weight) * text_vector + visual_weight * visual_vector
        norm = np.linalg.norm(fused)
        return fused if norm == 0 else fused / norm

    def _hash_embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self._dim, dtype=np.float32)
        features = tokenize(text)
        for feature in features:
            index = hash(feature) % self._dim
            sign = -1.0 if hash((feature, "sign")) % 2 else 1.0
            vector[index] += sign
        if features:
            vector /= math.sqrt(len(features))
        return vector


# ============================================================
# Two-layer memory store (Qdrant-based)
# ============================================================

class TwoLayerMemoryStore:
    def __init__(self, embedding_dim: int, storage_dir: str = "qdrant_data") -> None:
        self.embedding_dim = embedding_dim
        self.storage_dir = Path(storage_dir)
        if storage_dir != ":memory:":
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            self.state_path = self.storage_dir / "graph_state_v4.json"
            self.client = QdrantClient(path=str(self.storage_dir))
        else:
            self.state_path = None
            self.client = QdrantClient(":memory:")
        self.raw_turn_collection = "raw_turns_v4"
        self.coarse_collection = "coarse_events_v4"
        self.entity_collection = "entity_nodes_v4"
        self.temporal_collection = "temporal_nodes_v4"
        self.causal_collection = "causal_nodes_v4"
        self._ensure_collection(self.raw_turn_collection)
        self._ensure_collection(self.coarse_collection)
        self._ensure_collection(self.entity_collection)
        self._ensure_collection(self.temporal_collection)
        self._ensure_collection(self.causal_collection)
        self._memory_state: Optional[Dict[str, Any]] = None

    def _ensure_collection(self, collection_name: str) -> None:
        collections = {item.name for item in self.client.get_collections().collections}
        if collection_name in collections:
            try:
                info = self.client.get_collection(collection_name)
                vectors = info.config.params.vectors
                current_size = int(vectors.size if hasattr(vectors, "size") else list(vectors.values())[0].size)
                if current_size != self.embedding_dim:
                    self.client.delete_collection(collection_name)
            except Exception:
                self.client.delete_collection(collection_name)
        collections = {item.name for item in self.client.get_collections().collections}
        if collection_name not in collections:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=self.embedding_dim, distance=models.Distance.COSINE),
            )

    def load_state(self, cold_start_threshold: int) -> Dict[str, Any]:
        base = {
            "version": 4,
            "graph_initialized": False,
            "cold_start_threshold": cold_start_threshold,
            "raw_turn_order": [],
            "coarse_order": [],
            "entity_edges": [],
            "temporal_edges": [],
            "causal_edges": [],
        }
        if self.state_path is None:
            return dict(self._memory_state or base)
        if not self.state_path.exists():
            return base
        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        for key, value in base.items():
            payload.setdefault(key, value)
        payload["version"] = 4
        return payload

    def save_state(self, payload: Dict[str, Any]) -> None:
        if self.state_path is None:
            self._memory_state = dict(payload)
            return
        self.state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_raw_turns(self) -> List[RawTurn]:
        return [RawTurn.from_payload(item) for item in self._load_records(self.raw_turn_collection)]

    def load_coarse_events(self) -> List[CoarseEventNode]:
        return [CoarseEventNode.from_payload(item) for item in self._load_records(self.coarse_collection)]

    def load_entity_nodes(self) -> List[EntityNode]:
        return [EntityNode.from_payload(item) for item in self._load_records(self.entity_collection)]

    def load_temporal_nodes(self) -> List[TemporalNode]:
        return [TemporalNode.from_payload(item) for item in self._load_records(self.temporal_collection)]

    def load_causal_nodes(self) -> List[CausalNode]:
        return [CausalNode.from_payload(item) for item in self._load_records(self.causal_collection)]

    def _load_records(self, collection_name: str) -> List[Dict[str, Any]]:
        records, _ = self.client.scroll(
            collection_name=collection_name,
            limit=20_000,
            with_payload=True,
            with_vectors=False,
        )
        return [dict(record.payload or {}) for record in records]

    def upsert_raw_turn(self, turn: RawTurn) -> None:
        self._upsert_vector_object(self.raw_turn_collection, turn.id, turn.embedding, turn.to_payload())

    def upsert_coarse_event(self, event: CoarseEventNode) -> None:
        self._upsert_vector_object(self.coarse_collection, event.id, event.embedding, event.to_payload())

    def upsert_entity_node(self, node: EntityNode) -> None:
        self._upsert_vector_object(self.entity_collection, node.id, node.embedding, node.to_payload())

    def upsert_temporal_node(self, node: TemporalNode) -> None:
        self._upsert_vector_object(self.temporal_collection, node.id, node.embedding, node.to_payload())

    def upsert_causal_node(self, node: CausalNode) -> None:
        self._upsert_vector_object(self.causal_collection, node.id, node.embedding, node.to_payload())

    def delete_coarse_event(self, event_id: str) -> None:
        self._delete_vector_object(self.coarse_collection, event_id)

    def delete_raw_turn(self, turn_id: str) -> None:
        self._delete_vector_object(self.raw_turn_collection, turn_id)

    def delete_entity_node(self, node_id: str) -> None:
        self._delete_vector_object(self.entity_collection, node_id)

    def delete_temporal_node(self, node_id: str) -> None:
        self._delete_vector_object(self.temporal_collection, node_id)

    def delete_causal_node(self, node_id: str) -> None:
        self._delete_vector_object(self.causal_collection, node_id)

    def _upsert_vector_object(
        self,
        collection_name: str,
        object_id: str,
        vector: np.ndarray,
        payload: Dict[str, Any],
    ) -> None:
        self.client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=deterministic_qdrant_id(object_id),
                    vector=vector.astype(float).tolist(),
                    payload=payload,
                )
            ],
        )

    def _delete_vector_object(self, collection_name: str, object_id: str) -> None:
        self.client.delete(
            collection_name=collection_name,
            points_selector=models.PointIdsList(points=[deterministic_qdrant_id(object_id)]),
        )

    def close(self) -> None:
        self.client.close()


# ============================================================
# Graph Memory (Patent-aligned)
# ============================================================

class GraphMemory:
    """Patent-aligned dual-granularity multi-graph fire-scene memory.

    Coarse layer: event nodes (CoarseEventNode)
    Fine layer: three independent sub-graphs per event
      - Entity-Relation graph (EntityNode + EntityEdge)
      - Temporal Topology graph (TemporalNode + TemporalEdge)
      - Causal Relation graph (CausalNode + CausalEdge)
    """

    def __init__(
        self,
        cold_start_threshold: int = 6,
        initial_cluster_threshold: float = 0.24,
        new_event_threshold: float = 0.34,
        entity_merge_threshold: float = 0.78,
        temporal_merge_threshold: float = 0.75,
        causal_prune_threshold: float = 0.16,
        information_density_threshold: float = 0.65,
        summarizer: Optional[BaseSummarizer] = None,
        extractor: Optional[TurnExtractor] = None,
        embedding_model: Optional[EmbeddingModel] = None,
        storage_dir: str = "qdrant_data",
        decay_half_life_seconds: float = 6 * 3600,
        compress_threshold: float = 0.16,
        alpha: float = 0.35,
        beta: float = 0.25,
        gamma: float = 0.25,
        delta: float = 0.15,
    ) -> None:
        self.cold_start_threshold = cold_start_threshold
        self.initial_cluster_threshold = initial_cluster_threshold
        self.new_event_threshold = new_event_threshold
        self.entity_merge_threshold = entity_merge_threshold
        self.temporal_merge_threshold = temporal_merge_threshold
        self.causal_prune_threshold = causal_prune_threshold
        self.information_density_threshold = information_density_threshold
        self.decay_half_life_seconds = decay_half_life_seconds
        self.compress_threshold = compress_threshold

        # Active weight formula parameters: W = alpha*time + beta*access + gamma*importance + delta*semantic_change
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        
        # Track cumulative information density for cold-start
        self._cumulative_density = 0.0
        self._density_turn_count = 0

        self.summarizer = summarizer or QwenSummarizer()
        self.extractor = extractor or TurnExtractor()
        self.reviewer = CausalReviewer()
        self.embedding_model = embedding_model or EmbeddingModel()
        self.store = TwoLayerMemoryStore(
            embedding_dim=self.embedding_model.dimension,
            storage_dir=storage_dir,
        )

        self.state = self.store.load_state(cold_start_threshold=self.cold_start_threshold)
        self.raw_turns: Dict[str, RawTurn] = {turn.id: turn for turn in self.store.load_raw_turns()}
        self.coarse_events: Dict[str, CoarseEventNode] = {
            event.id: event for event in self.store.load_coarse_events()
        }
        self.entity_nodes: Dict[str, EntityNode] = {node.id: node for node in self.store.load_entity_nodes()}
        self.temporal_nodes: Dict[str, TemporalNode] = {node.id: node for node in self.store.load_temporal_nodes()}
        self.causal_nodes: Dict[str, CausalNode] = {node.id: node for node in self.store.load_causal_nodes()}

        self.entity_edges: Dict[str, EntityEdge] = {
            edge["id"]: EntityEdge.from_payload(edge) for edge in self.state.get("entity_edges", [])
        }
        self.temporal_edges: Dict[str, TemporalEdge] = {
            edge["id"]: TemporalEdge.from_payload(edge) for edge in self.state.get("temporal_edges", [])
        }
        self.causal_edges: Dict[str, CausalEdge] = {
            edge["id"]: CausalEdge.from_payload(edge) for edge in self.state.get("causal_edges", [])
        }

        self.raw_turn_order: List[str] = [
            turn_id for turn_id in self.state.get("raw_turn_order", []) if turn_id in self.raw_turns
        ]
        self.coarse_order: List[str] = [
            event_id for event_id in self.state.get("coarse_order", []) if event_id in self.coarse_events
        ]
        self.graph_initialized = bool(self.state.get("graph_initialized", False))
        self.recent_events: List[UpdateEvent] = []
        self._retrieval_trace: Dict[str, Any] = {"steps": []}

        self._repair_order_lists()
        self._repair_event_memberships()
        self._refresh_all_active_weights()

    # --- Public API ---

    def add_interaction(
        self,
        question: str,
        answer: str,
        valid: bool = True,
        image_paths: Optional[Sequence[str]] = None,
        visual_description: str = "",
        saliency_score: float = 0.0,
    ) -> RawTurn:
        self.recent_events = []
        image_list = [str(item) for item in image_paths or [] if str(item).strip()]
        visual = build_visual_description(image_list, visual_description)
        modality = classify_modality(image_list, visual)
        fused_text = normalize_text(f"Q: {question}\nA: {answer}\nV: {visual}")
        turn = RawTurn(
            id=self._new_id("turn"),
            question=question,
            answer=answer,
            text=fused_text,
            embedding=self.embedding_model.encode_multimodal(
                f"{question}\n{answer}",
                visual,
                saliency_score,
            ),
            timestamp=now_ts(),
            valid=valid,
            status="buffered",
            local_summary=self.summarizer.summarize_turn(question, answer, visual),
            image_paths=image_list,
            visual_description=visual,
            modality=modality,
            saliency_score=saliency_score,
        )
        self.raw_turns[turn.id] = turn
        self.raw_turn_order.append(turn.id)
        self.store.upsert_raw_turn(turn)
        self.recent_events.append(
            UpdateEvent(
                event_type="turn_buffered",
                message="新的火场交互已进入动态缓冲区。",
                details={"turn_id": turn.id, "valid": valid, "modality": modality},
            )
        )

        if not valid:
            self._persist_state()
            return turn

        if not self.graph_initialized:
            # Compute information density for this turn
            turn_density = _compute_information_density(fused_text, question, answer)
            turn.information_density = turn_density
            self._cumulative_density += turn_density
            self._density_turn_count += 1
            avg_density = self._cumulative_density / self._density_turn_count
            
            valid_count = self._count_valid_turns()
            
            # Dynamic cold-start trigger: 
            # Option 1: High-density single turn can trigger immediately
            # Option 2: Accumulated average density meets threshold
            # Option 3: Fallback to original turn count threshold
            should_trigger = (
                turn_density >= self.information_density_threshold or  # High-density immediate trigger
                (avg_density >= self.information_density_threshold * 0.7 and valid_count >= 2) or  # Medium density + min turns
                valid_count >= self.cold_start_threshold  # Original fallback
            )
            
            trigger_reason = "information_density_high" if turn_density >= self.information_density_threshold else \
                           "average_density_sufficient" if (avg_density >= self.information_density_threshold * 0.7 and valid_count >= 2) else \
                           "turn_count_threshold"
            
            self.recent_events.append(
                UpdateEvent(
                    event_type="cold_start_progress",
                    message="冷启动缓冲区已更新，等待达到事件级图谱初始化阈值。",
                    details={
                        "valid_turn_count": valid_count,
                        "cold_start_threshold": self.cold_start_threshold,
                        "current_turn_density": round(turn_density, 4),
                        "average_density": round(avg_density, 4),
                        "information_density_threshold": self.information_density_threshold,
                        "should_trigger": should_trigger,
                        "trigger_reason": trigger_reason if should_trigger else None,
                    },
                )
            )
            
            if should_trigger:
                self._initialize_from_buffer()
            self._persist_state()
            return turn

        self._incremental_update(turn.id)
        self._apply_compression()
        self._persist_state()
        return self.raw_turns[turn.id]

    def retrieve_context(self, query: str, k: Optional[int] = None) -> List[Any]:
        """Multi-layer retrieval: fine-grained entry -> coarse event context -> multi-dimensional aggregation."""
        self._refresh_all_active_weights()
        self._retrieval_trace = {
            "steps": [],
            "coarse_events": [],
            "entity_results": [],
            "temporal_results": [],
            "causal_results": [],
        }
        k = k or 6
        query_vector = self.embedding_model.encode(query)
        query_tokens = set(tokenize(query))

        # Step 1: Fine-grained entry point — vector match across all three sub-graphs
        fine_candidates = self._fine_grained_entry(query_vector, query, query_tokens)

        if not fine_candidates:
            self._retrieval_trace["steps"].append(
                {"step": "no_fine_entry", "message": "未找到匹配的细粒度候选节点。"}
            )
            return []

        # Step 2: Upward locate coarse event node
        event_ids = list(dict.fromkeys([c["event_id"] for c in fine_candidates]))
        event_scores = []
        for event_id in event_ids:
            if event_id not in self.coarse_events:
                continue
            event = self.coarse_events[event_id]
            semantic = cosine_similarity(query_vector, event.embedding)
            lexical = lexical_overlap(query, f"{event.title} {event.summary}")
            keyword = len(query_tokens & set(tokenize(f"{event.title} {event.summary}"))) / max(len(query_tokens), 1)
            combined = 0.48 * max(semantic, 0.0) + 0.20 * lexical + 0.17 * keyword + 0.15 * event.survival_weight
            event_scores.append((event_id, combined))

        event_scores.sort(key=lambda x: x[1], reverse=True)
        if not event_scores:
            return []

        anchor_event_id, anchor_score = event_scores[0]
        anchor_event = self.coarse_events[anchor_event_id]
        anchor_event.last_hit_at = now_ts()
        anchor_event.recent_hit_turn_ids = anchor_event.recent_hit_turn_ids[-7:] + [f"query_{int(now_ts())}"]
        anchor_event.survival_weight = min(1.0, anchor_event.survival_weight + 0.05)
        self.store.upsert_coarse_event(anchor_event)

        self._retrieval_trace["anchor"] = {
            "event_id": anchor_event_id,
            "title": anchor_event.title,
            "summary": anchor_event.summary,
            "score": round(anchor_score, 4),
        }

        # Step 3: Local retrieval within the anchor event — aggregate from three sub-graphs
        entity_results = self._retrieve_entity_graph(anchor_event_id, query_vector, query)
        temporal_results = self._retrieve_temporal_graph(anchor_event_id, query_vector, query)
        causal_results = self._retrieve_causal_graph(anchor_event_id, query_vector, query)

        self._retrieval_trace["entity_results"] = entity_results[:4]
        self._retrieval_trace["temporal_results"] = temporal_results[:4]
        self._retrieval_trace["causal_results"] = causal_results[:4]

        # Combine results into context blocks
        context_blocks = []
        for item in entity_results[:3]:
            context_blocks.append(f"[实体] {item['name']} ({item['entity_type']}): {item['summary']}")
        # Return more temporal nodes to show complete timeline
        for item in temporal_results[:6]:
            context_blocks.append(f"[时序] {item['stage_name']}: {item['summary']}")
        # Return more causal nodes to show complete causal chain
        for item in causal_results[:5]:
            context_blocks.append(f"[因果] ({item['node_type']}): {item['text']}")

        # Add event-level summary
        context_blocks.insert(0, f"[事件] {anchor_event.title}: {anchor_event.summary}")

        self._retrieval_trace["steps"].append(
            {
                "step": "multi_layer_retrieval",
                "event_id": anchor_event_id,
                "entity_count": len(entity_results),
                "temporal_count": len(temporal_results),
                "causal_count": len(causal_results),
            }
        )

        return context_blocks[:k]

    def get_retrieval_trace(self) -> Dict[str, Any]:
        return self._retrieval_trace

    def get_recent_events(self) -> List[Dict[str, Any]]:
        return [asdict(event) for event in self.recent_events]

    def graph_snapshot(self) -> Dict[str, Any]:
        self._refresh_all_active_weights()
        coarse_edges = self._compute_coarse_edges()
        coarse_nodes = []
        total_raw_chars = sum(len(self.raw_turns[turn_id].text) for turn_id in self.raw_turn_order if turn_id in self.raw_turns)
        total_memory_chars = 0

        for event_id in self.coarse_order:
            event = self.coarse_events[event_id]
            total_memory_chars += len(event.summary)
            total_memory_chars += sum(
                len(self.entity_nodes[eid].summary) for eid in event.entity_node_ids if eid in self.entity_nodes
            )
            total_memory_chars += sum(
                len(self.temporal_nodes[tid].summary) for tid in event.temporal_node_ids if tid in self.temporal_nodes
            )
            total_memory_chars += sum(
                len(self.causal_nodes[cid].text) for cid in event.causal_node_ids if cid in self.causal_nodes
            )
            coarse_nodes.append(
                {
                    "id": event.id,
                    "title": event.title,
                    "summary": event.summary,
                    "created_at": event.created_at,
                    "updated_at": event.updated_at,
                    "turn_count": len(event.turn_ids),
                    "entity_count": len(event.entity_node_ids),
                    "temporal_count": len(event.temporal_node_ids),
                    "causal_count": len(event.causal_node_ids),
                    "survival_weight": event.survival_weight,
                    "compression_ratio": event.compression_ratio,
                }
            )

        buffered_turns = [self._turn_snapshot(self.raw_turns[turn_id]) for turn_id in self.raw_turn_order if turn_id in self.raw_turns]
        valid_buffered = [
            turn for turn in buffered_turns if turn["valid"] and turn["status"] in VALID_TURN_STATUSES
        ]
        compression_ratio = round(total_memory_chars / total_raw_chars, 4) if total_raw_chars else 1.0

        return {
            "initialized": self.graph_initialized,
            "cold_start_threshold": self.cold_start_threshold,
            "valid_turn_count": self._count_valid_turns(),
            "buffered_turn_count": len(buffered_turns),
            "coarse_event_count": len(coarse_nodes),
            "coarse_edge_count": len(coarse_edges),
            "entity_node_count": len(self.entity_nodes),
            "entity_edge_count": len(self.entity_edges),
            "temporal_node_count": len(self.temporal_nodes),
            "temporal_edge_count": len(self.temporal_edges),
            "causal_node_count": len(self.causal_nodes),
            "causal_edge_count": len(self.causal_edges),
            "compression_ratio": compression_ratio,
            "estimated_space_saved": round(max(0.0, 1.0 - compression_ratio), 4),
            "events": coarse_nodes,
            "coarse_edges": coarse_edges,
            "raw_turns": buffered_turns,
            "valid_buffer": valid_buffered,
        }

    def export_graph(self) -> nx.Graph:
        exported = nx.Graph()
        for event_id in self.coarse_order:
            event = self.coarse_events[event_id]
            exported.add_node(
                event_id,
                title=event.title,
                summary=event.summary,
                entity_count=len(event.entity_node_ids),
                temporal_count=len(event.temporal_node_ids),
                causal_count=len(event.causal_node_ids),
                turn_count=len(event.turn_ids),
                survival_weight=event.survival_weight,
            )
        for edge in self._compute_coarse_edges():
            exported.add_edge(edge["source"], edge["target"], combined_score=edge["score"])
        return exported

    def close(self) -> None:
        self.store.close()

    # --- Internal: Initialization ---

    def _initialize_from_buffer(self) -> None:
        valid_turns = [
            self.raw_turns[turn_id]
            for turn_id in self.raw_turn_order
            if turn_id in self.raw_turns and self.raw_turns[turn_id].valid
        ]
        clusters = self._cluster_turns(valid_turns)
        for cluster_turn_ids in clusters:
            turns = [self.raw_turns[turn_id] for turn_id in cluster_turn_ids]
            event = self._create_coarse_event_from_turns(turns)
            for turn in turns:
                self._assign_turn_to_event(turn, event.id)
                extracted = self.extractor.extract(
                    turn.question,
                    turn.answer,
                    visual_description=turn.visual_description,
                )
                self._apply_turn_to_subgraphs(turn.id, event.id)
            self._update_event_summary(event.id, turns[-1])
            self._refresh_event_metrics(event.id)
        self.graph_initialized = True
        self.recent_events.append(
            UpdateEvent(
                event_type="graph_initialized",
                message="冷启动完成，已构建火灾场景双粒度多重记忆图谱。",
                details={"coarse_events": len(self.coarse_events), "entity_nodes": len(self.entity_nodes)},
            )
        )

    def _incremental_update(self, turn_id: str) -> None:
        turn = self.raw_turns[turn_id]
        event_id, score = self._route_turn_to_event(turn)
        is_new_event = False
        if event_id is None:
            event = self._create_coarse_event_from_turns([turn])
            event_id = event.id
            is_new_event = True
        else:
            event = self.coarse_events[event_id]
        
        # Count fine-grained nodes before
        old_entity_count = len(self._entity_nodes_for_event(event_id))
        old_temporal_count = len(self._temporal_nodes_for_event(event_id))
        old_causal_count = len(self._causal_nodes_for_event(event_id))
        
        self._assign_turn_to_event(turn, event_id)
        self._apply_turn_to_subgraphs(turn_id, event_id)
        self._update_event_summary(event_id, turn)
        self._refresh_event_metrics(event_id)
        
        # Count fine-grained nodes after
        new_entity_count = len(self._entity_nodes_for_event(event_id))
        new_temporal_count = len(self._temporal_nodes_for_event(event_id))
        new_causal_count = len(self._causal_nodes_for_event(event_id))
        
        # Record changes
        self.recent_events = []  # Clear old events
        if is_new_event:
            self.recent_events.append(
                UpdateEvent(
                    event_type="event_created",
                    message=f"新增粗粒度事件：{event.title}",
                    details={"event_id": event_id},
                )
            )
        else:
            self.recent_events.append(
                UpdateEvent(
                    event_type="event_updated",
                    message=f"更新粗粒度事件：{event.title}",
                    details={"event_id": event_id},
                )
            )
        
        # Report fine-grained node changes
        entity_delta = new_entity_count - old_entity_count
        temporal_delta = new_temporal_count - old_temporal_count
        causal_delta = new_causal_count - old_causal_count
        
        if entity_delta > 0:
            self.recent_events.append(
                UpdateEvent(
                    event_type="entity_created",
                    message=f"在「{event.title}」下新增 {entity_delta} 个实体节点",
                    details={"event_id": event_id, "count": entity_delta},
                )
            )
        if temporal_delta > 0:
            self.recent_events.append(
                UpdateEvent(
                    event_type="temporal_created",
                    message=f"在「{event.title}」下新增 {temporal_delta} 个时序节点",
                    details={"event_id": event_id, "count": temporal_delta},
                )
            )
        if causal_delta > 0:
            self.recent_events.append(
                UpdateEvent(
                    event_type="causal_created",
                    message=f"在「{event.title}」下新增 {causal_delta} 个因果节点",
                    details={"event_id": event_id, "count": causal_delta},
                )
            )
        
        # Detect and resolve conflicts
        corrections = self._detect_and_resolve_conflicts(
            event_id, turn,
            new_causal_ids=[n.id for n in self._causal_nodes_for_event(event_id) if turn.id in n.source_turn_ids],
            new_entity_ids=[n.id for n in self._entity_nodes_for_event(event_id) if turn.id in n.source_turn_ids]
        )
        self.recent_events.extend(corrections)
        
        if self._should_reconstruct(event_id):
            self._local_reconstruct(event_id)
        self._merge_similar_events()

    # --- Internal: Clustering ---

    def _cluster_turns(self, turns: Sequence[RawTurn]) -> List[List[str]]:
        clusters: List[Dict[str, Any]] = []
        for turn in turns:
            best_cluster_index = -1
            best_score = -1.0
            for index, cluster in enumerate(clusters):
                semantic = cosine_similarity(turn.embedding, cluster["centroid"])
                lexical = lexical_overlap(turn.text, cluster["text"])
                temporal = math.exp(-math.log(2) * abs(cluster["last_seen"] - turn.timestamp) / max(self.decay_half_life_seconds, 1))
                entity_match = self._extract_key_entities_match(turn.text, cluster["text"])
                score = 0.68 * max(semantic, 0.0) + 0.22 * lexical + 0.10 * temporal
                if entity_match < 0.3:
                    score *= 0.2
                if score > best_score:
                    best_score = score
                    best_cluster_index = index
            if best_cluster_index >= 0 and best_score >= self.initial_cluster_threshold:
                cluster = clusters[best_cluster_index]
                cluster["turn_ids"].append(turn.id)
                cluster["embeddings"].append(turn.embedding)
                cluster["centroid"] = np.mean(cluster["embeddings"], axis=0)
                cluster["centroid"] = cluster["centroid"] / max(np.linalg.norm(cluster["centroid"]), 1e-8)
                cluster["text"] += f" {turn.text}"
                cluster["last_seen"] = max(cluster["last_seen"], turn.timestamp)
            else:
                clusters.append(
                    {
                        "turn_ids": [turn.id],
                        "embeddings": [turn.embedding],
                        "centroid": turn.embedding.copy(),
                        "text": turn.text,
                        "last_seen": turn.timestamp,
                    }
                )
        return [cluster["turn_ids"] for cluster in clusters]

    def _extract_key_entities_match(self, text_a: str, text_b: str) -> float:
        """Extract key entities (locations, persons, specific events) and compute match score.
        Returns 0.0-1.0, where 1.0 means perfect match of key entities.
        """
        import re
        loc_patterns = [
            r'([\u4e00-\u9fa5]{2,4}(?:市|县|镇|村|区|路|街|栋|仓库|房屋|工厂|学校))',
            r'(工业园区|农村|城区|郊区|市区)',
        ]
        person_patterns = [
            r'([张王李赵刘陈杨黄周吴]{1,2}[某]?)',
            r'(管理员|消防员|警察|负责人|户主)',
        ]
        event_patterns = [
            r'(火灾|爆炸|泄漏|坍塌)',
        ]
        def extract_entities(text: str) -> Dict[str, set]:
            entities = {"location": set(), "person": set(), "event": set()}
            for pattern in loc_patterns:
                entities["location"].update(re.findall(pattern, text))
            for pattern in person_patterns:
                entities["person"].update(re.findall(pattern, text))
            for pattern in event_patterns:
                entities["event"].update(re.findall(pattern, text))
            return entities
        entities_a = extract_entities(text_a)
        entities_b = extract_entities(text_b)
        scores = []
        for key in ["location", "person", "event"]:
            set_a = entities_a[key]
            set_b = entities_b[key]
            if not set_a or not set_b:
                continue
            intersection = len(set_a & set_b)
            union = len(set_a | set_b)
            scores.append(intersection / union if union > 0 else 0.0)
        return sum(scores) / len(scores) if scores else 1.0

    # --- Internal: Coarse event creation ---

    def _create_coarse_event_from_turns(self, turns: Sequence[RawTurn]) -> CoarseEventNode:
        turn_texts = [turn.text for turn in turns]
        title = self.summarizer.build_event_title(turn_texts)
        summary = self.summarizer.build_event_summary(turn_texts)
        visual_description = "；".join(
            dict.fromkeys([turn.visual_description for turn in turns if turn.visual_description])
        )
        embedding_source = f"{title}\n{summary}\n{visual_description}"
        event = CoarseEventNode(
            id=self._new_id("event"),
            title=title,
            summary=summary,
            embedding=self.embedding_model.encode_multimodal(embedding_source, visual_description),
            created_at=now_ts(),
            updated_at=now_ts(),
            start_time=turns[0].timestamp if turns else now_ts(),
            end_time=turns[-1].timestamp if turns else now_ts(),
            visual_description=visual_description,
            survival_weight=1.0,
            importance_coefficient=self._compute_event_importance(turns),
        )
        self.coarse_events[event.id] = event
        self.coarse_order.append(event.id)
        self.store.upsert_coarse_event(event)
        return event

    def _compute_event_importance(self, turns: Sequence[RawTurn]) -> float:
        if not turns:
            return 0.5
        max_saliency = max((t.saliency_score for t in turns), default=0.0)
        has_critical = any(
            any(kw in t.text for kw in CRITICAL_KEYWORDS)
            for t in turns
        )
        return round(min(1.0, 0.3 + 0.3 * max_saliency + 0.4 * int(has_critical)), 4)

    # --- Internal: Routing ---

    def _route_turn_to_event(self, turn: RawTurn) -> Tuple[Optional[str], float]:
        best_id: Optional[str] = None
        best_score = -1.0
        for event_id in self.coarse_order:
            event = self.coarse_events[event_id]
            semantic = cosine_similarity(turn.embedding, event.embedding)
            text_overlap = lexical_overlap(turn.text, f"{event.title} {event.summary}")
            temporal = math.exp(-math.log(2) * abs(event.updated_at - turn.timestamp) / max(self.decay_half_life_seconds, 1))
            # Also match against all turn texts associated with this event for better entity matching
            event_all_texts = f"{event.title} {event.summary} " + " ".join([
                self.raw_turns[tid].text for tid in event.turn_ids if tid in self.raw_turns
            ])
            entity_match = self._extract_key_entities_match(turn.text, event_all_texts)
            recency_boost = 0.15 if event_id == self.coarse_order[-1] else 0.0
            score = 0.52 * max(semantic, 0.0) + 0.18 * text_overlap + 0.30 * temporal + recency_boost
            if entity_match < 0.3:
                score *= 0.15
            if score > best_score:
                best_score = score
                best_id = event_id
        effective_threshold = self.new_event_threshold
        if len(self.coarse_order) <= 3:
            effective_threshold *= 0.55
        if best_id is None or best_score < effective_threshold:
            return None, best_score
        return best_id, best_score

    def _assign_turn_to_event(self, turn: RawTurn, event_id: str) -> None:
        event = self.coarse_events[event_id]
        turn.event_id = event_id
        turn.status = "assigned"
        if turn.id not in event.turn_ids:
            event.turn_ids.append(turn.id)
        if turn.visual_description:
            visuals = list(dict.fromkeys([event.visual_description, turn.visual_description]))
            event.visual_description = "；".join([item for item in visuals if item])[:260]
        event.last_hit_at = now_ts()
        event.updated_at = now_ts()
        event.end_time = turn.timestamp
        event.recent_hit_turn_ids = (event.recent_hit_turn_ids + [turn.id])[-8:]
        consistency = cosine_similarity(turn.embedding, event.embedding)
        event.recent_consistency_scores = (event.recent_consistency_scores + [consistency])[-8:]
        event.survival_weight = min(1.0, event.survival_weight + 0.05)
        self.store.upsert_raw_turn(turn)
        self.store.upsert_coarse_event(event)

    # --- Internal: Sub-graph application (Patent Step 3) ---

    def _apply_turn_to_subgraphs(self, turn_id: str, event_id: str) -> None:
        turn = self.raw_turns[turn_id]
        
        # Step 1: Extract all information (no existing_info filter)
        extracted = self.extractor.extract(
            turn.question,
            turn.answer,
            visual_description=turn.visual_description,
        )
        
        # Step 2: Collect existing nodes for reviewer
        existing_causal = []
        for node in self._causal_nodes_for_event(event_id):
            if not node.corrected:
                existing_causal.append({
                    "causal_type": node.node_type,
                    "cause": node.text,
                    "effect": node.text,
                })
        
        existing_entities = {}
        for node in self._entity_nodes_for_event(event_id):
            if not node.corrected:
                existing_entities.setdefault(node.entity_type, []).append(node.name)
        
        existing_temporal = []
        for node in self._temporal_nodes_for_event(event_id):
            existing_temporal.append({
                "stage_name": node.stage_name,
                "summary": node.summary,
            })
        
        # Step 3: Reviewer reviews and potentially replaces entire causal chain
        should_replace_all = False
        if hasattr(self, 'reviewer') and self.reviewer:
            # Review causal
            causal_items = extracted.causal_info if isinstance(extracted.causal_info, list) else []
            if causal_items and existing_causal:
                # Reviewer may return a complete new causal chain (for corrections)
                reviewed_causal, causal_corrections, should_replace_all = self.reviewer.review_causal(
                    causal_items, existing_causal, turn.question + " " + turn.answer
                )
                extracted.causal_info = reviewed_causal
                self.recent_events.extend(causal_corrections)
                
                # If full replacement, delete ALL old causal nodes
                if should_replace_all:
                    print(f"\n========== 审查者全量替换因果链 ==========")
                    self._delete_all_old_causal_nodes(event_id)
            
            # Review temporal
            temporal_items = extracted.temporal_info if isinstance(extracted.temporal_info, list) else []
            if temporal_items:
                extracted.temporal_info = self.reviewer.review_temporal(
                    temporal_items, existing_temporal
                )
            
            # Review entities
            if extracted.entities:
                extracted.entities = self.reviewer.review_entities(
                    extracted.entities, existing_entities
                )
        
        candidate_ids = []

        # Apply to entity-relation graph
        entity_ids = self._upsert_entity_candidates(event_id, turn, extracted)
        candidate_ids.extend(entity_ids)
        self._create_entity_edges(event_id, turn.id, extracted.relations, entity_ids)

        # Apply to temporal topology graph - support multiple temporal nodes per turn
        temporal_ids = self._upsert_temporal_nodes(event_id, turn, extracted)
        candidate_ids.extend(temporal_ids)
        if temporal_ids:
            self._create_temporal_edges(event_id, turn.id, temporal_ids)

        # Apply to causal relation graph
        causal_ids = self._upsert_causal_candidates(event_id, turn, extracted)
        candidate_ids.extend(causal_ids)
        self._create_causal_edges(event_id, turn.id, extracted.causal_info, causal_ids)
        
        # Step 4: Resolve conflicts - actually delete/correct old nodes
        if causal_ids or entity_ids:
            conflict_corrections = self._detect_and_resolve_conflicts(
                event_id, turn, causal_ids, entity_ids
            )
            self.recent_events.extend(conflict_corrections)
            if conflict_corrections:
                print(f"\n========== 冲突解决：修正了 {len(conflict_corrections)} 个节点 ==========")

        turn.candidate_node_ids = list(dict.fromkeys(candidate_ids))
        self.store.upsert_raw_turn(turn)

    # --- Entity-Relation Graph ---

    def _upsert_entity_candidates(
        self,
        event_id: str,
        turn: RawTurn,
        extracted: ExtractedTurn,
    ) -> List[str]:
        event = self.coarse_events[event_id]
        created_ids: List[str] = []
        for entity_type in ENTITY_NODE_TYPES:
            for text in extracted.entities.get(entity_type, []):
                node = self._merge_or_create_entity_node(event_id, entity_type, text, turn)
                created_ids.append(node.id)
                if node.id not in event.entity_node_ids:
                    event.entity_node_ids.append(node.id)
        self.store.upsert_coarse_event(event)
        return created_ids

    def _merge_or_create_entity_node(
        self,
        event_id: str,
        entity_type: str,
        text: str,
        turn: RawTurn,
    ) -> EntityNode:
        normalized = normalize_text(text)
        embedding = self.embedding_model.encode(normalized)
        best_match_id: Optional[str] = None
        best_score = -1.0
        for node in self._entity_nodes_for_event(event_id):
            if node.entity_type != entity_type:
                continue
            semantic = cosine_similarity(embedding, node.embedding)
            lexical = lexical_overlap(normalized, node.name)
            score = 0.70 * max(semantic, 0.0) + 0.30 * lexical
            if score > best_score:
                best_score = score
                best_match_id = node.id

        if best_match_id and best_score >= self.entity_merge_threshold:
            node = self.entity_nodes[best_match_id]
            node.frequency += 1
            node.last_seen = turn.timestamp
            node.importance = max(node.importance, fire_importance(normalized, entity_type, turn.saliency_score))
            node.active_weight = self._compute_active_weight(node, semantic_change=best_score)
            if turn.id not in node.source_turn_ids:
                node.source_turn_ids.append(turn.id)
            node.summary = self.summarizer.fuse_entity(node.summary, normalized)
            self.store.upsert_entity_node(node)
            self.recent_events.append(
                UpdateEvent(
                    event_type="entity_merged",
                    message="实体节点已与同事件内相似实体融合。",
                    details={"event_id": event_id, "entity_id": node.id, "score": round(best_score, 4)},
                )
            )
            return node

        node = EntityNode(
            id=self._new_id("entity"),
            event_id=event_id,
            entity_type=entity_type,
            name=normalized,
            summary=normalized,
            embedding=embedding,
            frequency=1,
            first_seen=turn.timestamp,
            last_seen=turn.timestamp,
            source_turn_ids=[turn.id],
            importance=fire_importance(normalized, entity_type, turn.saliency_score),
            active_weight=1.0,
            pinned=any(keyword in normalized for keyword in ("起火原因", "受困", "伤亡", "爆炸", "坍塌")),
        )
        self.entity_nodes[node.id] = node
        self.store.upsert_entity_node(node)
        self.recent_events.append(
            UpdateEvent(
                event_type="entity_created",
                message="已在事件内创建新的实体节点。",
                details={"event_id": event_id, "entity_id": node.id, "entity_type": entity_type},
            )
        )
        return node

    def _create_entity_edges(
        self,
        event_id: str,
        turn_id: str,
        relations: Sequence[ExtractedRelation],
        candidate_ids: Sequence[str],
    ) -> None:
        # Only use LLM-extracted relations, no co-occurrence fallback
        if not relations:
            return
        
        # Build a map from entity text to node IDs
        text_to_id: Dict[str, List[str]] = {}
        for cid in candidate_ids:
            if cid in self.entity_nodes:
                node = self.entity_nodes[cid]
                text_to_id.setdefault(normalize_text(node.name), []).append(cid)
        
        for rel in relations:
            source_text = normalize_text(rel.source_text)
            target_text = normalize_text(rel.target_text)
            label = normalize_text(rel.relation_label) or "关联"
            
            # Find matching entity nodes
            source_ids = text_to_id.get(source_text, [])
            target_ids = text_to_id.get(target_text, [])
            
            if source_ids and target_ids:
                for sid in source_ids[:1]:  # Take first match to avoid duplicates
                    for tid in target_ids[:1]:
                        if sid != tid:
                            self._upsert_entity_edge(event_id, sid, tid, label, turn_id, 0.9)

    def _upsert_entity_edge(
        self,
        event_id: str,
        source_id: str,
        target_id: str,
        relation_label: str,
        turn_id: str,
        weight_delta: float,
    ) -> None:
        # Keep original direction, don't sort
        edge_id = f"{event_id}|{relation_label}|{source_id}|{target_id}"
        if edge_id in self.entity_edges:
            edge = self.entity_edges[edge_id]
            edge.weight = min(8.0, edge.weight + weight_delta)
            edge.frequency += 1
            edge.last_seen = now_ts()
            if turn_id not in edge.evidence_turn_ids:
                edge.evidence_turn_ids.append(turn_id)
        else:
            self.entity_edges[edge_id] = EntityEdge(
                id=edge_id,
                event_id=event_id,
                source_id=source_id,
                target_id=target_id,
                relation_label=relation_label,
                weight=weight_delta,
                frequency=1,
                last_seen=now_ts(),
                evidence_turn_ids=[turn_id],
            )

    # --- Temporal Topology Graph ---

    def _upsert_temporal_nodes(
        self,
        event_id: str,
        turn: RawTurn,
        extracted: ExtractedTurn,
    ) -> List[str]:
        """Upsert multiple temporal nodes from a single turn.
        
        LLM may return multiple temporal stages in one response (e.g., 上午起火, 下午死亡).
        This method handles all of them.
        """
        event = self.coarse_events[event_id]
        temporal_info = extracted.temporal_info
        
        # Handle list of temporal info (LLM may return multiple stages)
        temporal_items = []
        if isinstance(temporal_info, list):
            temporal_items = [item for item in temporal_info if isinstance(item, dict)]
        elif isinstance(temporal_info, dict) and temporal_info:
            temporal_items = [temporal_info]
        
        if not temporal_items:
            return []
        
        created_ids = []
        for info in temporal_items:
            stage_name = info.get("stage_name", "现场处置")
            summary = info.get("summary", "")
            sensor_ref = info.get("sensor_ref", "")
            
            # Try to merge with existing temporal node of same stage
            best_match_id: Optional[str] = None
            best_score = -1.0
            for node in self._temporal_nodes_for_event(event_id):
                if node.stage_name == stage_name:
                    semantic = cosine_similarity(self.embedding_model.encode(summary), node.embedding)
                    if semantic > best_score:
                        best_score = semantic
                        best_match_id = node.id
            
            if best_match_id and best_score >= self.temporal_merge_threshold:
                node = self.temporal_nodes[best_match_id]
                node.frequency += 1
                node.end_time = turn.timestamp
                node.active_weight = self._compute_active_weight(node, semantic_change=best_score)
                if turn.id not in node.source_turn_ids:
                    node.source_turn_ids.append(turn.id)
                node.summary = self.summarizer.fuse_temporal(node.summary, summary)
                self.store.upsert_temporal_node(node)
                created_ids.append(node.id)
            else:
                start_time_ref = info.get("start_time_ref", "")
                node = TemporalNode(
                    id=self._new_id("temporal"),
                    event_id=event_id,
                    node_type="stage",
                    stage_name=stage_name,
                    summary=summary,
                    embedding=self.embedding_model.encode(f"{stage_name} {summary}"),
                    start_time=start_time_ref if start_time_ref else turn.timestamp,
                    end_time=turn.timestamp,
                    frequency=1,
                    first_seen=turn.timestamp,
                    last_seen=turn.timestamp,
                    importance=0.5,
                    active_weight=1.0,
                    source_turn_ids=[turn.id],
                    sensor_ref=sensor_ref,
                    pinned=False,
                )
                self.temporal_nodes[node.id] = node
                if node.id not in event.temporal_node_ids:
                    event.temporal_node_ids.append(node.id)
                self.store.upsert_temporal_node(node)
                created_ids.append(node.id)
        
        return created_ids

    def _create_temporal_edges(
        self,
        event_id: str,
        turn_id: str,
        new_temporal_ids: List[str],
    ) -> None:
        """Create temporal edges connecting consecutive time windows within an event."""
        temporal_nodes = sorted(
            self._temporal_nodes_for_event(event_id),
            key=lambda n: n.start_time if isinstance(n.start_time, (int, float)) else 0,
        )
        for i in range(len(temporal_nodes) - 1):
            source = temporal_nodes[i]
            target = temporal_nodes[i + 1]
            edge_id = f"{event_id}|时序后继|{source.id}|{target.id}"
            if edge_id not in self.temporal_edges:
                self.temporal_edges[edge_id] = TemporalEdge(
                    id=edge_id,
                    event_id=event_id,
                    source_id=source.id,
                    target_id=target.id,
                    relation_label="之后",
                    weight=1.0,
                    last_seen=now_ts(),
                )

    # --- Causal Relation Graph ---

    def _upsert_causal_candidates(
        self,
        event_id: str,
        turn: RawTurn,
        extracted: ExtractedTurn,
    ) -> List[str]:
        causal_info = extracted.causal_info
        # Handle list of causal info (LLM may return multiple causal relationships)
        causal_items = []
        if isinstance(causal_info, list):
            causal_items = [item for item in causal_info if isinstance(item, dict)]
        elif isinstance(causal_info, dict) and causal_info:
            causal_items = [causal_info]
        
        if not causal_items:
            return []
        
        # Filter items that have both cause and effect
        valid_items = [item for item in causal_items if item.get("cause") and item.get("effect")]
        
        # If no valid cause-effect pairs, don't create any nodes (causal needs at least cause+effect)
        if not valid_items:
            return []
        
        event = self.coarse_events[event_id]
        created_ids: List[str] = []
        
        for item in valid_items:
            causal_type = item.get("causal_type", "state_change")
            cause_text = item.get("cause", "")
            effect_text = item.get("effect", "")
            full_text = item.get("text", f"{cause_text}导致{effect_text}")
            
            # Create cause node
            cause_node = self._merge_or_create_causal_node(event_id, "cause", cause_text, turn)
            if cause_node.id not in created_ids:
                created_ids.append(cause_node.id)
            if cause_node.id not in event.causal_node_ids:
                event.causal_node_ids.append(cause_node.id)
            
            # Create effect node
            effect_node = self._merge_or_create_causal_node(event_id, "effect", effect_text, turn)
            if effect_node.id not in created_ids:
                created_ids.append(effect_node.id)
            if effect_node.id not in event.causal_node_ids:
                event.causal_node_ids.append(effect_node.id)
        
        self.store.upsert_coarse_event(event)
        return created_ids

    def _merge_or_create_causal_node(
        self,
        event_id: str,
        causal_type: str,
        text: str,
        turn: RawTurn,
    ) -> CausalNode:
        normalized = normalize_text(text)
        embedding = self.embedding_model.encode(normalized)
        best_match_id: Optional[str] = None
        best_score = -1.0
        # Match nodes regardless of causal_type - same text should merge
        for node in self._causal_nodes_for_event(event_id):
            semantic = cosine_similarity(embedding, node.embedding)
            lexical = lexical_overlap(normalized, node.text)
            score = 0.70 * max(semantic, 0.0) + 0.30 * lexical
            if score > best_score:
                best_score = score
                best_match_id = node.id

        if best_match_id and best_score >= self.entity_merge_threshold:
            node = self.causal_nodes[best_match_id]
            node.frequency += 1
            node.last_seen = turn.timestamp
            node.importance = max(node.importance, fire_importance(normalized, causal_type, turn.saliency_score))
            node.active_weight = self._compute_active_weight(node, semantic_change=best_score)
            if turn.id not in node.source_turn_ids:
                node.source_turn_ids.append(turn.id)
            node.text = self.summarizer.fuse_causal(node.text, normalized)
            self.store.upsert_causal_node(node)
            return node

        node = CausalNode(
            id=self._new_id("causal"),
            event_id=event_id,
            node_type=causal_type,
            text=normalized,
            embedding=embedding,
            frequency=1,
            first_seen=turn.timestamp,
            last_seen=turn.timestamp,
            source_turn_ids=[turn.id],
            importance=fire_importance(normalized, causal_type, turn.saliency_score),
            active_weight=1.0,
            pinned=any(keyword in normalized for keyword in ("起火原因", "受困", "伤亡", "爆炸", "坍塌")),
        )
        self.causal_nodes[node.id] = node
        self.store.upsert_causal_node(node)
        self.recent_events.append(
            UpdateEvent(
                event_type="causal_created",
                message="已创建新的因果节点。",
                details={"event_id": event_id, "causal_id": node.id, "causal_type": causal_type},
            )
        )
        return node

    def _delete_all_old_causal_nodes(self, event_id: str) -> int:
        """Delete ALL old causal nodes for an event (for full replacement).
        Returns number of nodes deleted.
        """
        deleted_count = 0
        for node in self._causal_nodes_for_event(event_id):
            if node.corrected:
                continue
            
            print(f"  >>> 删除旧因果节点: {node.text[:40]}...")
            
            # Mark as corrected
            node.corrected = True
            node.correction_source = "full_replacement"
            node.active_weight *= 0.1
            node.confidence = 0.1
            
            # Delete edges connected to this node
            self._remove_edges_for_causal_node(node.id)
            
            # Remove from event's causal_node_ids
            event = self.coarse_events.get(event_id)
            if event and node.id in event.causal_node_ids:
                event.causal_node_ids.remove(node.id)
                self.store.upsert_coarse_event(event)
            
            self.store.upsert_causal_node(node)
            deleted_count += 1
        
        if deleted_count > 0:
            print(f"\n========== 全量替换：删除了 {deleted_count} 个旧因果节点 ==========")
        
        return deleted_count
    
    def _delete_old_causal_nodes_by_text(self, event_id: str, delete_instructions: List[Dict[str, Any]]) -> int:
        """Delete old causal nodes matching the cause/effect text from reviewer.
        Returns number of nodes deleted.
        """
        if not delete_instructions:
            return 0
        
        deleted_count = 0
        for instruction in delete_instructions:
            cause_text = normalize_text(instruction.get("cause", ""))
            effect_text = normalize_text(instruction.get("effect", ""))
            
            # Find matching causal nodes
            for node in self._causal_nodes_for_event(event_id):
                if node.corrected:
                    continue
                
                node_text = normalize_text(node.text)
                # Match if node text contains cause or effect text
                if (cause_text and (cause_text in node_text or node_text in cause_text)) or \
                   (effect_text and (effect_text in node_text or node_text in effect_text)):
                    print(f"  >>> 审查者标记删除旧节点: {node.text[:40]}...")
                    
                    # Mark as corrected
                    node.corrected = True
                    node.correction_source = "reviewer_replacement"
                    node.active_weight *= 0.1
                    node.confidence = 0.1
                    
                    # Delete edges connected to this node
                    self._remove_edges_for_causal_node(node.id)
                    
                    # Remove from event's causal_node_ids
                    event = self.coarse_events.get(event_id)
                    if event and node.id in event.causal_node_ids:
                        event.causal_node_ids.remove(node.id)
                        self.store.upsert_coarse_event(event)
                    
                    self.store.upsert_causal_node(node)
                    deleted_count += 1
        
        if deleted_count > 0:
            print(f"\n========== 审查者删除了 {deleted_count} 个旧因果节点 ==========")
        
        return deleted_count
    
    def _delete_old_causal_nodes_by_indices(self, event_id: str, indices_to_delete: List[int]) -> None:
        """Delete old causal nodes specified by reviewer (1-based indices)."""
        if not indices_to_delete:
            return
        
        # Get all non-corrected causal nodes for this event
        existing_causal_nodes = [
            node for node in self._causal_nodes_for_event(event_id)
            if not node.corrected
        ]
        
        # Sort by last_seen to match the order in review_causal
        existing_causal_nodes.sort(key=lambda n: n.last_seen)
        
        deleted_count = 0
        for idx in indices_to_delete:
            if 1 <= idx <= len(existing_causal_nodes):
                old_node = existing_causal_nodes[idx - 1]  # Convert to 0-based
                print(f"  >>> 审查者标记删除旧节点: {old_node.text[:40]}...")
                
                # Mark as corrected
                old_node.corrected = True
                old_node.correction_source = "reviewer_conflict_resolution"
                old_node.active_weight *= 0.1
                old_node.confidence = 0.1
                
                # Delete edges connected to this node
                self._remove_edges_for_causal_node(old_node.id)
                
                # Remove from event's causal_node_ids
                event = self.coarse_events.get(event_id)
                if event and old_node.id in event.causal_node_ids:
                    event.causal_node_ids.remove(old_node.id)
                    self.store.upsert_coarse_event(event)
                
                self.store.upsert_causal_node(old_node)  # Update in storage
                deleted_count += 1
                
                self.recent_events.append(
                    UpdateEvent(
                        event_type="causal_corrected",
                        message=f"审查者修正因果节点：「{old_node.text[:30]}」已被标记为过时",
                        details={"old_id": old_node.id, "correction_source": "reviewer"},
                    )
                )
        
        if deleted_count > 0:
            print(f"\n========== 审查者删除了 {deleted_count} 个旧因果节点 ==========")
    
    def _create_causal_edges(
        self,
        event_id: str,
        turn_id: str,
        causal_info: Dict[str, Any],
        candidate_ids: Sequence[str],
    ) -> None:
        if not causal_info:
            return
        
        # Handle list format - process ALL causal items
        causal_items = []
        if isinstance(causal_info, list):
            causal_items = [item for item in causal_info if isinstance(item, dict) and item.get("cause") and item.get("effect")]
        elif isinstance(causal_info, dict) and causal_info.get("cause") and causal_info.get("effect"):
            causal_items = [causal_info]
        
        if not causal_items:
            return
        
        # Get all causal nodes for this event
        all_causal_nodes = {
            node.id: node 
            for node in self.causal_nodes.values() 
            if node.event_id == event_id
        }
        
        # For each causal item, create edge between cause and effect nodes
        for item in causal_items:
            cause_text = normalize_text(item.get("cause", ""))
            effect_text = normalize_text(item.get("effect", ""))
            
            # Find matching nodes
            cause_node_id = None
            effect_node_id = None
            
            for node_id, node in all_causal_nodes.items():
                node_text = normalize_text(node.text)
                if cause_text in node_text or node_text in cause_text:
                    cause_node_id = node_id
                if effect_text in node_text or node_text in effect_text:
                    effect_node_id = node_id
            
            if cause_node_id and effect_node_id and cause_node_id != effect_node_id:
                edge_id = f"{event_id}|导致|{cause_node_id}|{effect_node_id}"
                if edge_id not in self.causal_edges:
                    self.causal_edges[edge_id] = CausalEdge(
                        id=edge_id,
                        event_id=event_id,
                        source_id=cause_node_id,
                        target_id=effect_node_id,
                        relation_label="导致",
                        weight=1.0,
                        frequency=1,
                        last_seen=now_ts(),
                        evidence_turn_ids=[turn_id],
                    )

    # --- Event summary update ---

    def _update_event_summary(self, event_id: str, turn: RawTurn) -> None:
        event = self.coarse_events[event_id]
        recent_turn_texts = [
            self.raw_turns[turn_id].text
            for turn_id in event.turn_ids[-4:]
            if turn_id in self.raw_turns
        ]
        event.summary = self.summarizer.fuse_summary(
            previous_summary=event.summary,
            local_summary=turn.local_summary,
            recent_turn_texts=recent_turn_texts,
        )
        internal_texts = (
            [node.summary for node in self._entity_nodes_for_event(event_id)]
            + [node.summary for node in self._temporal_nodes_for_event(event_id)]
            + [node.text for node in self._causal_nodes_for_event(event_id)]
        )
        embedding_source = f"{event.title}\n{event.summary}\n{' '.join(internal_texts[-8:])}"
        event.embedding = self.embedding_model.encode_multimodal(embedding_source, event.visual_description)
        event.compression_ratio = self._event_compression_ratio(event_id)
        event.updated_at = now_ts()
        self.store.upsert_coarse_event(event)

    def _event_compression_ratio(self, event_id: str) -> float:
        event = self.coarse_events[event_id]
        raw_chars = sum(len(self.raw_turns[tid].text) for tid in event.turn_ids if tid in self.raw_turns)
        memory_chars = len(event.summary)
        memory_chars += sum(len(self.entity_nodes[eid].summary) for eid in event.entity_node_ids if eid in self.entity_nodes)
        memory_chars += sum(len(self.temporal_nodes[tid].summary) for tid in event.temporal_node_ids if tid in self.temporal_nodes)
        memory_chars += sum(len(self.causal_nodes[cid].text) for cid in event.causal_node_ids if cid in self.causal_nodes)
        return round(memory_chars / raw_chars, 4) if raw_chars else 1.0

    def _refresh_event_metrics(self, event_id: str) -> None:
        if event_id not in self.coarse_events:
            return
        event = self.coarse_events[event_id]
        entity_nodes = self._entity_nodes_for_event(event_id)
        event.node_count = len(entity_nodes) + len(self._temporal_nodes_for_event(event_id)) + len(self._causal_nodes_for_event(event_id))
        event.duplicate_ratio = self._entity_duplicate_ratio(entity_nodes)
        event.compression_ratio = self._event_compression_ratio(event_id)
        event.survival_weight = self._compute_event_survival_weight(event)
        self.store.upsert_coarse_event(event)

    # --- Active Weight Formula (Patent Step 5) ---

    def _compute_active_weight(
        self,
        node: Any,
        semantic_change: float = 1.0,
    ) -> float:
        """W(i,t) = alpha * time_freshness + beta * access_freq + gamma * importance + delta * semantic_change"""
        time_freshness = math.exp(-math.log(2) * max(0.0, now_ts() - node.last_seen) / max(self.decay_half_life_seconds, 1))
        access_freq = min(1.0, node.access_count / 5.0) if hasattr(node, 'access_count') else 0.0
        importance = node.importance if hasattr(node, 'importance') else 0.5
        normalized_semantic = min(1.0, max(0.0, semantic_change))
        weight = (
            self.alpha * time_freshness
            + self.beta * access_freq
            + self.gamma * importance
            + self.delta * normalized_semantic
        )
        return round(min(1.0, max(0.0, weight)), 4)

    def _refresh_all_active_weights(self) -> None:
        current_time = now_ts()
        for node in self.entity_nodes.values():
            if node.pinned:
                node.active_weight = 1.0
            else:
                node.active_weight = self._compute_active_weight(node)
        for node in self.temporal_nodes.values():
            if node.pinned:
                node.active_weight = 1.0
            else:
                node.active_weight = self._compute_active_weight(node)
        for node in self.causal_nodes.values():
            if node.pinned:
                node.active_weight = 1.0
            else:
                node.active_weight = self._compute_active_weight(node)
        for event in self.coarse_events.values():
            time_weight = math.exp(-math.log(2) * max(0.0, current_time - event.updated_at) / max(self.decay_half_life_seconds, 1))
            event.survival_weight = round(min(1.0, time_weight + 0.08 * min(1.0, event.node_count / 10)), 4)

    def _compute_event_survival_weight(self, event: CoarseEventNode) -> float:
        time_weight = math.exp(-math.log(2) * max(0.0, now_ts() - event.updated_at) / max(self.decay_half_life_seconds, 1))
        return round(min(1.0, time_weight + 0.08 * min(1.0, event.node_count / 10)), 4)

    # --- Dynamic Compression (Patent Step 5) ---

    def _detect_conflict_signals(self, text: str) -> float:
        """Detect conflict/correction signals in text.
        Returns 0.0-1.0, where 1.0 means strong conflict signal.
        """
        conflict_keywords = [
            "并非", "不是", "实际是", "其实是", "真实原因", "经专家评定", 
            "经调查", "纠正", "修正", "错误", "误导", "澄清", "重新认定",
            "排除", "否定", "推翻", "改为", "更改", "更正"
        ]
        text_lower = text.lower()
        match_count = sum(1 for kw in conflict_keywords if kw in text_lower)
        return min(1.0, match_count / 2.0)  # 2 matches = 1.0

    def _compute_evidence_weight(self, text: str, frequency: int = 1, importance: float = 0.5, last_seen: float = 0.0) -> float:
        """Compute evidence weight for a piece of information.
        Higher weight = more credible evidence.
        """
        # Signal strength from correction keywords
        signal = self._detect_conflict_signals(text)
        
        # Authority boost (expert review, investigation, etc.)
        authority_keywords = ["经专家", "经调查", "警方认定", "官方", "权威", " confirmed", "verified"]
        authority_boost = 0.3 if any(kw in text for kw in authority_keywords) else 0.0
        
        # Recency boost
        time_diff = max(0.0, now_ts() - last_seen) if last_seen > 0 else 0
        recency = math.exp(-math.log(2) * time_diff / max(self.decay_half_life_seconds, 1))
        
        # Base weight
        base = 0.3 + 0.2 * min(1.0, frequency / 5.0) + 0.2 * importance
        
        # Final weight
        weight = base * (1.0 + signal + authority_boost) + 0.3 * recency
        return min(1.0, weight)

    def _detect_and_resolve_conflicts(self, event_id: str, turn: RawTurn, new_causal_ids: List[str], new_entity_ids: List[str]) -> List[UpdateEvent]:
        """Detect conflicts between new nodes and existing nodes, resolve them.
        Returns list of UpdateEvent describing corrections made.
        """
        corrections = []
        conflict_signal = self._detect_conflict_signals(turn.text)
        
        print(f"\n========== 冲突检测开始 ==========")
        print(f"冲突信号强度: {conflict_signal:.2f}")
        print(f"新因果节点数: {len(new_causal_ids)}, 新实体节点数: {len(new_entity_ids)}")
        
        if conflict_signal < 0.3:
            print(f"冲突信号弱，跳过冲突解决")
            return corrections  # No strong conflict signal, skip
        
        # Check causal nodes for conflicts
        for new_id in new_causal_ids:
            if new_id not in self.causal_nodes:
                continue
            new_node = self.causal_nodes[new_id]
            new_weight = self._compute_evidence_weight(
                new_node.text, 
                frequency=new_node.frequency,
                importance=new_node.importance,
                last_seen=new_node.last_seen
            )
            print(f"检查因果节点: {new_node.text[:30]}... 权重={new_weight:.2f}")
            
            # Find conflicting old nodes (same event, different source turns, low semantic similarity)
            for old_node in self._causal_nodes_for_event(event_id):
                if old_node.id == new_id or old_node.corrected:
                    continue
                # Skip if from same turn
                if any(tid in old_node.source_turn_ids for tid in new_node.source_turn_ids):
                    continue
                
                semantic_sim = cosine_similarity(new_node.embedding, old_node.embedding)
                print(f"  对比旧节点: {old_node.text[:30]}... 相似度={semantic_sim:.2f}")
                # If semantically different but same causal_type, likely conflict
                if semantic_sim < 0.5 and new_node.node_type == old_node.node_type:
                    old_weight = self._compute_evidence_weight(
                        old_node.text,
                        frequency=old_node.frequency,
                        importance=old_node.importance,
                        last_seen=old_node.last_seen
                    )
                    print(f"  旧节点权重: {old_weight:.2f}, 新节点需>{old_weight*1.3:.2f}")
                    
                    # If new evidence is stronger, mark old node as corrected
                    if new_weight > old_weight * 1.3:  # 30% stronger threshold
                        print(f"  >>> 修正旧节点！")
                        old_node.corrected = True
                        old_node.correction_source = "conflict_resolution"
                        old_node.active_weight *= 0.1  # Drastically reduce weight
                        old_node.confidence = 0.1
                        
                        # Delete edges connected to corrected node
                        self._remove_edges_for_causal_node(old_node.id)
                        
                        corrections.append(
                            UpdateEvent(
                                event_type="causal_corrected",
                                message=f"修正因果节点：「{old_node.text[:30]}」→「{new_node.text[:30]}」",
                                details={"old_id": old_node.id, "new_id": new_id},
                            )
                        )
        
        # Check entity nodes for conflicts (similar logic)
        for new_id in new_entity_ids:
            if new_id not in self.entity_nodes:
                continue
            new_node = self.entity_nodes[new_id]
            new_weight = self._compute_evidence_weight(
                new_node.name,
                frequency=new_node.frequency,
                importance=new_node.importance,
                last_seen=new_node.last_seen
            )
            
            for old_node in self._entity_nodes_for_event(event_id):
                if old_node.id == new_id or old_node.corrected:
                    continue
                if any(tid in old_node.source_turn_ids for tid in new_node.source_turn_ids):
                    continue
                
                semantic_sim = cosine_similarity(new_node.embedding, old_node.embedding)
                if semantic_sim < 0.5 and new_node.entity_type == old_node.entity_type:
                    old_weight = self._compute_evidence_weight(
                        old_node.name,
                        frequency=old_node.frequency,
                        importance=old_node.importance,
                        last_seen=old_node.last_seen
                    )
                    
                    if new_weight > old_weight * 1.3:
                        old_node.corrected = True
                        old_node.correction_source = "conflict_resolution"
                        old_node.active_weight *= 0.1
                        old_node.confidence = 0.1
                        
                        self._remove_edges_for_entity_node(old_node.id)
                        
                        corrections.append(
                            UpdateEvent(
                                event_type="entity_corrected",
                                message=f"修正实体节点：「{old_node.name}」→「{new_node.name}」",
                                details={"old_id": old_node.id, "new_id": new_id},
                            )
                        )
        
        return corrections

    def _remove_edges_for_causal_node(self, node_id: str) -> None:
        """Remove all causal edges connected to a node."""
        to_remove = [eid for eid, edge in self.causal_edges.items() 
                     if edge.source_id == node_id or edge.target_id == node_id]
        for eid in to_remove:
            del self.causal_edges[eid]

    def _remove_edges_for_entity_node(self, node_id: str) -> None:
        """Remove all entity edges connected to a node."""
        to_remove = [eid for eid, edge in self.entity_edges.items() 
                     if edge.source_id == node_id or edge.target_id == node_id]
        for eid in to_remove:
            del self.entity_edges[eid]

    def _apply_compression(self) -> None:
        """Differentiated compression for the three sub-graphs."""
        self._refresh_all_active_weights()
        removed_entities = 0
        removed_temporal = 0
        removed_causal = 0

        # Entity graph: merge semantically redundant nodes
        for event_id in self.coarse_order:
            entity_nodes = self._entity_nodes_for_event(event_id)
            for i, source in enumerate(entity_nodes):
                if source.pinned or source.active_weight >= self.compress_threshold:
                    continue
                for target in entity_nodes[i + 1:]:
                    if target.pinned or target.active_weight >= self.compress_threshold:
                        continue
                    if source.entity_type != target.entity_type:
                        continue
                    semantic = cosine_similarity(source.embedding, target.embedding)
                    lexical = lexical_overlap(source.name, target.name)
                    if 0.70 * max(semantic, 0.0) + 0.30 * lexical >= self.entity_merge_threshold:
                        self._merge_entity_nodes(source, target)
                        removed_entities += 1

        # Temporal graph: merge consecutive time windows with small changes
        for event_id in self.coarse_order:
            temporal_nodes = sorted(
                self._temporal_nodes_for_event(event_id), 
                key=lambda n: n.start_time if isinstance(n.start_time, (int, float)) else 0
            )
            for i in range(len(temporal_nodes) - 1):
                source = temporal_nodes[i]
                target = temporal_nodes[i + 1]
                if source.pinned or target.pinned:
                    continue
                if source.active_weight >= self.compress_threshold and target.active_weight >= self.compress_threshold:
                    continue
                semantic = cosine_similarity(source.embedding, target.embedding)
                if semantic >= self.temporal_merge_threshold:
                    self._merge_temporal_nodes(source, target)
                    removed_temporal += 1

        # Causal graph: prune short-lived transitional nodes
        for node_id, node in list(self.causal_nodes.items()):
            if node.pinned:
                continue
            if node.active_weight >= self.compress_threshold:
                continue
            if node.importance >= 0.70:
                continue
            self._remove_causal_node(node_id)
            removed_causal += 1

        if removed_entities:
            self.recent_events.append(
                UpdateEvent(
                    event_type="entity_compressed",
                    message="实体节点融合压缩已完成。",
                    details={"removed": removed_entities},
                )
            )
        if removed_temporal:
            self.recent_events.append(
                UpdateEvent(
                    event_type="temporal_compressed",
                    message="时序阶段节点归并压缩已完成。",
                    details={"removed": removed_temporal},
                )
            )
        if removed_causal:
            self.recent_events.append(
                UpdateEvent(
                    event_type="causal_pruned",
                    message="低活跃权重因果节点已结构性剪枝。",
                    details={"removed": removed_causal, "threshold": self.compress_threshold},
                )
            )

    def _merge_entity_nodes(self, source: EntityNode, target: EntityNode) -> None:
        source.frequency += target.frequency
        source.last_seen = max(source.last_seen, target.last_seen)
        source.access_count += target.access_count
        source.summary = self.summarizer.fuse_entity(source.summary, target.name)
        source.source_turn_ids = list(dict.fromkeys(source.source_turn_ids + target.source_turn_ids))
        self._remove_entity_node(target.id)
        self.store.upsert_entity_node(source)

    def _merge_temporal_nodes(self, source: TemporalNode, target: TemporalNode) -> None:
        source.frequency += target.frequency
        source.end_time = max(source.end_time, target.end_time)
        source.access_count += target.access_count
        source.summary = self.summarizer.fuse_temporal(source.summary, target.summary)
        source.source_turn_ids = list(dict.fromkeys(source.source_turn_ids + target.source_turn_ids))
        self._remove_temporal_node(target.id)
        self.store.upsert_temporal_node(source)

    # --- Retrieval: Multi-layer (Patent Step 6) ---

    def _fine_grained_entry(
        self,
        query_vector: np.ndarray,
        query: str,
        query_tokens: set,
    ) -> List[Dict[str, Any]]:
        """Search across all three sub-graphs to find candidate fine-grained entry points."""
        candidates: List[Dict[str, Any]] = []

        # Entity graph search
        for node in self.entity_nodes.values():
            semantic = cosine_similarity(query_vector, node.embedding)
            lexical = lexical_overlap(query, f"{node.name} {node.summary}")
            score = 0.52 * max(semantic, 0.0) + 0.18 * lexical + 0.18 * node.importance + 0.12 * node.active_weight
            if score > 0.16:
                candidates.append({
                    "node_id": node.id,
                    "event_id": node.event_id,
                    "graph_type": "entity",
                    "name": node.name,
                    "summary": node.summary,
                    "entity_type": node.entity_type,
                    "score": round(score, 4),
                    "active_weight": node.active_weight,
                })

        # Temporal graph search
        for node in self.temporal_nodes.values():
            semantic = cosine_similarity(query_vector, node.embedding)
            lexical = lexical_overlap(query, f"{node.stage_name} {node.summary}")
            score = 0.52 * max(semantic, 0.0) + 0.18 * lexical + 0.18 * node.importance + 0.12 * node.active_weight
            if score > 0.16:
                candidates.append({
                    "node_id": node.id,
                    "event_id": node.event_id,
                    "graph_type": "temporal",
                    "stage_name": node.stage_name,
                    "summary": node.summary,
                    "score": round(score, 4),
                    "active_weight": node.active_weight,
                })

        # Causal graph search
        for node in self.causal_nodes.values():
            semantic = cosine_similarity(query_vector, node.embedding)
            lexical = lexical_overlap(query, node.text)
            score = 0.52 * max(semantic, 0.0) + 0.18 * lexical + 0.18 * node.importance + 0.12 * node.active_weight
            if score > 0.16:
                candidates.append({
                    "node_id": node.id,
                    "event_id": node.event_id,
                    "graph_type": "causal",
                    "text": node.text,
                    "node_type": node.node_type,
                    "score": round(score, 4),
                    "active_weight": node.active_weight,
                })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:8]

    def _retrieve_entity_graph(
        self,
        event_id: str,
        query_vector: np.ndarray,
        query: str,
    ) -> List[Dict[str, Any]]:
        """Retrieve from entity-relation graph within an event."""
        results = []
        for node in self._entity_nodes_for_event(event_id):
            semantic = cosine_similarity(query_vector, node.embedding)
            lexical = lexical_overlap(query, f"{node.name} {node.summary}")
            score = 0.52 * max(semantic, 0.0) + 0.18 * lexical + 0.18 * node.importance + 0.12 * node.active_weight
            if score > 0.16:
                node.access_count += 1
                node.active_weight = self._compute_active_weight(node)
                self.store.upsert_entity_node(node)
                results.append({
                    "node_id": node.id,
                    "name": node.name,
                    "entity_type": node.entity_type,
                    "summary": node.summary,
                    "importance": node.importance,
                    "active_weight": node.active_weight,
                    "score": round(score, 4),
                })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def _retrieve_temporal_graph(
        self,
        event_id: str,
        query_vector: np.ndarray,
        query: str,
    ) -> List[Dict[str, Any]]:
        """Retrieve from temporal topology graph within an event.
        
        Temporal nodes are few (typically 3-10 per event), so we return almost all
        to show the complete timeline chain. Lower threshold to 0.08 and return up to 20 nodes.
        """
        results = []
        for node in self._temporal_nodes_for_event(event_id):
            semantic = cosine_similarity(query_vector, node.embedding)
            lexical = lexical_overlap(query, f"{node.stage_name} {node.summary}")
            score = 0.52 * max(semantic, 0.0) + 0.18 * lexical + 0.18 * node.importance + 0.12 * node.active_weight
            # Lower threshold to include more temporal nodes for complete timeline
            if score > 0.08:
                node.access_count += 1
                node.active_weight = self._compute_active_weight(node)
                self.store.upsert_temporal_node(node)
                results.append({
                    "node_id": node.id,
                    "stage_name": node.stage_name,
                    "summary": node.summary,
                    "start_time": node.start_time,
                    "end_time": node.end_time,
                    "score": round(score, 4),
                })
        # Sort by time sequence first, then by score
        results.sort(key=lambda x: x.get("start_time", 0))
        return results[:20]  # Cap at 20 to avoid too many nodes

    def _retrieve_causal_graph(
        self,
        event_id: str,
        query_vector: np.ndarray,
        query: str,
    ) -> List[Dict[str, Any]]:
        """Retrieve from causal relation graph within an event.
        
        Causal nodes are few (typically 2-8 per event), so we return almost all
        to show the complete causal chain. Lower threshold to 0.08 and return up to 15 nodes.
        """
        results = []
        for node in self._causal_nodes_for_event(event_id):
            semantic = cosine_similarity(query_vector, node.embedding)
            lexical = lexical_overlap(query, node.text)
            score = 0.52 * max(semantic, 0.0) + 0.18 * lexical + 0.18 * node.importance + 0.12 * node.active_weight
            # Lower threshold to include more causal nodes for complete causal chain
            if score > 0.08:
                node.access_count += 1
                node.active_weight = self._compute_active_weight(node)
                self.store.upsert_causal_node(node)
                results.append({
                    "node_id": node.id,
                    "node_type": node.node_type,
                    "text": node.text,
                    "importance": node.importance,
                    "active_weight": node.active_weight,
                    "score": round(score, 4),
                })
        # Sort by importance to show key causal relationships first
        results.sort(key=lambda x: x.get("importance", 0), reverse=True)
        return results[:15]  # Cap at 15 to avoid too many nodes

    # --- Event merge & reconstruct ---

    def _should_reconstruct(self, event_id: str) -> bool:
        event = self.coarse_events[event_id]
        avg_consistency = (
            sum(event.recent_consistency_scores) / len(event.recent_consistency_scores)
            if event.recent_consistency_scores
            else 1.0
        )
        return (
            event.node_count >= 28
            or (event.node_count >= 10 and event.duplicate_ratio >= 0.24)
            or (len(event.recent_consistency_scores) >= 5 and avg_consistency <= 0.24)
        )

    def _local_reconstruct(self, event_id: str) -> None:
        if event_id not in self.coarse_events:
            return
        event = self.coarse_events[event_id]
        turn_ids = list(event.turn_ids)
        if len(turn_ids) < 3:
            return
        drift_turn_ids = self._detect_drift_turns(event_id, turn_ids)
        if len(drift_turn_ids) >= 3 and len(drift_turn_ids) < len(turn_ids):
            stay_turn_ids = [turn_id for turn_id in turn_ids if turn_id not in drift_turn_ids]
            new_event = self._create_coarse_event_from_turns(
                [self.raw_turns[turn_id] for turn_id in drift_turn_ids if turn_id in self.raw_turns]
            )
            self._rebuild_single_event(event_id, stay_turn_ids)
            self._rebuild_single_event(new_event.id, drift_turn_ids)
            self.recent_events.append(
                UpdateEvent(
                    event_type="event_split",
                    message="局部重构检测到语义漂移，并拆分出新的事件节点。",
                    details={"from_event_id": event_id, "to_event_id": new_event.id},
                )
            )
        else:
            self._rebuild_single_event(event_id, turn_ids)
            self.recent_events.append(
                UpdateEvent(
                    event_type="event_rebuilt",
                    message="已对一个事件节点执行局部重构和节点融合。",
                    details={"event_id": event_id},
                )
            )

    def _detect_drift_turns(self, event_id: str, turn_ids: Sequence[str]) -> List[str]:
        event = self.coarse_events[event_id]
        drift_turn_ids: List[str] = []
        for turn_id in turn_ids[-10:]:
            if turn_id not in self.raw_turns:
                continue
            turn = self.raw_turns[turn_id]
            semantic = cosine_similarity(turn.embedding, event.embedding)
            lexical = lexical_overlap(turn.text, event.summary)
            score = 0.70 * max(semantic, 0.0) + 0.30 * lexical
            if score < 0.20:
                drift_turn_ids.append(turn_id)
        return drift_turn_ids

    def _rebuild_single_event(self, event_id: str, turn_ids: Sequence[str]) -> None:
        if event_id not in self.coarse_events:
            return
        event = self.coarse_events[event_id]
        for fine_id in list(event.entity_node_ids):
            self._remove_entity_node(fine_id)
        for fine_id in list(event.temporal_node_ids):
            self._remove_temporal_node(fine_id)
        for fine_id in list(event.causal_node_ids):
            self._remove_causal_node(fine_id)
        event.turn_ids = []
        event.entity_node_ids = []
        event.temporal_node_ids = []
        event.causal_node_ids = []
        event.recent_hit_turn_ids = []
        event.recent_consistency_scores = []
        event.node_count = 0
        event.duplicate_ratio = 0.0
        event.reconstruct_count += 1

        turns = [self.raw_turns[turn_id] for turn_id in turn_ids if turn_id in self.raw_turns]
        event.title = self.summarizer.build_event_title([turn.text for turn in turns], previous_title=event.title)
        event.summary = self.summarizer.build_event_summary([turn.text for turn in turns], event.summary)
        event.visual_description = "；".join(
            dict.fromkeys([turn.visual_description for turn in turns if turn.visual_description])
        )[:260]
        event.embedding = self.embedding_model.encode_multimodal(
            f"{event.title}\n{event.summary}",
            event.visual_description,
        )
        event.updated_at = now_ts()

        for turn in turns:
            self._assign_turn_to_event(turn, event_id)
            self._apply_turn_to_subgraphs(turn.id, event_id)
        self._refresh_event_metrics(event_id)

    def _detect_similar_events(self, threshold: float = 0.56) -> List[Tuple[str, str, float]]:
        similar_pairs: List[Tuple[str, str, float]] = []
        for index, source_id in enumerate(self.coarse_order):
            source = self.coarse_events[source_id]
            for target_id in self.coarse_order[index + 1:]:
                target = self.coarse_events[target_id]
                semantic = cosine_similarity(source.embedding, target.embedding)
                lexical = lexical_overlap(
                    f"{source.title} {source.summary}",
                    f"{target.title} {target.summary}",
                )
                entity_match = self._extract_key_entities_match(
                    f"{source.title} {source.summary}",
                    f"{target.title} {target.summary}",
                )
                score = 0.76 * max(semantic, 0.0) + 0.24 * lexical
                if entity_match < 0.3:
                    score *= 0.1
                if score >= threshold:
                    similar_pairs.append((source_id, target_id, score))
        return similar_pairs

    def _merge_similar_events(self) -> None:
        merged_ids: set[str] = set()
        for source_id, target_id, score in self._detect_similar_events():
            if source_id in merged_ids or target_id in merged_ids:
                continue
            if source_id not in self.coarse_events or target_id not in self.coarse_events:
                continue
            self._merge_events(source_id, target_id, score)
            merged_ids.add(target_id)

    def _merge_events(self, source_id: str, target_id: str, score: float) -> None:
        source = self.coarse_events[source_id]
        target = self.coarse_events[target_id]
        for turn_id in target.turn_ids:
            if turn_id in self.raw_turns:
                turn = self.raw_turns[turn_id]
                turn.event_id = source_id
                if turn_id not in source.turn_ids:
                    source.turn_ids.append(turn_id)
                self.store.upsert_raw_turn(turn)
        for fine_id in list(target.entity_node_ids):
            node = self.entity_nodes.get(fine_id)
            if not node:
                continue
            node.event_id = source_id
            if fine_id not in source.entity_node_ids:
                source.entity_node_ids.append(fine_id)
            self.store.upsert_entity_node(node)
        for fine_id in list(target.temporal_node_ids):
            node = self.temporal_nodes.get(fine_id)
            if not node:
                continue
            node.event_id = source_id
            if fine_id not in source.temporal_node_ids:
                source.temporal_node_ids.append(fine_id)
            self.store.upsert_temporal_node(node)
        for fine_id in list(target.causal_node_ids):
            node = self.causal_nodes.get(fine_id)
            if not node:
                continue
            node.event_id = source_id
            if fine_id not in source.causal_node_ids:
                source.causal_node_ids.append(fine_id)
            self.store.upsert_causal_node(node)
        source.summary = self.summarizer.fuse_summary(source.summary, target.summary, [])
        source.visual_description = "；".join(
            dict.fromkeys([source.visual_description, target.visual_description])
        ).strip("；")[:260]
        source.embedding = self.embedding_model.encode_multimodal(
            f"{source.title}\n{source.summary}",
            source.visual_description,
        )
        self._drop_event(target_id, keep_turn_status=True)
        self._refresh_event_metrics(source_id)
        self.recent_events.append(
            UpdateEvent(
                event_type="event_merge",
                message="相似事件节点已合并，减少宏观态势冗余。",
                details={"kept_id": source_id, "merged_id": target_id, "similarity": round(score, 4)},
            )
        )

    # --- Node removal helpers ---

    def delete_coarse_node(self, node_id: str) -> bool:
        """Public API: delete a coarse event node and all its fine-grained children."""
        if node_id not in self.coarse_events:
            return False
        self._drop_event(node_id, keep_turn_status=False)
        return True

    def delete_turn_by_id(self, turn_id: str) -> bool:
        """Delete a conversation turn and rebuild affected events."""
        if turn_id not in self.raw_turns:
            return False
        turn = self.raw_turns[turn_id]
        affected_event_id = turn.event_id
        
        # Remove turn from storage
        self.store.delete_raw_turn(turn_id)
        if turn_id in self.raw_turn_order:
            self.raw_turn_order.remove(turn_id)
        self.raw_turns.pop(turn_id, None)
        
        # If turn was part of an event, rebuild that event
        if affected_event_id and affected_event_id in self.coarse_events:
            self._rebuild_event_event(affected_event_id)
        
        return True

    def _rebuild_event_event(self, event_id: str) -> None:
        """Rebuild a coarse event from its remaining turns."""
        if event_id not in self.coarse_events:
            return
        event = self.coarse_events[event_id]
        
        # Get remaining turns
        remaining_turn_ids = [tid for tid in event.turn_ids if tid in self.raw_turns]
        if not remaining_turn_ids:
            # No turns left, drop the entire event
            self._drop_event(event_id, keep_turn_status=False)
            return
        
        # Clear existing fine-grained nodes
        for fine_id in list(event.entity_node_ids):
            self._remove_entity_node(fine_id)
        for fine_id in list(event.temporal_node_ids):
            self._remove_temporal_node(fine_id)
        for fine_id in list(event.causal_node_ids):
            self._remove_causal_node(fine_id)
        for edge_id in list(self.entity_edges.keys()):
            if self.entity_edges[edge_id].event_id == event_id:
                self.entity_edges.pop(edge_id, None)
        for edge_id in list(self.temporal_edges.keys()):
            if self.temporal_edges[edge_id].event_id == event_id:
                self.temporal_edges.pop(edge_id, None)
        for edge_id in list(self.causal_edges.keys()):
            if self.causal_edges[edge_id].event_id == event_id:
                self.causal_edges.pop(edge_id, None)
        
        event.entity_node_ids = []
        event.temporal_node_ids = []
        event.causal_node_ids = []
        event.turn_ids = remaining_turn_ids
        
        # Re-extract and rebuild from remaining turns
        all_entities = []
        all_temporal = []
        all_causal = []
        chunks = []
        for tid in remaining_turn_ids:
            turn = self.raw_turns[tid]
            extracted = self.extractor.extract(turn.question, turn.answer)
            all_entities.extend(extracted.entities)
            all_temporal.append(extracted.temporal)
            all_causal.extend(extracted.causal_info)
            chunks.append(turn.text)
        
        # Rebuild summary
        event.summary = self.summarizer.summarize_text(chunks) if chunks else event.summary
        event.embedding = self.embedding_model.encode(event.summary)
        
        # Rebuild fine-grained nodes
        for tid in remaining_turn_ids:
            turn = self.raw_turns[tid]
            self._apply_turn_to_subgraphs(event_id, tid)
        
        self.store.upsert_coarse_event(event)

    def _drop_event(self, event_id: str, keep_turn_status: bool = False) -> None:
        if event_id not in self.coarse_events:
            return
        event = self.coarse_events.pop(event_id)
        if event_id in self.coarse_order:
            self.coarse_order.remove(event_id)
        for fine_id in list(event.entity_node_ids):
            if keep_turn_status and fine_id in self.entity_nodes:
                continue
            self._remove_entity_node(fine_id)
        for fine_id in list(event.temporal_node_ids):
            if keep_turn_status and fine_id in self.temporal_nodes:
                continue
            self._remove_temporal_node(fine_id)
        for fine_id in list(event.causal_node_ids):
            if keep_turn_status and fine_id in self.causal_nodes:
                continue
            self._remove_causal_node(fine_id)
        for edge_id in list(self.entity_edges.keys()):
            if self.entity_edges[edge_id].event_id == event_id:
                self.entity_edges.pop(edge_id, None)
        for edge_id in list(self.temporal_edges.keys()):
            if self.temporal_edges[edge_id].event_id == event_id:
                self.temporal_edges.pop(edge_id, None)
        for edge_id in list(self.causal_edges.keys()):
            if self.causal_edges[edge_id].event_id == event_id:
                self.causal_edges.pop(edge_id, None)
        if not keep_turn_status:
            for turn_id in event.turn_ids:
                if turn_id in self.raw_turns:
                    turn = self.raw_turns[turn_id]
                    turn.event_id = None
                    turn.status = "buffered"
                    turn.candidate_node_ids = []
                    self.store.upsert_raw_turn(turn)
        self.store.delete_coarse_event(event_id)

    def _remove_entity_node(self, fine_id: str) -> None:
        node = self.entity_nodes.pop(fine_id, None)
        if not node:
            return
        if node.event_id in self.coarse_events:
            event = self.coarse_events[node.event_id]
            event.entity_node_ids = [item for item in event.entity_node_ids if item != fine_id]
            self.store.upsert_coarse_event(event)
        for edge_id in list(self.entity_edges.keys()):
            edge = self.entity_edges[edge_id]
            if edge.source_id == fine_id or edge.target_id == fine_id:
                self.entity_edges.pop(edge_id, None)
        self.store.delete_entity_node(fine_id)

    def _remove_temporal_node(self, fine_id: str) -> None:
        node = self.temporal_nodes.pop(fine_id, None)
        if not node:
            return
        if node.event_id in self.coarse_events:
            event = self.coarse_events[node.event_id]
            event.temporal_node_ids = [item for item in event.temporal_node_ids if item != fine_id]
            self.store.upsert_coarse_event(event)
        for edge_id in list(self.temporal_edges.keys()):
            edge = self.temporal_edges[edge_id]
            if edge.source_id == fine_id or edge.target_id == fine_id:
                self.temporal_edges.pop(edge_id, None)
        self.store.delete_temporal_node(fine_id)

    def _remove_causal_node(self, fine_id: str) -> None:
        node = self.causal_nodes.pop(fine_id, None)
        if not node:
            return
        if node.event_id in self.coarse_events:
            event = self.coarse_events[node.event_id]
            event.causal_node_ids = [item for item in event.causal_node_ids if item != fine_id]
            self.store.upsert_coarse_event(event)
        for edge_id in list(self.causal_edges.keys()):
            edge = self.causal_edges[edge_id]
            if edge.source_id == fine_id or edge.target_id == fine_id:
                self.causal_edges.pop(edge_id, None)
        self.store.delete_causal_node(fine_id)

    # --- Utility helpers ---

    def _entity_nodes_for_event(self, event_id: str) -> List[EntityNode]:
        return [node for node in self.entity_nodes.values() if node.event_id == event_id]

    def _temporal_nodes_for_event(self, event_id: str) -> List[TemporalNode]:
        return [node for node in self.temporal_nodes.values() if node.event_id == event_id]

    def _causal_nodes_for_event(self, event_id: str) -> List[CausalNode]:
        return [node for node in self.causal_nodes.values() if node.event_id == event_id]

    def _entity_duplicate_ratio(self, entity_nodes: Sequence[EntityNode]) -> float:
        if len(entity_nodes) < 2:
            return 0.0
        duplicate_hits = 0
        for index, source in enumerate(entity_nodes):
            for target in entity_nodes[index + 1:]:
                if source.entity_type != target.entity_type:
                    continue
                semantic = cosine_similarity(source.embedding, target.embedding)
                lexical = lexical_overlap(source.name, target.name)
                if 0.70 * max(semantic, 0.0) + 0.30 * lexical >= self.entity_merge_threshold:
                    duplicate_hits += 1
        return round(duplicate_hits / max(len(entity_nodes), 1), 4)

    def _compute_coarse_edges(self) -> List[Dict[str, Any]]:
        edges: List[Dict[str, Any]] = []
        for index, source_id in enumerate(self.coarse_order):
            source = self.coarse_events[source_id]
            for target_id in self.coarse_order[index + 1:]:
                target = self.coarse_events[target_id]
                semantic = cosine_similarity(source.embedding, target.embedding)
                lexical = lexical_overlap(
                    f"{source.title} {source.summary}",
                    f"{target.title} {target.summary}",
                )
                temporal_gap = abs(source.updated_at - target.updated_at)
                temporal = math.exp(-temporal_gap / max(self.decay_half_life_seconds, 1.0))
                score = 0.62 * max(semantic, 0.0) + 0.22 * lexical + 0.16 * temporal
                if score >= 0.24:
                    edges.append(
                        {
                            "source": source_id,
                            "target": target_id,
                            "score": round(score, 4),
                            "shared_terms": shared_keywords(
                                f"{source.title} {source.summary}",
                                f"{target.title} {target.summary}",
                            ),
                        }
                    )
        edges.sort(key=lambda item: item["score"], reverse=True)
        return edges

    def _turn_snapshot(self, turn: RawTurn) -> Dict[str, Any]:
        return {
            "id": turn.id,
            "question": turn.question,
            "answer": turn.answer,
            "text": turn.text,
            "timestamp": turn.timestamp,
            "valid": turn.valid,
            "status": turn.status,
            "event_id": turn.event_id,
            "local_summary": turn.local_summary,
            "candidate_node_ids": list(turn.candidate_node_ids),
            "image_paths": list(turn.image_paths),
            "visual_description": turn.visual_description,
            "modality": turn.modality,
            "saliency_score": turn.saliency_score,
        }

    def _count_valid_turns(self) -> int:
        return sum(1 for turn in self.raw_turns.values() if turn.valid)

    def _repair_order_lists(self) -> None:
        for turn_id in sorted(self.raw_turns.keys(), key=lambda item: self.raw_turns[item].timestamp):
            if turn_id not in self.raw_turn_order:
                self.raw_turn_order.append(turn_id)
        for event_id in sorted(self.coarse_events.keys(), key=lambda item: self.coarse_events[item].created_at):
            if event_id not in self.coarse_order:
                self.coarse_order.append(event_id)

    def _repair_event_memberships(self) -> None:
        for event in self.coarse_events.values():
            event.turn_ids = [turn_id for turn_id in event.turn_ids if turn_id in self.raw_turns]
            event.entity_node_ids = [fine_id for fine_id in event.entity_node_ids if fine_id in self.entity_nodes]
            event.temporal_node_ids = [fine_id for fine_id in event.temporal_node_ids if fine_id in self.temporal_nodes]
            event.causal_node_ids = [fine_id for fine_id in event.causal_node_ids if fine_id in self.causal_nodes]
        for turn in self.raw_turns.values():
            if turn.event_id and turn.event_id in self.coarse_events:
                event = self.coarse_events[turn.event_id]
                if turn.id not in event.turn_ids:
                    event.turn_ids.append(turn.id)
        for node in self.entity_nodes.values():
            if node.event_id in self.coarse_events:
                event = self.coarse_events[node.event_id]
                if node.id not in event.entity_node_ids:
                    event.entity_node_ids.append(node.id)
        for node in self.temporal_nodes.values():
            if node.event_id in self.coarse_events:
                event = self.coarse_events[node.event_id]
                if node.id not in event.temporal_node_ids:
                    event.temporal_node_ids.append(node.id)
        for node in self.causal_nodes.values():
            if node.event_id in self.coarse_events:
                event = self.coarse_events[node.event_id]
                if node.id not in event.causal_node_ids:
                    event.causal_node_ids.append(node.id)

    def _persist_state(self) -> None:
        self.state["version"] = 4
        self.state["graph_initialized"] = self.graph_initialized
        self.state["cold_start_threshold"] = self.cold_start_threshold
        self.state["raw_turn_order"] = list(self.raw_turn_order)
        self.state["coarse_order"] = list(self.coarse_order)
        self.state["entity_edges"] = [edge.to_payload() for edge in self.entity_edges.values()]
        self.state["temporal_edges"] = [edge.to_payload() for edge in self.temporal_edges.values()]
        self.state["causal_edges"] = [edge.to_payload() for edge in self.causal_edges.values()]
        self.store.save_state(self.state)

        for turn_id in self.raw_turn_order:
            if turn_id in self.raw_turns:
                self.store.upsert_raw_turn(self.raw_turns[turn_id])
        for event_id in self.coarse_order:
            if event_id in self.coarse_events:
                self.store.upsert_coarse_event(self.coarse_events[event_id])
        for node in self.entity_nodes.values():
            self.store.upsert_entity_node(node)
        for node in self.temporal_nodes.values():
            self.store.upsert_temporal_node(node)
        for node in self.causal_nodes.values():
            self.store.upsert_causal_node(node)

    def _new_id(self, prefix: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex[:10]}"
