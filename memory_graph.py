"""Dual-granularity multimodal fire-scene memory graph.

The project started as a generic two-layer memory prototype.  This module
implements the patent-document version: coarse topic skeletons, fine-grained
fire-scene entities, multimodal evidence descriptors, streaming compression,
time-decay forgetting, and BFS retrieval traces.
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


FINE_NODE_TYPES = (
    "incident",
    "location",
    "person",
    "equipment",
    "hazard",
    "action",
    "status",
    "visual_evidence",
    "constraint",
    "reference",
)
VALID_TURN_STATUSES = {"buffered", "assigned"}
CRITICAL_KEYWORDS = (
    "起火",
    "火势",
    "烟雾",
    "爆炸",
    "坍塌",
    "受困",
    "伤亡",
    "高温",
    "易燃",
    "化学品",
    "疏散",
    "消防",
    "灭火",
    "救援",
    "出口",
    "楼梯",
    "电梯",
    "排烟",
    "水源",
)


def now_ts() -> float:
    return time.time()


def normalize_text(text: str) -> str:
    return " ".join((text or "").replace("\n", " ").split())


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
        "可以",
        "这个",
        "那个",
        "需要",
        "进行",
        "系统",
        "用户",
        "问题",
        "回答",
        "情况",
        "信息",
        "现场",
        "火灾",
    }
    overlap = sorted(set(tokenize(text_a)) & set(tokenize(text_b)))
    return [token for token in overlap if len(token) > 1 and token not in stop_tokens][:limit]


def deterministic_qdrant_id(value: str) -> str:
    return uuid.uuid5(uuid.NAMESPACE_DNS, value).hex


def decay_weight(
    last_seen: float,
    *,
    current_time: Optional[float] = None,
    half_life_seconds: float = 6 * 3600,
) -> float:
    """Exponential survival weight used by pruning and retrieval."""
    current_time = current_time or now_ts()
    age = max(0.0, current_time - last_seen)
    if half_life_seconds <= 0:
        return 1.0
    return float(math.exp(-math.log(2) * age / half_life_seconds))


def fire_importance(text: str, node_type: str = "status", saliency_score: float = 0.0) -> float:
    normalized = normalize_text(text)
    keyword_hits = sum(1 for item in CRITICAL_KEYWORDS if item in normalized)
    type_boost = {
        "incident": 0.22,
        "hazard": 0.20,
        "person": 0.16,
        "location": 0.12,
        "equipment": 0.10,
        "visual_evidence": 0.12,
        "action": 0.10,
        "constraint": 0.08,
    }.get(node_type, 0.04)
    score = 0.30 + min(0.30, keyword_hits * 0.06) + type_boost + min(0.22, saliency_score * 0.22)
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
    coarse_id: Optional[str] = None
    local_summary: str = ""
    candidate_node_ids: List[str] = field(default_factory=list)
    image_paths: List[str] = field(default_factory=list)
    visual_description: str = ""
    modality: str = "text"
    saliency_score: float = 0.0

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
            "coarse_id": self.coarse_id,
            "local_summary": self.local_summary,
            "candidate_node_ids": list(self.candidate_node_ids),
            "image_paths": list(self.image_paths),
            "visual_description": self.visual_description,
            "modality": self.modality,
            "saliency_score": self.saliency_score,
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
            coarse_id=payload.get("coarse_id"),
            local_summary=str(payload.get("local_summary", "")),
            candidate_node_ids=[str(item) for item in payload.get("candidate_node_ids", [])],
            image_paths=[str(item) for item in payload.get("image_paths", [])],
            visual_description=str(payload.get("visual_description", "")),
            modality=str(payload.get("modality", "text")),
            saliency_score=float(payload.get("saliency_score", 0.0)),
        )


@dataclass
class FineNode:
    id: str
    coarse_id: str
    node_type: str
    text: str
    normalized_text: str
    embedding: np.ndarray
    frequency: int
    first_seen: float
    last_seen: float
    source_turn_ids: List[str] = field(default_factory=list)
    modality: str = "text"
    visual_description: str = ""
    image_paths: List[str] = field(default_factory=list)
    importance: float = 0.5
    survival_weight: float = 1.0
    pinned: bool = False

    def to_payload(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "coarse_id": self.coarse_id,
            "node_type": self.node_type,
            "text": self.text,
            "normalized_text": self.normalized_text,
            "embedding": self.embedding.astype(float).tolist(),
            "frequency": self.frequency,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "source_turn_ids": list(self.source_turn_ids),
            "modality": self.modality,
            "visual_description": self.visual_description,
            "image_paths": list(self.image_paths),
            "importance": self.importance,
            "survival_weight": self.survival_weight,
            "pinned": self.pinned,
        }

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "FineNode":
        text = str(payload.get("text", ""))
        node_type = str(payload.get("node_type", "status"))
        return cls(
            id=str(payload["id"]),
            coarse_id=str(payload.get("coarse_id", "")),
            node_type=node_type,
            text=text,
            normalized_text=str(payload.get("normalized_text", text)),
            embedding=np.asarray(payload.get("embedding", []), dtype=np.float32),
            frequency=int(payload.get("frequency", 1)),
            first_seen=float(payload.get("first_seen", now_ts())),
            last_seen=float(payload.get("last_seen", now_ts())),
            source_turn_ids=[str(item) for item in payload.get("source_turn_ids", [])],
            modality=str(payload.get("modality", "text")),
            visual_description=str(payload.get("visual_description", "")),
            image_paths=[str(item) for item in payload.get("image_paths", [])],
            importance=float(payload.get("importance", fire_importance(text, node_type))),
            survival_weight=float(payload.get("survival_weight", 1.0)),
            pinned=bool(payload.get("pinned", False)),
        )


@dataclass
class FineEdge:
    id: str
    coarse_id: str
    source_id: str
    target_id: str
    edge_type: str
    weight: float
    frequency: int
    last_seen: float
    evidence_turn_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "FineEdge":
        return cls(
            id=str(payload["id"]),
            coarse_id=str(payload.get("coarse_id", "")),
            source_id=str(payload.get("source_id", "")),
            target_id=str(payload.get("target_id", "")),
            edge_type=str(payload.get("edge_type", "RELATES_TO")),
            weight=float(payload.get("weight", 1.0)),
            frequency=int(payload.get("frequency", 1)),
            last_seen=float(payload.get("last_seen", now_ts())),
            evidence_turn_ids=[str(item) for item in payload.get("evidence_turn_ids", [])],
        )


@dataclass
class CoarseTopicNode:
    id: str
    title: str
    summary: str
    embedding: np.ndarray
    created_at: float
    updated_at: float
    turn_ids: List[str] = field(default_factory=list)
    fine_node_ids: List[str] = field(default_factory=list)
    recent_hit_turn_ids: List[str] = field(default_factory=list)
    recent_consistency_scores: List[float] = field(default_factory=list)
    key_points: List[str] = field(default_factory=list)
    node_count: int = 0
    duplicate_ratio: float = 0.0
    edge_density: float = 0.0
    last_hit_at: float = 0.0
    reconstruct_count: int = 0
    visual_description: str = ""
    survival_weight: float = 1.0
    compression_ratio: float = 1.0

    def to_payload(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "summary": self.summary,
            "embedding": self.embedding.astype(float).tolist(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "turn_ids": list(self.turn_ids),
            "fine_node_ids": list(self.fine_node_ids),
            "recent_hit_turn_ids": list(self.recent_hit_turn_ids),
            "recent_consistency_scores": list(self.recent_consistency_scores),
            "key_points": list(self.key_points),
            "node_count": self.node_count,
            "duplicate_ratio": self.duplicate_ratio,
            "edge_density": self.edge_density,
            "last_hit_at": self.last_hit_at,
            "reconstruct_count": self.reconstruct_count,
            "visual_description": self.visual_description,
            "survival_weight": self.survival_weight,
            "compression_ratio": self.compression_ratio,
        }

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "CoarseTopicNode":
        return cls(
            id=str(payload["id"]),
            title=str(payload.get("title", "")),
            summary=str(payload.get("summary", "")),
            embedding=np.asarray(payload.get("embedding", []), dtype=np.float32),
            created_at=float(payload.get("created_at", now_ts())),
            updated_at=float(payload.get("updated_at", now_ts())),
            turn_ids=[str(item) for item in payload.get("turn_ids", [])],
            fine_node_ids=[str(item) for item in payload.get("fine_node_ids", [])],
            recent_hit_turn_ids=[str(item) for item in payload.get("recent_hit_turn_ids", [])],
            recent_consistency_scores=[
                float(item) for item in payload.get("recent_consistency_scores", [])
            ],
            key_points=[str(item) for item in payload.get("key_points", [])],
            node_count=int(payload.get("node_count", 0)),
            duplicate_ratio=float(payload.get("duplicate_ratio", 0.0)),
            edge_density=float(payload.get("edge_density", 0.0)),
            last_hit_at=float(payload.get("last_hit_at", 0.0)),
            reconstruct_count=int(payload.get("reconstruct_count", 0)),
            visual_description=str(payload.get("visual_description", "")),
            survival_weight=float(payload.get("survival_weight", 1.0)),
            compression_ratio=float(payload.get("compression_ratio", 1.0)),
        )


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
    candidates: Dict[str, List[str]]
    relations: List[ExtractedRelation]
    internal_facts: List[InternalFact] = field(default_factory=list)


class BaseSummarizer:
    def summarize_turn(self, question: str, answer: str, visual_description: str = "") -> str:
        raise NotImplementedError

    def build_topic_title(self, turn_texts: Sequence[str], previous_title: Optional[str] = None) -> str:
        raise NotImplementedError

    def build_topic_summary(
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


class HeuristicSummarizer(BaseSummarizer):
    def __init__(self, max_summary_chars: int = 240) -> None:
        self.max_summary_chars = max_summary_chars

    def summarize_turn(self, question: str, answer: str, visual_description: str = "") -> str:
        text = normalize_text(f"{question} {answer} {visual_description}")
        return self._clip(text, 120)

    def build_topic_title(self, turn_texts: Sequence[str], previous_title: Optional[str] = None) -> str:
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

    def build_topic_summary(
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
            question,
            answer,
            visual_description,
        )

    def build_topic_title(self, turn_texts: Sequence[str], previous_title: Optional[str] = None) -> str:
        prompt = (
            "为以下火灾场景记忆生成一个粗粒度话题标题。"
            "要求8到14个汉字，突出宏观态势，不要解释。\n"
            f"已有标题:{previous_title or '无'}\n片段:\n- " + "\n- ".join(turn_texts[:5])
        )
        return (self._complete_text(prompt, max_tokens=40) or self._fallback.build_topic_title(turn_texts))[:18]

    def build_topic_summary(
        self,
        turn_texts: Sequence[str],
        previous_summary: Optional[str] = None,
    ) -> str:
        prompt = (
            "根据以下同一粗粒度话题下的火灾记忆，生成结构化摘要。"
            "保留灾情演化、关键实体、空间位置、图像证据和处置结果；控制在160字以内。\n"
            f"已有摘要:{previous_summary or '无'}\n片段:\n- " + "\n- ".join(turn_texts[:8])
        )
        return self._complete_text(prompt, max_tokens=220) or self._fallback.build_topic_summary(
            turn_texts,
            previous_summary,
        )

    def fuse_summary(
        self,
        previous_summary: str,
        local_summary: str,
        recent_turn_texts: Sequence[str],
    ) -> str:
        prompt = (
            "请把已有粗粒度话题摘要与新的局部摘要融合为火灾态势记忆。"
            "删除重复和过时表述，保留高重要性事件、地点、人员、设备、危险源、视觉证据和处置动作；"
            "最终不超过160字。\n"
            f"已有摘要:{previous_summary or '无'}\n当前摘要:{local_summary}\n近期片段:\n- "
            + "\n- ".join(recent_turn_texts[-4:])
        )
        return self._complete_text(prompt, max_tokens=220) or self._fallback.fuse_summary(
            previous_summary,
            local_summary,
            recent_turn_texts,
        )

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


class ChatModel:
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

    def answer(self, question: str, context_blocks: Sequence[str]) -> str:
        if self._client is None:
            if context_blocks:
                return "根据已检索到的火场记忆：" + "；".join(context_blocks[:3])
            return "已记录该火灾现场信息，当前未配置大模型密钥，因此返回规则式占位回答。"
        context = "\n".join(f"{index + 1}. {item}" for index, item in enumerate(context_blocks[:6]))
        prompt = (
            "你是智慧消防记忆增强助手。请优先依据检索到的图谱记忆回答；"
            "如果上下文不足，请明确说明缺口，不要编造。\n"
            f"图谱记忆:\n{context or '无'}\n\n用户问题:{question}"
        )
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=700,
        )
        return normalize_text(response.choices[0].message.content or "")


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
    ) -> ExtractedTurn:
        if self._client is None:
            return self._heuristic_extract(question, answer, visual_description)
        prompt = (
            "从火灾现场交互中抽取细粒度原子信息和事实关系，只返回严格JSON。\n"
            "实体类型必须使用: incident, location, person, equipment, hazard, action, status, "
            "visual_evidence, constraint, reference。\n"
            "关系标签建议使用: RELATES_TO, LOCATED_IN, CAUSED_BY, TEMPORAL_NEXT, SUPPORTS, CONSTRAINS。\n"
            "只保留有决策价值的信息，例如起火点、火势、烟雾、温度、受困人员、消防设备、危险品、疏散路径、图像证据。\n"
            '输出格式: {"internal_facts":[{"text":"...","info_type":"location"}],'
            '"relations":[{"source_text":"...","target_text":"...","relation_label":"LOCATED_IN"}]}\n'
            f"输入:{question}\n回答:{answer}\n视觉描述:{visual_description or '无'}"
        )
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=700,
            )
            payload = json.loads(strip_json_fence(response.choices[0].message.content or ""))
            return self._payload_to_extracted(payload)
        except Exception:
            return self._heuristic_extract(question, answer, visual_description)

    def _payload_to_extracted(self, payload: Dict[str, Any]) -> ExtractedTurn:
        internal_facts = [
            InternalFact(
                text=normalize_text(item.get("text", "")),
                info_type=str(item.get("info_type", "status"))
                if item.get("info_type") in FINE_NODE_TYPES
                else "status",
            )
            for item in payload.get("internal_facts", [])
            if normalize_text(item.get("text", ""))
        ]
        candidates: Dict[str, List[str]] = {node_type: [] for node_type in FINE_NODE_TYPES}
        for fact in internal_facts:
            candidates[fact.info_type].append(fact.text)
        relations = [
            ExtractedRelation(
                source_text=normalize_text(item.get("source_text", "")),
                target_text=normalize_text(item.get("target_text", "")),
                relation_label=normalize_text(item.get("relation_label", "RELATES_TO")).upper(),
            )
            for item in payload.get("relations", [])
            if normalize_text(item.get("source_text", ""))
            and normalize_text(item.get("target_text", ""))
        ]
        return ExtractedTurn(candidates=candidates, relations=relations, internal_facts=internal_facts)

    def _heuristic_extract(self, question: str, answer: str = "", visual_description: str = "") -> ExtractedTurn:
        merged = normalize_text(f"{question} {answer} {visual_description}")
        chunks = [
            normalize_text(part)
            for part in re.split(r"[。；;，,\n]", merged)
            if normalize_text(part)
        ]
        candidates: Dict[str, List[str]] = {node_type: [] for node_type in FINE_NODE_TYPES}
        for chunk in chunks[:18]:
            node_type = self._classify_chunk(chunk)
            candidates[node_type].append(chunk[:48])
        if visual_description:
            candidates["visual_evidence"].append(visual_description[:80])
        deduped: Dict[str, List[str]] = {}
        for node_type, items in candidates.items():
            deduped[node_type] = list(dict.fromkeys([item for item in items if item]))[:8]

        facts = [
            InternalFact(text=item, info_type=node_type)
            for node_type, items in deduped.items()
            for item in items
        ]
        relations = self._infer_relations(facts)
        return ExtractedTurn(candidates=deduped, relations=relations, internal_facts=facts)

    def _classify_chunk(self, chunk: str) -> str:
        if any(key in chunk for key in ("东", "西", "南", "北", "楼", "层", "区", "室", "出口", "通道", "仓库", "坐标")):
            return "location"
        if any(key in chunk for key in ("受困", "人员", "消防员", "指挥员", "队员", "伤员")):
            return "person"
        if any(key in chunk for key in ("灭火器", "水枪", "喷淋", "排烟", "防火门", "摄像头", "热成像", "泵")):
            return "equipment"
        if any(key in chunk for key in ("易燃", "化学品", "燃气", "爆炸", "高温", "有毒", "坍塌")):
            return "hazard"
        if any(key in chunk for key in ("疏散", "灭火", "封控", "救援", "切断", "排烟", "转移", "部署")):
            return "action"
        if any(key in chunk for key in ("图像", "照片", "画面", "视频", "航拍", "红外", "热成像")):
            return "visual_evidence"
        if any(key in chunk for key in ("必须", "禁止", "限制", "无法", "不能", "需要")):
            return "constraint"
        if any(key in chunk for key in ("起火", "火势", "烟雾", "蔓延", "报警", "坍塌", "发生")):
            return "incident"
        return "status"

    def _infer_relations(self, facts: Sequence[InternalFact]) -> List[ExtractedRelation]:
        relations: List[ExtractedRelation] = []
        locations = [fact for fact in facts if fact.info_type == "location"]
        for fact in facts:
            if fact.info_type != "location" and locations:
                relations.append(
                    ExtractedRelation(
                        source_text=fact.text,
                        target_text=locations[0].text,
                        relation_label="LOCATED_IN",
                    )
                )
        hazards = [fact for fact in facts if fact.info_type == "hazard"]
        incidents = [fact for fact in facts if fact.info_type == "incident"]
        if hazards and incidents:
            relations.append(
                ExtractedRelation(
                    source_text=hazards[0].text,
                    target_text=incidents[0].text,
                    relation_label="RELATES_TO",
                )
            )
        return relations[:10]


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


class TwoLayerMemoryStore:
    def __init__(self, embedding_dim: int, storage_dir: str = "qdrant_data") -> None:
        self.embedding_dim = embedding_dim
        self.storage_dir = Path(storage_dir)
        if storage_dir != ":memory:":
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            self.state_path = self.storage_dir / "graph_state_v3.json"
            self.client = QdrantClient(path=str(self.storage_dir))
        else:
            self.state_path = None
            self.client = QdrantClient(":memory:")
        self.raw_turn_collection = "raw_turns_v3"
        self.coarse_collection = "coarse_topics_v3"
        self.fine_collection = "fine_nodes_v3"
        self._ensure_collection(self.raw_turn_collection)
        self._ensure_collection(self.coarse_collection)
        self._ensure_collection(self.fine_collection)
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
            "version": 3,
            "graph_initialized": False,
            "cold_start_threshold": cold_start_threshold,
            "raw_turn_order": [],
            "coarse_order": [],
            "fine_edges": [],
        }
        if self.state_path is None:
            return dict(self._memory_state or base)
        if not self.state_path.exists():
            return base
        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        for key, value in base.items():
            payload.setdefault(key, value)
        payload["version"] = 3
        return payload

    def save_state(self, payload: Dict[str, Any]) -> None:
        if self.state_path is None:
            self._memory_state = dict(payload)
            return
        self.state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_raw_turns(self) -> List[RawTurn]:
        return [RawTurn.from_payload(item) for item in self._load_records(self.raw_turn_collection)]

    def load_coarse_topics(self) -> List[CoarseTopicNode]:
        return [CoarseTopicNode.from_payload(item) for item in self._load_records(self.coarse_collection)]

    def load_fine_nodes(self) -> List[FineNode]:
        return [FineNode.from_payload(item) for item in self._load_records(self.fine_collection)]

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

    def upsert_coarse_topic(self, topic: CoarseTopicNode) -> None:
        self._upsert_vector_object(self.coarse_collection, topic.id, topic.embedding, topic.to_payload())

    def upsert_fine_node(self, node: FineNode) -> None:
        self._upsert_vector_object(self.fine_collection, node.id, node.embedding, node.to_payload())

    def delete_coarse_topic(self, coarse_id: str) -> None:
        self._delete_vector_object(self.coarse_collection, coarse_id)

    def delete_fine_node(self, fine_id: str) -> None:
        self._delete_vector_object(self.fine_collection, fine_id)

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


class GraphMemory:
    def __init__(
        self,
        cold_start_threshold: int = 6,
        initial_cluster_threshold: float = 0.24,
        new_topic_threshold: float = 0.34,
        fine_merge_threshold: float = 0.78,
        summarizer: Optional[BaseSummarizer] = None,
        extractor: Optional[TurnExtractor] = None,
        embedding_model: Optional[EmbeddingModel] = None,
        storage_dir: str = "qdrant_data",
        decay_half_life_seconds: float = 6 * 3600,
        prune_threshold: float = 0.16,
    ) -> None:
        self.cold_start_threshold = cold_start_threshold
        self.initial_cluster_threshold = initial_cluster_threshold
        self.new_topic_threshold = new_topic_threshold
        self.fine_merge_threshold = fine_merge_threshold
        self.decay_half_life_seconds = decay_half_life_seconds
        self.prune_threshold = prune_threshold
        self.summarizer = summarizer or QwenSummarizer()
        self.extractor = extractor or TurnExtractor()
        self.embedding_model = embedding_model or EmbeddingModel()
        self.store = TwoLayerMemoryStore(
            embedding_dim=self.embedding_model.dimension,
            storage_dir=storage_dir,
        )

        self.state = self.store.load_state(cold_start_threshold=self.cold_start_threshold)
        self.raw_turns: Dict[str, RawTurn] = {turn.id: turn for turn in self.store.load_raw_turns()}
        self.coarse_topics: Dict[str, CoarseTopicNode] = {
            topic.id: topic for topic in self.store.load_coarse_topics()
        }
        self.fine_nodes: Dict[str, FineNode] = {node.id: node for node in self.store.load_fine_nodes()}
        self.fine_edges: Dict[str, FineEdge] = {
            edge["id"]: FineEdge.from_dict(edge) for edge in self.state.get("fine_edges", [])
        }
        self.raw_turn_order: List[str] = [
            turn_id for turn_id in self.state.get("raw_turn_order", []) if turn_id in self.raw_turns
        ]
        self.coarse_order: List[str] = [
            coarse_id for coarse_id in self.state.get("coarse_order", []) if coarse_id in self.coarse_topics
        ]
        self.graph_initialized = bool(self.state.get("graph_initialized", False))
        self.recent_events: List[UpdateEvent] = []
        self._retrieval_trace: Dict[str, Any] = {"steps": []}

        self._repair_order_lists()
        self._repair_coarse_memberships()
        self._refresh_all_decay()

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
            valid_count = self._count_valid_turns()
            self.recent_events.append(
                UpdateEvent(
                    event_type="cold_start_progress",
                    message="冷启动缓冲区已更新，等待达到双粒度图谱初始化阈值。",
                    details={
                        "valid_turn_count": valid_count,
                        "cold_start_threshold": self.cold_start_threshold,
                    },
                )
            )
            if valid_count >= self.cold_start_threshold:
                self._initialize_from_buffer()
            self._persist_state()
            return turn

        self._incremental_update(turn.id)
        self._apply_forgetting()
        self._persist_state()
        return self.raw_turns[turn.id]

    def retrieve_context(self, query: str, k: Optional[int] = None) -> List[Any]:
        self._refresh_all_decay()
        self._retrieval_trace = {
            "steps": [],
            "visited_topics": [],
            "matched_fine_nodes": [],
            "pruned_nodes": [],
        }
        k = k or 6
        query_vector = self.embedding_model.encode(query)
        query_tokens = set(tokenize(query))
        candidates: List[Tuple[str, float, float, float, float]] = []
        for coarse_id in self.coarse_order:
            topic = self.coarse_topics[coarse_id]
            semantic = cosine_similarity(query_vector, topic.embedding)
            lexical = lexical_overlap(query, f"{topic.title} {topic.summary}")
            keyword = len(query_tokens & set(tokenize(f"{topic.title} {topic.summary}"))) / max(
                len(query_tokens),
                1,
            )
            combined = (
                0.48 * max(semantic, 0.0)
                + 0.20 * lexical
                + 0.17 * keyword
                + 0.15 * topic.survival_weight
            )
            candidates.append((coarse_id, semantic, lexical, keyword, combined))
        candidates.sort(key=lambda item: item[4], reverse=True)
        anchor = next((item for item in candidates if item[4] >= 0.14), None)
        if anchor is None:
            self._retrieval_trace["steps"].append(
                {"step": "anchor_not_found", "message": "未找到匹配的粗粒度火场话题。"}
            )
            return []

        anchor_id, anchor_semantic, anchor_lexical, anchor_keyword, anchor_score = anchor
        self._retrieval_trace["anchor"] = {
            "coarse_id": anchor_id,
            "title": self.coarse_topics[anchor_id].title,
            "semantic": round(anchor_semantic, 4),
            "lexical": round(anchor_lexical, 4),
            "keyword": round(anchor_keyword, 4),
            "survival_weight": round(self.coarse_topics[anchor_id].survival_weight, 4),
            "combined": round(anchor_score, 4),
        }

        visited: set[str] = set()
        queue: List[Tuple[str, int, float, str]] = [(anchor_id, 0, 1.0, anchor_id)]
        explored_topics: List[Dict[str, Any]] = []
        matched_fine: List[Dict[str, Any]] = []
        while queue and len(explored_topics) < 6:
            current_id, depth, path_score, previous_id = queue.pop(0)
            if current_id in visited or current_id not in self.coarse_topics:
                continue
            visited.add(current_id)
            topic = self.coarse_topics[current_id]
            topic_semantic = cosine_similarity(query_vector, topic.embedding)
            topic_lexical = lexical_overlap(query, f"{topic.title} {topic.summary}")
            topic_keyword = len(query_tokens & set(tokenize(f"{topic.title} {topic.summary}"))) / max(
                len(query_tokens),
                1,
            )
            topic_combined = (
                0.48 * max(topic_semantic, 0.0)
                + 0.20 * topic_lexical
                + 0.17 * topic_keyword
                + 0.15 * topic.survival_weight
            )
            prune_reason = None
            if depth > 0:
                edge_score = self._get_coarse_edge_score(previous_id, current_id)
                if edge_score < 0.26:
                    prune_reason = f"话题边权不足({round(edge_score, 4)})"
                elif anchor_semantic - topic_semantic > 0.25:
                    prune_reason = f"语义漂移({round(anchor_semantic - topic_semantic, 4)})"
                elif topic.survival_weight < self.prune_threshold:
                    prune_reason = f"话题生存权重过低({round(topic.survival_weight, 4)})"
                elif topic_combined < 0.10:
                    prune_reason = f"综合得分过低({round(topic_combined, 4)})"

            if prune_reason:
                self._retrieval_trace["steps"].append(
                    {
                        "step": "pruned_topic",
                        "coarse_id": current_id,
                        "title": topic.title,
                        "depth": depth,
                        "prune_reason": prune_reason,
                    }
                )
                continue

            topic.last_hit_at = now_ts()
            topic.recent_hit_turn_ids = topic.recent_hit_turn_ids[-7:] + [f"query_{int(topic.last_hit_at)}"]
            topic.survival_weight = min(1.0, topic.survival_weight + 0.05)
            self.store.upsert_coarse_topic(topic)

            explored_topics.append(
                {
                    "coarse_id": current_id,
                    "title": topic.title,
                    "depth": depth,
                    "semantic": round(topic_semantic, 4),
                    "lexical": round(topic_lexical, 4),
                    "keyword": round(topic_keyword, 4),
                    "survival_weight": round(topic.survival_weight, 4),
                    "combined": round(topic_combined, 4),
                    "path_score": round(path_score, 4),
                }
            )
            topic_fine = self._diffuse_fine_nodes(current_id, query_vector, query)
            matched_fine.extend(topic_fine)
            self._retrieval_trace["steps"].append(
                {
                    "step": "explore_topic",
                    "coarse_id": current_id,
                    "title": topic.title,
                    "depth": depth,
                    "fine_nodes_found": len(topic_fine),
                    "path_score": round(path_score, 4),
                }
            )

            if depth < 2:
                for target_id in self.coarse_order:
                    if target_id in visited or target_id == current_id:
                        continue
                    edge_score = self._get_coarse_edge_score(current_id, target_id)
                    if edge_score >= 0.22:
                        queue.append((target_id, depth + 1, path_score * edge_score, current_id))

        matched_fine.sort(key=lambda item: item.get("fine_score", 0.0), reverse=True)
        matched_fine = matched_fine[:k]
        self._retrieval_trace["visited_topics"] = explored_topics
        self._retrieval_trace["matched_fine_nodes"] = matched_fine
        if matched_fine:
            self._retrieval_trace["steps"].append(
                {
                    "step": "fine_retrieval",
                    "count": len(matched_fine),
                    "nodes": [
                        {
                            "node_id": item["node_id"],
                            "text": item["text"],
                            "score": item.get("fine_score", 0.0),
                            "survival_weight": item.get("survival_weight", 0.0),
                            "modality": item.get("modality", "text"),
                        }
                        for item in matched_fine
                    ],
                }
            )
        return [self._context_block(item) for item in matched_fine]

    def _diffuse_fine_nodes(self, coarse_id: str, query_vector: np.ndarray, query: str) -> List[Dict[str, Any]]:
        topic = self.coarse_topics[coarse_id]
        fine_nodes = [self.fine_nodes[fid] for fid in topic.fine_node_ids if fid in self.fine_nodes]
        if not fine_nodes:
            return []
        scored: List[Tuple[FineNode, float]] = []
        for node in fine_nodes:
            semantic = cosine_similarity(query_vector, node.embedding)
            lexical = lexical_overlap(query, f"{node.text} {node.visual_description}")
            node_score = (
                0.52 * max(semantic, 0.0)
                + 0.18 * lexical
                + 0.18 * node.importance
                + 0.12 * node.survival_weight
            )
            if not node.pinned and node.survival_weight < self.prune_threshold and node.importance < 0.62:
                self._retrieval_trace["pruned_nodes"].append(
                    {
                        "node_id": node.id,
                        "text": node.text,
                        "reason": f"生存权重过低({round(node.survival_weight, 4)})",
                    }
                )
                continue
            scored.append((node, node_score))
        scored.sort(key=lambda item: item[1], reverse=True)
        seeds = [node for node, score in scored if score > 0.16][:3] or [node for node, _ in scored[:1]]

        visited: set[str] = set()
        result: List[Dict[str, Any]] = []
        for seed in seeds:
            if seed.id in visited:
                continue
            visited.add(seed.id)
            result.append(self._fine_node_result(seed, query_vector, is_seed=True))
            seed.last_seen = now_ts()
            seed.frequency += 1
            seed.survival_weight = min(1.0, seed.survival_weight + 0.04)
            self.store.upsert_fine_node(seed)

        for seed in seeds:
            for edge in self._fine_edges_for_coarse(coarse_id):
                if edge.weight < 0.75:
                    continue
                neighbor_id = None
                if edge.source_id == seed.id:
                    neighbor_id = edge.target_id
                elif edge.target_id == seed.id:
                    neighbor_id = edge.source_id
                if not neighbor_id or neighbor_id in visited:
                    continue
                neighbor = self.fine_nodes.get(neighbor_id)
                if not neighbor:
                    continue
                if neighbor.survival_weight < self.prune_threshold and neighbor.importance < 0.62:
                    self._retrieval_trace["pruned_nodes"].append(
                        {
                            "node_id": neighbor.id,
                            "text": neighbor.text,
                            "reason": f"邻接节点生存权重过低({round(neighbor.survival_weight, 4)})",
                        }
                    )
                    continue
                visited.add(neighbor_id)
                item = self._fine_node_result(neighbor, query_vector, is_seed=False)
                item["edge_type"] = edge.edge_type
                item["edge_weight"] = round(edge.weight, 3)
                result.append(item)
        result.sort(key=lambda item: item.get("fine_score", 0.0), reverse=True)
        return result[:8]

    def _fine_node_result(self, node: FineNode, query_vector: np.ndarray, is_seed: bool) -> Dict[str, Any]:
        semantic = max(cosine_similarity(query_vector, node.embedding), 0.0)
        fine_score = 0.52 * semantic + 0.18 * node.importance + 0.18 * node.survival_weight + 0.12 * min(
            1.0,
            node.frequency / 5,
        )
        return {
            "node_id": node.id,
            "info_type": node.node_type,
            "text": node.text,
            "visual_description": node.visual_description,
            "image_paths": list(node.image_paths),
            "modality": node.modality,
            "fine_score": round(fine_score, 4),
            "frequency": node.frequency,
            "importance": round(node.importance, 4),
            "survival_weight": round(node.survival_weight, 4),
            "is_seed": is_seed,
        }

    def _context_block(self, item: Dict[str, Any]) -> str:
        pieces = [
            f"[{item.get('info_type', 'status')}] {item.get('text', '')}",
            f"重要性:{item.get('importance', 0)}",
            f"生存权重:{item.get('survival_weight', 0)}",
        ]
        if item.get("visual_description"):
            pieces.append(f"视觉证据:{item['visual_description']}")
        if item.get("image_paths"):
            pieces.append(f"图像路径:{', '.join(item['image_paths'])}")
        return "；".join(pieces)

    def get_retrieval_trace(self) -> Dict[str, Any]:
        return self._retrieval_trace

    def get_recent_events(self) -> List[Dict[str, Any]]:
        return [asdict(event) for event in self.recent_events]

    def graph_snapshot(self) -> Dict[str, Any]:
        self._refresh_all_decay()
        coarse_edges = self._compute_coarse_edges()
        coarse_nodes = []
        subgraphs: Dict[str, Any] = {}
        total_raw_chars = sum(len(self.raw_turns[turn_id].text) for turn_id in self.raw_turn_order if turn_id in self.raw_turns)
        total_memory_chars = 0
        for coarse_id in self.coarse_order:
            topic = self.coarse_topics[coarse_id]
            total_memory_chars += len(topic.summary)
            total_memory_chars += sum(
                len(self.fine_nodes[fid].text)
                for fid in topic.fine_node_ids
                if fid in self.fine_nodes
            )
            coarse_nodes.append(
                {
                    "id": topic.id,
                    "question": topic.title,
                    "answer": topic.summary,
                    "title": topic.title,
                    "summary": topic.summary,
                    "embedding": topic.embedding.astype(float).tolist(),
                    "created_at": topic.created_at,
                    "updated_at": topic.updated_at,
                    "turn_count": len(topic.turn_ids),
                    "source_ids": list(topic.turn_ids),
                    "access_count": len(topic.recent_hit_turn_ids),
                    "node_count": topic.node_count,
                    "duplicate_ratio": topic.duplicate_ratio,
                    "edge_density": topic.edge_density,
                    "recent_hit_count": len(topic.recent_hit_turn_ids),
                    "last_hit_at": topic.last_hit_at,
                    "reconstruct_count": topic.reconstruct_count,
                    "visual_description": topic.visual_description,
                    "survival_weight": topic.survival_weight,
                    "compression_ratio": topic.compression_ratio,
                }
            )
            subgraphs[coarse_id] = self._subgraph_snapshot(coarse_id)

        buffered_turns = [self._turn_snapshot(self.raw_turns[turn_id]) for turn_id in self.raw_turn_order if turn_id in self.raw_turns]
        valid_buffered = [
            turn for turn in buffered_turns if turn["valid"] and turn["status"] in VALID_TURN_STATUSES
        ]
        compression_ratio = (
            round(total_memory_chars / total_raw_chars, 4) if total_raw_chars else 1.0
        )
        return {
            "initialized": self.graph_initialized,
            "cold_start_threshold": self.cold_start_threshold,
            "valid_turn_count": self._count_valid_turns(),
            "buffered_turn_count": len(buffered_turns),
            "coarse_node_count": len(coarse_nodes),
            "coarse_edge_count": len(coarse_edges),
            "fine_node_count": len(self.fine_nodes),
            "fine_edge_count": len(self.fine_edges),
            "node_count": len(coarse_nodes),
            "edge_count": len(coarse_edges),
            "compression_ratio": compression_ratio,
            "estimated_space_saved": round(max(0.0, 1.0 - compression_ratio), 4),
            "nodes": coarse_nodes,
            "edges": coarse_edges,
            "raw_turns": buffered_turns,
            "valid_buffer": valid_buffered,
            "subgraphs": subgraphs,
        }

    def export_graph(self) -> nx.Graph:
        exported = nx.Graph()
        for coarse_id in self.coarse_order:
            topic = self.coarse_topics[coarse_id]
            exported.add_node(
                coarse_id,
                title=topic.title,
                summary=topic.summary,
                node_count=topic.node_count,
                turn_count=len(topic.turn_ids),
                duplicate_ratio=topic.duplicate_ratio,
                survival_weight=topic.survival_weight,
            )
        for edge in self._compute_coarse_edges():
            exported.add_edge(edge["source"], edge["target"], combined_score=edge["score"])
        return exported

    def delete_node(self, node_id: str) -> bool:
        if node_id not in self.coarse_topics:
            return False
        self._drop_coarse_topic(node_id)
        self._persist_state()
        self.recent_events = [
            UpdateEvent(
                event_type="coarse_deleted",
                message="已删除一个粗粒度话题及其细粒度子图。",
                details={"coarse_id": node_id},
            )
        ]
        return True

    def _initialize_from_buffer(self) -> None:
        valid_turns = [
            self.raw_turns[turn_id]
            for turn_id in self.raw_turn_order
            if turn_id in self.raw_turns and self.raw_turns[turn_id].valid
        ]
        clusters = self._cluster_turns(valid_turns)
        for cluster_turn_ids in clusters:
            turns = [self.raw_turns[turn_id] for turn_id in cluster_turn_ids]
            coarse = self._create_coarse_topic_from_turns(turns)
            previous_facts: Optional[List[InternalFact]] = None
            for turn in turns:
                self._assign_turn_to_coarse(turn, coarse.id)
                extracted = self.extractor.extract(
                    turn.question,
                    turn.answer,
                    visual_description=turn.visual_description,
                )
                previous_facts = extracted.internal_facts if extracted.internal_facts else previous_facts
                self._apply_turn_to_subgraph(turn.id, coarse.id, previous_facts=previous_facts)
            self._update_coarse_summary(coarse.id, turns[-1])
            self._refresh_coarse_metrics(coarse.id)
        self.graph_initialized = True
        self.recent_events.append(
            UpdateEvent(
                event_type="graph_initialized",
                message="冷启动完成，已构建火灾场景双粒度记忆图谱。",
                details={"coarse_topics": len(self.coarse_topics), "fine_nodes": len(self.fine_nodes)},
            )
        )

    def _incremental_update(self, turn_id: str) -> None:
        turn = self.raw_turns[turn_id]
        coarse_id, score = self._route_turn_to_coarse(turn)
        if coarse_id is None:
            coarse = self._create_coarse_topic_from_turns([turn])
            coarse_id = coarse.id
            self.recent_events.append(
                UpdateEvent(
                    event_type="coarse_created",
                    message="发现新的火场事态方向，初始化新的粗粒度话题。",
                    details={"coarse_id": coarse_id, "routing_score": round(score, 4)},
                )
            )
        else:
            self.recent_events.append(
                UpdateEvent(
                    event_type="coarse_selected",
                    message="新交互已挂载到最相关的粗粒度话题。",
                    details={"coarse_id": coarse_id, "routing_score": round(score, 4)},
                )
            )
        self._assign_turn_to_coarse(turn, coarse_id)
        self._apply_turn_to_subgraph(turn_id, coarse_id, previous_facts=None)
        self._update_coarse_summary(coarse_id, turn)
        self._refresh_coarse_metrics(coarse_id)
        if self._should_reconstruct(coarse_id):
            self._local_reconstruct(coarse_id)
        self._merge_similar_coarse_topics()

    def _cluster_turns(self, turns: Sequence[RawTurn]) -> List[List[str]]:
        clusters: List[Dict[str, Any]] = []
        for turn in turns:
            best_cluster_index = -1
            best_score = -1.0
            for index, cluster in enumerate(clusters):
                semantic = cosine_similarity(turn.embedding, cluster["centroid"])
                lexical = lexical_overlap(turn.text, cluster["text"])
                temporal = decay_weight(
                    max(cluster["last_seen"], turn.timestamp),
                    current_time=max(cluster["last_seen"], turn.timestamp),
                    half_life_seconds=self.decay_half_life_seconds,
                )
                score = 0.68 * max(semantic, 0.0) + 0.22 * lexical + 0.10 * temporal
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

    def _create_coarse_topic_from_turns(self, turns: Sequence[RawTurn]) -> CoarseTopicNode:
        turn_texts = [turn.text for turn in turns]
        title = self.summarizer.build_topic_title(turn_texts)
        summary = self.summarizer.build_topic_summary(turn_texts)
        visual_description = "；".join(
            dict.fromkeys([turn.visual_description for turn in turns if turn.visual_description])
        )
        embedding_source = f"{title}\n{summary}\n{visual_description}"
        coarse = CoarseTopicNode(
            id=self._new_id("topic"),
            title=title,
            summary=summary,
            embedding=self.embedding_model.encode_multimodal(embedding_source, visual_description),
            created_at=now_ts(),
            updated_at=now_ts(),
            visual_description=visual_description,
            survival_weight=1.0,
        )
        self.coarse_topics[coarse.id] = coarse
        self.coarse_order.append(coarse.id)
        self.store.upsert_coarse_topic(coarse)
        return coarse

    def _route_turn_to_coarse(self, turn: RawTurn) -> Tuple[Optional[str], float]:
        best_id: Optional[str] = None
        best_score = -1.0
        for coarse_id in self.coarse_order:
            topic = self.coarse_topics[coarse_id]
            semantic = cosine_similarity(turn.embedding, topic.embedding)
            text_overlap = lexical_overlap(turn.text, f"{topic.title} {topic.summary}")
            temporal = decay_weight(
                topic.updated_at,
                current_time=turn.timestamp,
                half_life_seconds=self.decay_half_life_seconds,
            )
            score = 0.62 * max(semantic, 0.0) + 0.22 * text_overlap + 0.16 * temporal
            if score > best_score:
                best_score = score
                best_id = coarse_id
        if best_id is None or best_score < self.new_topic_threshold:
            return None, best_score
        return best_id, best_score

    def _assign_turn_to_coarse(self, turn: RawTurn, coarse_id: str) -> None:
        topic = self.coarse_topics[coarse_id]
        turn.coarse_id = coarse_id
        turn.status = "assigned"
        if turn.id not in topic.turn_ids:
            topic.turn_ids.append(turn.id)
        if turn.visual_description:
            visuals = list(dict.fromkeys([topic.visual_description, turn.visual_description]))
            topic.visual_description = "；".join([item for item in visuals if item])[:260]
        topic.last_hit_at = now_ts()
        topic.updated_at = now_ts()
        topic.recent_hit_turn_ids = (topic.recent_hit_turn_ids + [turn.id])[-8:]
        consistency = cosine_similarity(turn.embedding, topic.embedding)
        topic.recent_consistency_scores = (topic.recent_consistency_scores + [consistency])[-8:]
        topic.survival_weight = min(1.0, topic.survival_weight + 0.05)
        self.store.upsert_raw_turn(turn)
        self.store.upsert_coarse_topic(topic)

    def _apply_turn_to_subgraph(
        self,
        turn_id: str,
        coarse_id: str,
        previous_facts: Optional[List[InternalFact]],
    ) -> None:
        turn = self.raw_turns[turn_id]
        extracted = self.extractor.extract(
            turn.question,
            turn.answer,
            visual_description=turn.visual_description,
        )
        candidate_mapping = self._upsert_fine_candidates(coarse_id, turn, extracted)
        turn.candidate_node_ids = list(dict.fromkeys(candidate_mapping.values()))
        self.store.upsert_raw_turn(turn)
        self._create_cooccur_edges(coarse_id, turn.id, turn.candidate_node_ids)
        self._create_relation_edges(coarse_id, turn.id, extracted.relations, candidate_mapping)

    def _upsert_fine_candidates(
        self,
        coarse_id: str,
        turn: RawTurn,
        extracted: ExtractedTurn,
    ) -> Dict[str, str]:
        topic = self.coarse_topics[coarse_id]
        text_to_node_id: Dict[str, str] = {}
        for node_type in FINE_NODE_TYPES:
            for text in extracted.candidates.get(node_type, []):
                node = self._merge_or_create_fine_node(coarse_id, node_type, text, turn)
                text_to_node_id[f"{node_type}:{normalize_text(text)}"] = node.id
                if node.id not in topic.fine_node_ids:
                    topic.fine_node_ids.append(node.id)
        self.store.upsert_coarse_topic(topic)
        return text_to_node_id

    def _merge_or_create_fine_node(
        self,
        coarse_id: str,
        node_type: str,
        text: str,
        turn: RawTurn,
    ) -> FineNode:
        normalized = normalize_text(text)
        visual = turn.visual_description if node_type == "visual_evidence" else ""
        embedding = self.embedding_model.encode_multimodal(normalized, visual, turn.saliency_score)
        best_match_id: Optional[str] = None
        best_score = -1.0
        for node in self._fine_nodes_for_coarse(coarse_id):
            if node.node_type != node_type:
                continue
            semantic = cosine_similarity(embedding, node.embedding)
            lexical = lexical_overlap(normalized, node.normalized_text)
            score = 0.70 * max(semantic, 0.0) + 0.30 * lexical
            if score > best_score:
                best_score = score
                best_match_id = node.id

        if best_match_id and best_score >= self.fine_merge_threshold:
            node = self.fine_nodes[best_match_id]
            node.frequency += 1
            node.last_seen = turn.timestamp
            node.importance = max(node.importance, fire_importance(normalized, node_type, turn.saliency_score))
            node.survival_weight = min(1.0, node.survival_weight + 0.08)
            if turn.id not in node.source_turn_ids:
                node.source_turn_ids.append(turn.id)
            node.image_paths = list(dict.fromkeys(node.image_paths + turn.image_paths))
            if visual:
                node.visual_description = "；".join(
                    dict.fromkeys([node.visual_description, visual])
                ).strip("；")[:260]
            self.store.upsert_fine_node(node)
            self.recent_events.append(
                UpdateEvent(
                    event_type="fine_node_merged",
                    message="细粒度原子信息已与同话题内相似节点融合。",
                    details={"coarse_id": coarse_id, "fine_node_id": node.id, "score": round(best_score, 4)},
                )
            )
            return node

        node = FineNode(
            id=self._new_id("fine"),
            coarse_id=coarse_id,
            node_type=node_type,
            text=normalized,
            normalized_text=normalized,
            embedding=embedding,
            frequency=1,
            first_seen=turn.timestamp,
            last_seen=turn.timestamp,
            source_turn_ids=[turn.id],
            modality=turn.modality if node_type == "visual_evidence" else "text",
            visual_description=visual,
            image_paths=list(turn.image_paths) if node_type == "visual_evidence" else [],
            importance=fire_importance(normalized, node_type, turn.saliency_score),
            survival_weight=1.0,
            pinned=any(keyword in normalized for keyword in ("起火原因", "受困", "伤亡", "爆炸", "坍塌")),
        )
        self.fine_nodes[node.id] = node
        self.store.upsert_fine_node(node)
        self.recent_events.append(
            UpdateEvent(
                event_type="fine_node_created",
                message="已在粗粒度话题内创建新的细粒度火场实体节点。",
                details={"coarse_id": coarse_id, "fine_node_id": node.id, "node_type": node_type},
            )
        )
        return node

    def _create_cooccur_edges(self, coarse_id: str, turn_id: str, node_ids: Sequence[str]) -> None:
        unique_ids = list(dict.fromkeys(node_ids))
        if len(unique_ids) < 2 or len(unique_ids) > 10:
            return
        for index, source_id in enumerate(unique_ids):
            for target_id in unique_ids[index + 1 :]:
                self._upsert_edge(coarse_id, source_id, target_id, "RELATES_TO", turn_id, 0.7)

    def _create_relation_edges(
        self,
        coarse_id: str,
        turn_id: str,
        relations: Sequence[ExtractedRelation],
        candidate_mapping: Dict[str, str],
    ) -> None:
        allowed = {"RELATES_TO", "LOCATED_IN", "CAUSED_BY", "TEMPORAL_NEXT", "SUPPORTS", "CONSTRAINS"}
        for relation in relations:
            source_id = self._candidate_node_lookup(candidate_mapping, relation.source_text)
            target_id = self._candidate_node_lookup(candidate_mapping, relation.target_text)
            if not source_id or not target_id or source_id == target_id:
                continue
            label = relation.relation_label if relation.relation_label in allowed else "RELATES_TO"
            self._upsert_edge(coarse_id, source_id, target_id, label, turn_id, 1.3)

    def _upsert_edge(
        self,
        coarse_id: str,
        source_id: str,
        target_id: str,
        edge_type: str,
        turn_id: str,
        weight_delta: float,
    ) -> None:
        edge_id = self._edge_id(coarse_id, source_id, target_id, edge_type)
        if edge_id in self.fine_edges:
            edge = self.fine_edges[edge_id]
            edge.weight = min(8.0, edge.weight + weight_delta)
            edge.frequency += 1
            edge.last_seen = now_ts()
            if turn_id not in edge.evidence_turn_ids:
                edge.evidence_turn_ids.append(turn_id)
        else:
            self.fine_edges[edge_id] = FineEdge(
                id=edge_id,
                coarse_id=coarse_id,
                source_id=source_id,
                target_id=target_id,
                edge_type=edge_type,
                weight=weight_delta,
                frequency=1,
                last_seen=now_ts(),
                evidence_turn_ids=[turn_id],
            )

    def _update_coarse_summary(self, coarse_id: str, turn: RawTurn) -> None:
        topic = self.coarse_topics[coarse_id]
        recent_turn_texts = [
            self.raw_turns[turn_id].text
            for turn_id in topic.turn_ids[-4:]
            if turn_id in self.raw_turns
        ]
        topic.summary = self.summarizer.fuse_summary(
            previous_summary=topic.summary,
            local_summary=turn.local_summary,
            recent_turn_texts=recent_turn_texts,
        )
        internal_texts = [node.text for node in self._fine_nodes_for_coarse(coarse_id)]
        embedding_source = f"{topic.title}\n{topic.summary}\n{' '.join(internal_texts[-8:])}"
        topic.embedding = self.embedding_model.encode_multimodal(embedding_source, topic.visual_description)
        topic.compression_ratio = self._topic_compression_ratio(coarse_id)
        topic.updated_at = now_ts()
        self.store.upsert_coarse_topic(topic)
        self.recent_events.append(
            UpdateEvent(
                event_type="coarse_summary_updated",
                message="粗粒度话题摘要已根据新增原子节点完成流式压缩更新。",
                details={"coarse_id": coarse_id, "compression_ratio": topic.compression_ratio},
            )
        )

    def _topic_compression_ratio(self, coarse_id: str) -> float:
        topic = self.coarse_topics[coarse_id]
        raw_chars = sum(len(self.raw_turns[tid].text) for tid in topic.turn_ids if tid in self.raw_turns)
        memory_chars = len(topic.summary) + sum(
            len(self.fine_nodes[fid].text) for fid in topic.fine_node_ids if fid in self.fine_nodes
        )
        return round(memory_chars / raw_chars, 4) if raw_chars else 1.0

    def _refresh_coarse_metrics(self, coarse_id: str) -> None:
        if coarse_id not in self.coarse_topics:
            return
        topic = self.coarse_topics[coarse_id]
        fine_nodes = self._fine_nodes_for_coarse(coarse_id)
        fine_edges = self._fine_edges_for_coarse(coarse_id)
        topic.node_count = len(fine_nodes)
        topic.duplicate_ratio = self._duplicate_ratio(fine_nodes)
        max_possible_edges = max(1, len(fine_nodes) * (len(fine_nodes) - 1) / 2)
        unique_pairs = {
            tuple(sorted([edge.source_id, edge.target_id]))
            for edge in fine_edges
            if edge.source_id != edge.target_id
        }
        topic.edge_density = min(1.0, len(unique_pairs) / max_possible_edges) if fine_nodes else 0.0
        topic.compression_ratio = self._topic_compression_ratio(coarse_id)
        topic.survival_weight = decay_weight(
            topic.updated_at,
            half_life_seconds=self.decay_half_life_seconds,
        )
        self.store.upsert_coarse_topic(topic)

    def _apply_forgetting(self) -> None:
        self._refresh_all_decay()
        removed = 0
        for node_id, node in list(self.fine_nodes.items()):
            if node.pinned or node.importance >= 0.70:
                continue
            if node.survival_weight >= self.prune_threshold:
                continue
            self._remove_fine_node(node_id)
            removed += 1
        if removed:
            self.recent_events.append(
                UpdateEvent(
                    event_type="fine_nodes_pruned",
                    message="低重要性、低生存权重的细粒度节点已从活跃图谱剪枝。",
                    details={"removed": removed, "threshold": self.prune_threshold},
                )
            )

    def _refresh_all_decay(self) -> None:
        current_time = now_ts()
        for node in self.fine_nodes.values():
            if node.pinned:
                node.survival_weight = 1.0
            else:
                time_weight = decay_weight(
                    node.last_seen,
                    current_time=current_time,
                    half_life_seconds=self.decay_half_life_seconds,
                )
                node.survival_weight = round(min(1.0, time_weight + 0.18 * node.importance), 4)
        for topic in self.coarse_topics.values():
            time_weight = decay_weight(
                topic.updated_at,
                current_time=current_time,
                half_life_seconds=self.decay_half_life_seconds,
            )
            topic.survival_weight = round(min(1.0, time_weight + 0.08 * min(1.0, topic.node_count / 10)), 4)

    def _should_reconstruct(self, coarse_id: str) -> bool:
        topic = self.coarse_topics[coarse_id]
        avg_consistency = (
            sum(topic.recent_consistency_scores) / len(topic.recent_consistency_scores)
            if topic.recent_consistency_scores
            else 1.0
        )
        return (
            topic.node_count >= 28
            or (topic.node_count >= 10 and topic.duplicate_ratio >= 0.24)
            or (len(topic.recent_consistency_scores) >= 5 and avg_consistency <= 0.24)
        )

    def _local_reconstruct(self, coarse_id: str) -> None:
        if coarse_id not in self.coarse_topics:
            return
        topic = self.coarse_topics[coarse_id]
        turn_ids = list(topic.turn_ids)
        if len(turn_ids) < 3:
            return
        drift_turn_ids = self._detect_drift_turns(coarse_id, turn_ids)
        if len(drift_turn_ids) >= 3 and len(drift_turn_ids) < len(turn_ids):
            stay_turn_ids = [turn_id for turn_id in turn_ids if turn_id not in drift_turn_ids]
            new_topic = self._create_coarse_topic_from_turns(
                [self.raw_turns[turn_id] for turn_id in drift_turn_ids if turn_id in self.raw_turns]
            )
            self._rebuild_single_coarse(coarse_id, stay_turn_ids)
            self._rebuild_single_coarse(new_topic.id, drift_turn_ids)
            self.recent_events.append(
                UpdateEvent(
                    event_type="coarse_split",
                    message="局部重构检测到语义漂移，并拆分出新的火场话题。",
                    details={"from_coarse_id": coarse_id, "to_coarse_id": new_topic.id},
                )
            )
        else:
            self._rebuild_single_coarse(coarse_id, turn_ids)
            self.recent_events.append(
                UpdateEvent(
                    event_type="coarse_rebuilt",
                    message="已对一个粗粒度话题执行局部重构和节点融合。",
                    details={"coarse_id": coarse_id},
                )
            )

    def _detect_drift_turns(self, coarse_id: str, turn_ids: Sequence[str]) -> List[str]:
        topic = self.coarse_topics[coarse_id]
        drift_turn_ids: List[str] = []
        for turn_id in turn_ids[-10:]:
            if turn_id not in self.raw_turns:
                continue
            turn = self.raw_turns[turn_id]
            semantic = cosine_similarity(turn.embedding, topic.embedding)
            lexical = lexical_overlap(turn.text, topic.summary)
            score = 0.70 * max(semantic, 0.0) + 0.30 * lexical
            if score < 0.20:
                drift_turn_ids.append(turn_id)
        return drift_turn_ids

    def _rebuild_single_coarse(self, coarse_id: str, turn_ids: Sequence[str]) -> None:
        if coarse_id not in self.coarse_topics:
            return
        topic = self.coarse_topics[coarse_id]
        for fine_id in list(topic.fine_node_ids):
            self._remove_fine_node(fine_id)
        topic.turn_ids = []
        topic.fine_node_ids = []
        topic.recent_hit_turn_ids = []
        topic.recent_consistency_scores = []
        topic.node_count = 0
        topic.duplicate_ratio = 0.0
        topic.edge_density = 0.0
        topic.reconstruct_count += 1

        turns = [self.raw_turns[turn_id] for turn_id in turn_ids if turn_id in self.raw_turns]
        topic.title = self.summarizer.build_topic_title([turn.text for turn in turns], previous_title=topic.title)
        topic.summary = self.summarizer.build_topic_summary([turn.text for turn in turns], topic.summary)
        topic.visual_description = "；".join(
            dict.fromkeys([turn.visual_description for turn in turns if turn.visual_description])
        )[:260]
        topic.embedding = self.embedding_model.encode_multimodal(
            f"{topic.title}\n{topic.summary}",
            topic.visual_description,
        )
        topic.updated_at = now_ts()

        for turn in turns:
            self._assign_turn_to_coarse(turn, coarse_id)
            self._apply_turn_to_subgraph(turn.id, coarse_id, previous_facts=None)
        self._refresh_coarse_metrics(coarse_id)

    def _detect_similar_coarse_topics(self, threshold: float = 0.56) -> List[Tuple[str, str, float]]:
        similar_pairs: List[Tuple[str, str, float]] = []
        for index, source_id in enumerate(self.coarse_order):
            source = self.coarse_topics[source_id]
            for target_id in self.coarse_order[index + 1 :]:
                target = self.coarse_topics[target_id]
                semantic = cosine_similarity(source.embedding, target.embedding)
                lexical = lexical_overlap(
                    f"{source.title} {source.summary}",
                    f"{target.title} {target.summary}",
                )
                score = 0.76 * max(semantic, 0.0) + 0.24 * lexical
                if score >= threshold:
                    similar_pairs.append((source_id, target_id, score))
        return similar_pairs

    def _merge_similar_coarse_topics(self) -> None:
        merged_ids: set[str] = set()
        for source_id, target_id, score in self._detect_similar_coarse_topics():
            if source_id in merged_ids or target_id in merged_ids:
                continue
            if source_id not in self.coarse_topics or target_id not in self.coarse_topics:
                continue
            self._merge_coarse_topics(source_id, target_id, score)
            merged_ids.add(target_id)

    def _merge_coarse_topics(self, source_id: str, target_id: str, score: float) -> None:
        source = self.coarse_topics[source_id]
        target = self.coarse_topics[target_id]
        for turn_id in target.turn_ids:
            if turn_id in self.raw_turns:
                turn = self.raw_turns[turn_id]
                turn.coarse_id = source_id
                if turn_id not in source.turn_ids:
                    source.turn_ids.append(turn_id)
                self.store.upsert_raw_turn(turn)
        for fine_id in list(target.fine_node_ids):
            node = self.fine_nodes.get(fine_id)
            if not node:
                continue
            node.coarse_id = source_id
            if fine_id not in source.fine_node_ids:
                source.fine_node_ids.append(fine_id)
            self.store.upsert_fine_node(node)
        source.summary = self.summarizer.fuse_summary(source.summary, target.summary, [])
        source.visual_description = "；".join(
            dict.fromkeys([source.visual_description, target.visual_description])
        ).strip("；")[:260]
        source.embedding = self.embedding_model.encode_multimodal(
            f"{source.title}\n{source.summary}",
            source.visual_description,
        )
        self._drop_coarse_topic(target_id, keep_turn_status=True)
        self._refresh_coarse_metrics(source_id)
        self.recent_events.append(
            UpdateEvent(
                event_type="coarse_merge",
                message="相似粗粒度话题已合并，减少宏观态势冗余。",
                details={"kept_id": source_id, "merged_id": target_id, "similarity": round(score, 4)},
            )
        )

    def _drop_coarse_topic(self, coarse_id: str, keep_turn_status: bool = False) -> None:
        if coarse_id not in self.coarse_topics:
            return
        topic = self.coarse_topics.pop(coarse_id)
        if coarse_id in self.coarse_order:
            self.coarse_order.remove(coarse_id)
        for fine_id in list(topic.fine_node_ids):
            if keep_turn_status and fine_id in self.fine_nodes:
                continue
            self._remove_fine_node(fine_id)
        for edge_id in list(self.fine_edges.keys()):
            if self.fine_edges[edge_id].coarse_id == coarse_id:
                self.fine_edges.pop(edge_id, None)
        if not keep_turn_status:
            for turn_id in topic.turn_ids:
                if turn_id in self.raw_turns:
                    turn = self.raw_turns[turn_id]
                    turn.coarse_id = None
                    turn.status = "buffered"
                    turn.candidate_node_ids = []
                    self.store.upsert_raw_turn(turn)
        self.store.delete_coarse_topic(coarse_id)

    def _remove_fine_node(self, fine_id: str) -> None:
        node = self.fine_nodes.pop(fine_id, None)
        if not node:
            return
        if node.coarse_id in self.coarse_topics:
            topic = self.coarse_topics[node.coarse_id]
            topic.fine_node_ids = [item for item in topic.fine_node_ids if item != fine_id]
            self.store.upsert_coarse_topic(topic)
        for edge_id in list(self.fine_edges.keys()):
            edge = self.fine_edges[edge_id]
            if edge.source_id == fine_id or edge.target_id == fine_id:
                self.fine_edges.pop(edge_id, None)
        self.store.delete_fine_node(fine_id)

    def _duplicate_ratio(self, fine_nodes: Sequence[FineNode]) -> float:
        if len(fine_nodes) < 2:
            return 0.0
        duplicate_hits = 0
        for index, source in enumerate(fine_nodes):
            for target in fine_nodes[index + 1 :]:
                if source.node_type != target.node_type:
                    continue
                semantic = cosine_similarity(source.embedding, target.embedding)
                lexical = lexical_overlap(source.text, target.text)
                if 0.70 * max(semantic, 0.0) + 0.30 * lexical >= self.fine_merge_threshold:
                    duplicate_hits += 1
        return round(duplicate_hits / max(len(fine_nodes), 1), 4)

    def _compute_coarse_edges(self) -> List[Dict[str, Any]]:
        edges: List[Dict[str, Any]] = []
        for index, source_id in enumerate(self.coarse_order):
            source = self.coarse_topics[source_id]
            for target_id in self.coarse_order[index + 1 :]:
                target = self.coarse_topics[target_id]
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

    def _subgraph_snapshot(self, coarse_id: str) -> Dict[str, Any]:
        fine_nodes = [
            {
                "id": node.id,
                "type": node.node_type,
                "text": node.text,
                "frequency": node.frequency,
                "first_seen": node.first_seen,
                "last_seen": node.last_seen,
                "source_turn_ids": list(node.source_turn_ids),
                "embedding": node.embedding.astype(float).tolist(),
                "modality": node.modality,
                "visual_description": node.visual_description,
                "image_paths": list(node.image_paths),
                "importance": node.importance,
                "survival_weight": node.survival_weight,
                "pinned": node.pinned,
            }
            for node in self._fine_nodes_for_coarse(coarse_id)
        ]
        fine_edges = [edge.to_dict() for edge in self._fine_edges_for_coarse(coarse_id)]
        fine_nodes.sort(key=lambda item: (item["type"], -item["importance"], -item["frequency"], item["text"]))
        fine_edges.sort(key=lambda item: (-item["weight"], item["edge_type"]))
        return {
            "node_count": len(fine_nodes),
            "edge_count": len(fine_edges),
            "nodes": fine_nodes,
            "edges": fine_edges,
        }

    def _turn_snapshot(self, turn: RawTurn) -> Dict[str, Any]:
        return {
            "id": turn.id,
            "question": turn.question,
            "answer": turn.answer,
            "text": turn.text,
            "timestamp": turn.timestamp,
            "valid": turn.valid,
            "status": turn.status,
            "coarse_id": turn.coarse_id,
            "local_summary": turn.local_summary,
            "candidate_node_ids": list(turn.candidate_node_ids),
            "image_paths": list(turn.image_paths),
            "visual_description": turn.visual_description,
            "modality": turn.modality,
            "saliency_score": turn.saliency_score,
        }

    def _get_coarse_edge_score(self, source_id: str, target_id: str) -> float:
        for edge in self._compute_coarse_edges():
            if (edge["source"] == source_id and edge["target"] == target_id) or (
                edge["source"] == target_id and edge["target"] == source_id
            ):
                return float(edge["score"])
        return 0.0

    def _candidate_node_lookup(self, mapping: Dict[str, str], text: str) -> Optional[str]:
        normalized = normalize_text(text)
        for node_type in FINE_NODE_TYPES:
            key = f"{node_type}:{normalized}"
            if key in mapping:
                return mapping[key]
        return None

    def _edge_id(self, coarse_id: str, source_id: str, target_id: str, edge_type: str) -> str:
        if edge_type == "RELATES_TO":
            a_id, b_id = sorted([source_id, target_id])
        else:
            a_id, b_id = source_id, target_id
        return f"{coarse_id}|{edge_type}|{a_id}|{b_id}"

    def _fine_nodes_for_coarse(self, coarse_id: str) -> List[FineNode]:
        return [node for node in self.fine_nodes.values() if node.coarse_id == coarse_id]

    def _fine_edges_for_coarse(self, coarse_id: str) -> List[FineEdge]:
        return [edge for edge in self.fine_edges.values() if edge.coarse_id == coarse_id]

    def _count_valid_turns(self) -> int:
        return sum(1 for turn in self.raw_turns.values() if turn.valid)

    def _repair_order_lists(self) -> None:
        for turn_id in sorted(self.raw_turns.keys(), key=lambda item: self.raw_turns[item].timestamp):
            if turn_id not in self.raw_turn_order:
                self.raw_turn_order.append(turn_id)
        for coarse_id in sorted(self.coarse_topics.keys(), key=lambda item: self.coarse_topics[item].created_at):
            if coarse_id not in self.coarse_order:
                self.coarse_order.append(coarse_id)

    def _repair_coarse_memberships(self) -> None:
        for topic in self.coarse_topics.values():
            topic.turn_ids = [turn_id for turn_id in topic.turn_ids if turn_id in self.raw_turns]
            topic.fine_node_ids = [fine_id for fine_id in topic.fine_node_ids if fine_id in self.fine_nodes]
        for turn in self.raw_turns.values():
            if turn.coarse_id and turn.coarse_id in self.coarse_topics:
                topic = self.coarse_topics[turn.coarse_id]
                if turn.id not in topic.turn_ids:
                    topic.turn_ids.append(turn.id)
        for node in self.fine_nodes.values():
            if node.coarse_id in self.coarse_topics:
                topic = self.coarse_topics[node.coarse_id]
                if node.id not in topic.fine_node_ids:
                    topic.fine_node_ids.append(node.id)

    def _persist_state(self) -> None:
        self.state["version"] = 3
        self.state["graph_initialized"] = self.graph_initialized
        self.state["cold_start_threshold"] = self.cold_start_threshold
        self.state["raw_turn_order"] = list(self.raw_turn_order)
        self.state["coarse_order"] = list(self.coarse_order)
        self.state["fine_edges"] = [edge.to_dict() for edge in self.fine_edges.values()]
        self.store.save_state(self.state)

        for turn_id in self.raw_turn_order:
            if turn_id in self.raw_turns:
                self.store.upsert_raw_turn(self.raw_turns[turn_id])
        for coarse_id in self.coarse_order:
            if coarse_id in self.coarse_topics:
                self.store.upsert_coarse_topic(self.coarse_topics[coarse_id])
        for node in self.fine_nodes.values():
            self.store.upsert_fine_node(node)

    def close(self) -> None:
        self.store.close()

    def _new_id(self, prefix: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex[:10]}"
