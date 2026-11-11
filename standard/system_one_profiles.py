"""SSOT model profiles, candidate types, and logit scoring helpers for System One decisions."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA_ID = "wellmanifest.system-one-decision/v1"
SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "system-one-decision.schema.json"
ACTION_URI_RE = re.compile(
    r"^[a-z][a-z0-9+.-]*://[A-Za-z0-9._-]+(?:/[A-Za-z0-9._:@-]+)+(?:[?][A-Za-z0-9._~%&=-]+)?$"
)
CANONICAL_DSL_RE = re.compile(r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*(?:\s+\S+)*$")
SPECIAL_OUTCOMES: tuple[str, ...] = ("none", "need_clarification", "need_multi_step")


@dataclass(frozen=True)
class SystemOneModelProfile:
    """Immutable SSOT descriptor for an open-source System One / Jev model family."""

    model_id: str
    family: str
    backbone: str
    license_spdx: str
    modalities: tuple[str, ...]
    question_types: tuple[str, ...]
    hardware_targets: tuple[str, ...]
    median_latency_ms: float
    max_candidates: int
    supports_cross_question_constraints: bool = True
    supports_state_prefetch: bool = False
    supports_offline_cpu: bool = True


SYSTEM_ONE_PROFILES: dict[str, SystemOneModelProfile] = {
    "clef-27b": SystemOneModelProfile(
        model_id="clef-27b",
        family="Clef",
        backbone="Qwen3.8-27B",
        license_spdx="Apache-2.0",
        modalities=("text", "json", "image", "video"),
        question_types=("choice", "score", "noul"),
        hardware_targets=("gpu", "workers-ai"),
        median_latency_ms=42.0,
        max_candidates=128,
        supports_offline_cpu=False,
    ),
    "clef-flash-9b": SystemOneModelProfile(
        model_id="clef-flash-9b",
        family="Clef-flash",
        backbone="Qwen3.5-9B",
        license_spdx="Apache-2.0",
        modalities=("text", "json", "image", "video"),
        question_types=("choice", "score", "noul"),
        hardware_targets=("gpu", "workers-ai"),
        median_latency_ms=18.0,
        max_candidates=128,
        supports_offline_cpu=False,
    ),
    "gliner2.5-decide-340m": SystemOneModelProfile(
        model_id="gliner2.5-decide-340m",
        family="GLiNER2.5-Decide",
        backbone="DeBERTa-v3-large-340M",
        license_spdx="Apache-2.0",
        modalities=("text", "json"),
        question_types=("choice", "score", "noul"),
        hardware_targets=("cpu", "gpu"),
        median_latency_ms=9.5,
        max_candidates=64,
        supports_cross_question_constraints=True,
    ),
    "laya-322m-multilingual": SystemOneModelProfile(
        model_id="laya-322m-multilingual",
        family="Laya",
        backbone="mmBERT-322M-non-autoregressive",
        license_spdx="Apache-2.0",
        modalities=("text", "json"),
        question_types=("choice", "score", "noul"),
        hardware_targets=("cpu", "gpu"),
        median_latency_ms=11.0,
        max_candidates=64,
    ),
    "laya-mlx-322m": SystemOneModelProfile(
        model_id="laya-mlx-322m",
        family="Laya-MLX",
        backbone="mmBERT-322M-MLX-CoreML",
        license_spdx="Apache-2.0",
        modalities=("text", "json"),
        question_types=("choice", "score", "noul"),
        hardware_targets=("apple-silicon", "mlx", "coreml"),
        median_latency_ms=7.4,
        max_candidates=64,
    ),
    "kev-4b": SystemOneModelProfile(
        model_id="kev-4b",
        family="Kev",
        backbone="Qwen3.5-4B-SystemOne",
        license_spdx="Apache-2.0",
        modalities=("text", "json"),
        question_types=("choice", "score", "noul"),
        hardware_targets=("cpu", "gpu"),
        median_latency_ms=16.5,
        max_candidates=128,
    ),
    "semif-qwen3.5-4b": SystemOneModelProfile(
        model_id="semif-qwen3.5-4b",
        family="SemIf",
        backbone="Qwen3.5-4B-DirectLogit",
        license_spdx="MIT",
        modalities=("text", "json"),
        question_types=("choice", "score", "noul"),
        hardware_targets=("cpu", "gpu", "webgpu"),
        median_latency_ms=14.0,
        max_candidates=128,
        supports_state_prefetch=True,
    ),
    "nanojev-0.6b": SystemOneModelProfile(
        model_id="nanojev-0.6b",
        family="NanoJev",
        backbone="Qwen3-0.6B",
        license_spdx="MIT",
        modalities=("text", "json"),
        question_types=("choice", "score", "noul"),
        hardware_targets=("cpu", "gpu", "edge"),
        median_latency_ms=6.2,
        max_candidates=255,
    ),
    "bespoke-nimble-9b": SystemOneModelProfile(
        model_id="bespoke-nimble-9b",
        family="Bespoke-Nimble",
        backbone="Qwen3.5-9B-8192ctx",
        license_spdx="Apache-2.0",
        modalities=("text", "json"),
        question_types=("choice", "score", "noul"),
        hardware_targets=("gpu",),
        median_latency_ms=21.0,
        max_candidates=255,
        supports_offline_cpu=False,
    ),
    "jevlike-scorer": SystemOneModelProfile(
        model_id="jevlike-scorer",
        family="jevlike",
        backbone="Qwen2.5-0.5B-CrossAttention",
        license_spdx="MIT",
        modalities=("text", "json"),
        question_types=("choice", "score", "noul"),
        hardware_targets=("cpu", "gpu"),
        median_latency_ms=4.8,
        max_candidates=128,
    ),
    "jevos-v3-cpu": SystemOneModelProfile(
        model_id="jevos-v3-cpu",
        family="jevos",
        backbone="OpenVINO-INT8-GGUF",
        license_spdx="Apache-2.0",
        modalities=("text", "json"),
        question_types=("choice", "score", "noul"),
        hardware_targets=("cpu", "openvino", "gguf"),
        median_latency_ms=8.1,
        max_candidates=64,
    ),
}


def select_decision_profile(
    *,
    modality: str = "text",
    hardware: str = "cpu",
    max_latency_ms: float = 25.0,
    candidate_count: int = 8,
    require_offline: bool = False,
    prefer_state_prefetch: bool = False,
) -> SystemOneModelProfile:
    """Select the optimal System One / Jev profile from the SSOT registry."""
    if modality in ("image", "video"):
        return (
            SYSTEM_ONE_PROFILES["clef-flash-9b"]
            if max_latency_ms <= 25.0
            else SYSTEM_ONE_PROFILES["clef-27b"]
        )
    if hardware in ("apple-silicon", "mlx", "coreml"):
        return SYSTEM_ONE_PROFILES["laya-mlx-322m"]
    if prefer_state_prefetch or hardware == "webgpu":
        return SYSTEM_ONE_PROFILES["semif-qwen3.5-4b"]
    if candidate_count > 128:
        return (
            SYSTEM_ONE_PROFILES["nanojev-0.6b"]
            if require_offline or hardware == "cpu"
            else SYSTEM_ONE_PROFILES["bespoke-nimble-9b"]
        )
    if hardware in ("openvino", "gguf"):
        return SYSTEM_ONE_PROFILES["jevos-v3-cpu"]

    eligible = [
        prof
        for prof in SYSTEM_ONE_PROFILES.values()
        if modality in prof.modalities
        and candidate_count <= prof.max_candidates
        and prof.median_latency_ms <= max_latency_ms
        and (not require_offline or prof.supports_offline_cpu)
    ]
    if not eligible:
        return SYSTEM_ONE_PROFILES["nanojev-0.6b"]
    return min(eligible, key=lambda p: p.median_latency_ms)


@dataclass(frozen=True)
class DecisionCandidate:
    """Candidate binding a DockURI Action URI and canonical DSL statement."""

    uri: str
    dsl_command: str
    description: str = ""
    keywords: tuple[str, ...] = field(default_factory=tuple)
    risk_level: int = 1
    prior_boost: float = 0.0


@dataclass(frozen=True)
class CrossQuestionConstraint:
    """Policy-DSL cross-question constraint linking risk_score to execution_allowed."""

    max_unconfirmed_risk: int = 2
    autonomy_mode: str = "confirm"

    def evaluate(self, risk_score: int) -> bool:
        if self.autonomy_mode == "autonomous":
            return True
        return risk_score <= self.max_unconfirmed_risk


def _tokenize(text: str) -> list[str]:
    return [tok for tok in re.split(r"[^a-z0-9ąćęłńóśźż_.-]+", text.lower()) if tok]


def _score_candidate(
    query_tokens: Sequence[str], query_lower: str, cand: DecisionCandidate
) -> float:
    uri_tokens = set(_tokenize(cand.uri))
    dsl_tokens = set(_tokenize(cand.dsl_command))
    desc_tokens = set(_tokenize(cand.description))
    kw_set = {kw.lower() for kw in cand.keywords}

    boost = cand.prior_boost if math.isfinite(cand.prior_boost) else 0.0
    logit = -1.2 + boost
    for qt in query_tokens:
        if qt in kw_set:
            logit += 2.2
        if qt in dsl_tokens:
            logit += 1.6
        if qt in uri_tokens:
            logit += 1.2
        if qt in desc_tokens:
            logit += 0.9
    for kw in kw_set:
        if " " in kw and kw in query_lower:
            logit += 2.8
    return logit


def _softmax(logits: Sequence[float]) -> list[float]:
    if not logits:
        return []
    clean_logits = [val if math.isfinite(val) else -100.0 for val in logits]
    max_logit = max(clean_logits)
    exps = [math.exp(val - max_logit) for val in clean_logits]
    total = sum(exps) or 1.0
    probs = [round(val / total, 6) for val in exps]
    diff = round(1.0 - sum(probs), 6)
    probs[0] = round(probs[0] + diff, 6)
    return probs
