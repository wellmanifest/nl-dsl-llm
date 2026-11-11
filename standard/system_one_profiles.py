"""SSOT model profiles, candidate types, and logit scoring helpers for System One decisions."""

from __future__ import annotations

import functools
import json
import math
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote

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


@functools.lru_cache(maxsize=1)
def _load_schema_keys() -> tuple[frozenset[str], frozenset[str]]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    fd = os.open(SCHEMA_PATH, flags)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_size <= 0 or st.st_size > 1024 * 1024:
            raise ValueError("Invalid system-one-decision schema file")
        raw = os.read(fd, 1024 * 1024 + 1)
    finally:
        os.close(fd)
    if len(raw) > 1024 * 1024:
        raise ValueError("Oversized system-one-decision schema file")
    schema_doc = json.loads(raw.decode("utf-8"))
    return frozenset(schema_doc["required"]), frozenset(schema_doc["properties"].keys())


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _parse_contract_risk(raw: Any) -> tuple[int, bool]:
    if raw is None:
        return 1, True
    if type(raw) is int and 1 <= raw <= 3:
        return raw, True
    return 3, False


def _format_bound_dsl(action_id: str, bound_arguments: Mapping[str, Any]) -> str:
    base = action_id if "." in action_id else f"{action_id}.execute"
    if not bound_arguments:
        return base
    parts: list[str] = []
    for key in sorted(bound_arguments):
        raw_val = bound_arguments[key]
        if isinstance(raw_val, (dict, list)):
            formatted = json.dumps(
                raw_val, ensure_ascii=False, separators=(",", ":"), allow_nan=False
            )
        else:
            text_val = str(raw_val)
            needs_quote = not text_val or any(
                ch.isspace() or ch in ('"', "'", "\\", "=") or ord(ch) < 32 or ord(ch) == 127
                for ch in text_val
            )
            formatted = json.dumps(text_val, ensure_ascii=False) if needs_quote else text_val
        parts.append(f"{key}={formatted}")
    return f"{base} " + " ".join(parts)


def _format_action_uri(uri_base: str, bound_arguments: Mapping[str, Any]) -> str:
    if not bound_arguments or "?" in uri_base:
        return uri_base
    query = "&".join(
        f"{quote(str(k), safe='._~-')}={quote(str(bound_arguments[k]), safe='._~-')}"
        for k in sorted(bound_arguments)
    )
    return f"{uri_base}?{query}"


def validate_system_one_decision_envelope(envelope: Mapping[str, Any]) -> None:
    """Deterministic validator for wellmanifest.system-one-decision/v1 envelopes."""
    invalid_env = not isinstance(envelope, Mapping)
    if invalid_env:
        raise ValueError("Envelope must be a mapping")
    required_keys, allowed_keys = _load_schema_keys()
    if any(not isinstance(k, str) for k in envelope):
        raise ValueError("Envelope keys must be strings")
    actual_keys = set(envelope.keys())

    missing, extra = required_keys - actual_keys, actual_keys - allowed_keys
    if missing or extra:
        raise ValueError(
            f"Invalid envelope keys: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    if envelope.get("schema") != SCHEMA_ID:
        raise ValueError(f"Invalid schema: {envelope.get('schema')}")
    model_id = envelope.get("model_id")
    if not isinstance(model_id, str) or model_id not in SYSTEM_ONE_PROFILES:
        raise ValueError(f"Unsupported model_id: {model_id}")
    expected_family = SYSTEM_ONE_PROFILES[model_id].family
    if envelope.get("model_family") != expected_family:
        raise ValueError(
            f"model_family mismatch: {envelope.get('model_family')} != {expected_family}"
        )

    questions = envelope.get("questions")
    if (
        not isinstance(questions, list)
        or len(questions) != 3
        or not all(isinstance(q, Mapping) for q in questions)
    ):
        raise ValueError("questions must be a 3-element list of objects")
    q_choice, q_score, q_noul = questions
    if q_choice.get("name") != "uri_choice" or q_choice.get("type") != "choice":
        raise ValueError("questions[0] must be uri_choice (choice)")
    if (
        q_score.get("name") != "risk_score"
        or q_score.get("type") != "score"
        or q_score.get("options") != ["1", "2", "3"]
    ):
        raise ValueError("questions[1] must be risk_score (score) with options ['1', '2', '3']")
    if (
        q_noul.get("name") != "execution_allowed"
        or q_noul.get("type") != "noul"
        or q_noul.get("options") != ["yes", "no"]
    ):
        raise ValueError(
            "questions[2] must be execution_allowed (noul) with options ['yes', 'no']"
        )

    uri_options = q_choice.get("options")
    if (
        not isinstance(uri_options, list)
        or not (1 <= len(uri_options) <= 255)
        or not all(isinstance(u, str) for u in uri_options)
        or len(set(uri_options)) != len(uri_options)
    ):
        raise ValueError("uri_choice options must be 1..255 unique Action URIs")
    for uri in uri_options:
        if not ACTION_URI_RE.fullmatch(uri):
            raise ValueError(f"Invalid Action URI in options: {uri}")

    answers = envelope.get("answers")
    expected_q_names = {"uri_choice", "risk_score", "execution_allowed"}
    if not isinstance(answers, Mapping) or set(answers.keys()) != expected_q_names:
        raise ValueError("answers must contain exactly uri_choice, risk_score, execution_allowed")
    if (
        answers["uri_choice"] not in uri_options
        or envelope.get("selected_uri") != answers["uri_choice"]
    ):
        raise ValueError("selected_uri and answers.uri_choice must match an option in questions[0]")
    raw_risk = envelope.get("risk_score")
    if (
        answers["risk_score"] not in ("1", "2", "3")
        or type(raw_risk) is not int
        or raw_risk != int(answers["risk_score"])
    ):
        raise ValueError("risk_score must be an integer matching answers.risk_score in {1, 2, 3}")
    raw_allowed = envelope.get("execution_allowed")
    if (
        answers["execution_allowed"] not in ("yes", "no")
        or not isinstance(raw_allowed, bool)
        or raw_allowed != (answers["execution_allowed"] == "yes")
    ):
        raise ValueError("execution_allowed boolean must match answers.execution_allowed")

    confidence = envelope.get("confidence")
    if not _is_finite_number(confidence) or not (0.0 <= float(confidence) <= 1.0):
        raise ValueError(f"Invalid confidence: {confidence}")
    latency_ms = envelope.get("latency_ms")
    if not _is_finite_number(latency_ms) or float(latency_ms) < 0.0:
        raise ValueError(f"Invalid latency_ms: {latency_ms}")

    if "selected_dsl" in envelope and (
        not isinstance(envelope["selected_dsl"], str)
        or not CANONICAL_DSL_RE.fullmatch(envelope["selected_dsl"])
    ):
        raise ValueError(f"Invalid selected_dsl: {envelope['selected_dsl']}")

    probs = envelope.get("probabilities")
    if not isinstance(probs, Mapping) or set(probs.keys()) != expected_q_names:
        raise ValueError("probabilities must contain uri_choice, risk_score, execution_allowed")
    if not all(isinstance(probs[g], Mapping) for g in expected_q_names):
        raise ValueError("probability groups must be mappings")
    if set(probs["uri_choice"].keys()) != set(uri_options):
        raise ValueError("probabilities.uri_choice keys must match questions[0].options")
    if set(probs["risk_score"].keys()) != {"1", "2", "3"}:
        raise ValueError("probabilities.risk_score keys must be {'1', '2', '3'}")
    if set(probs["execution_allowed"].keys()) != {"yes", "no"}:
        raise ValueError("probabilities.execution_allowed keys must be {'yes', 'no'}")

    for group_name, dist in probs.items():
        vals = list(dist.values())
        if any(not _is_finite_number(v) or float(v) < 0.0 or float(v) > 1.0 for v in vals):
            raise ValueError(f"Invalid probability value in {group_name}")
        if abs(sum(float(v) for v in vals) - 1.0) > 1e-4:
            raise ValueError(f"Probabilities in {group_name} must sum to 1.0")
