"""
wellmanifest/nl-dsl-llm — System One / Jev-Compatible Typed Decision Layer (v1.0)

Implements single-pass typed decisions (choice, score, noul) over DockURI Action URIs
and canonical DSL commands conforming to wellmanifest/nohardcode, wellmanifest/ssot,
wellmanifest/dsl, wellmanifest/policy-dsl, and wellmanifest/nl-*.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from standard.nl_dsl_llm import DigitalTwinContext, DSLCommand, OptionNetworkEngine, OptionNode

SCHEMA_ID = "wellmanifest.system-one-decision/v1"
SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "system-one-decision.schema.json"
ACTION_URI_RE = re.compile(
    r"^[a-z][a-z0-9+.-]*://[A-Za-z0-9._-]+(?:/[A-Za-z0-9._:@-]+)+(?:[?][A-Za-z0-9._~%&=-]+)?$"
)
CANONICAL_DSL_RE = re.compile(r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*(?:\s+\S+)*$")


@dataclass(frozen=True)
class SystemOneModelProfile:
    """Immutable SSOT descriptor for an open-source System One / Jev model family."""

    model_id: str
    family: str
    backbone: str
    license_spdx: str
    modalities: Tuple[str, ...]
    question_types: Tuple[str, ...]
    hardware_targets: Tuple[str, ...]
    median_latency_ms: float
    max_candidates: int
    supports_cross_question_constraints: bool = True
    supports_state_prefetch: bool = False
    supports_offline_cpu: bool = True


SYSTEM_ONE_PROFILES: Dict[str, SystemOneModelProfile] = {
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
    """Selects the optimal System One / Jev profile from the SSOT registry."""
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
    keywords: Tuple[str, ...] = field(default_factory=tuple)
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


def _tokenize(text: str) -> List[str]:
    return [tok for tok in re.split(r"[^a-z0-9ąćęłńóśźż_.-]+", text.lower()) if tok]


def _score_candidate(query_tokens: Sequence[str], query_lower: str, cand: DecisionCandidate) -> float:
    uri_tokens = set(_tokenize(cand.uri))
    dsl_tokens = set(_tokenize(cand.dsl_command))
    desc_tokens = set(_tokenize(cand.description))
    kw_set = {kw.lower() for kw in cand.keywords}

    logit = -1.2 + cand.prior_boost
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


def _softmax(logits: Sequence[float]) -> List[float]:
    if not logits:
        return []
    max_logit = max(logits)
    exps = [math.exp(val - max_logit) for val in logits]
    total = sum(exps) or 1.0
    probs = [round(val / total, 6) for val in exps]
    diff = round(1.0 - sum(probs), 6)
    probs[0] = round(probs[0] + diff, 6)
    return probs


def _format_bound_dsl(action_id: str, bound_arguments: Mapping[str, Any]) -> str:
    base = action_id if "." in action_id else f"{action_id}.execute"
    if not bound_arguments:
        return base
    parts = [f"{k}={bound_arguments[k]}" for k in sorted(bound_arguments)]
    return f"{base} " + " ".join(parts)


def _format_action_uri(uri_base: str, bound_arguments: Mapping[str, Any]) -> str:
    if not bound_arguments or "?" in uri_base:
        return uri_base
    query = "&".join(f"{k}={bound_arguments[k]}" for k in sorted(bound_arguments))
    return f"{uri_base}?{query}"


def candidates_from_option_nodes(nodes: Sequence[OptionNode]) -> List[DecisionCandidate]:
    """Converts OptionNetworkEngine OptionNodes into System One DecisionCandidates."""
    seen_uris: set[str] = set()
    candidates: List[DecisionCandidate] = []
    for node in nodes:
        uri = _format_action_uri(node.resolved_uri, node.bound_arguments)
        if uri in seen_uris:
            continue
        seen_uris.add(uri)
        dsl_str = _format_bound_dsl(node.action_id, node.bound_arguments)
        kw = tuple(_tokenize(f"{node.completion_text} {node.display_label} {node.badge}"))
        risk = 3 if node.requires_confirmation else (2 if node.badge == "PROD" else 1)
        boost = 0.6 if node.badge == "DEGRADED" else 0.0
        candidates.append(
            DecisionCandidate(
                uri=uri,
                dsl_command=dsl_str,
                description=node.display_label,
                keywords=kw,
                risk_level=risk,
                prior_boost=boost,
            )
        )
    return candidates


def candidates_from_dsl_commands(
    commands: Sequence[DSLCommand],
    namespace: str = "taskand.dev",
) -> List[DecisionCandidate]:
    """Projects canonical DSLCommand objects into DockURI proc:// DecisionCandidates."""
    candidates: List[DecisionCandidate] = []
    seen_uris: set[str] = set()
    for cmd in commands:
        merged_params = {**cmd.arguments, **cmd.filters}
        uri_base = f"proc://{namespace}/{cmd.entity}/{cmd.operation}/v1"
        uri = _format_action_uri(uri_base, merged_params)
        if uri in seen_uris:
            continue
        seen_uris.add(uri)
        dsl_str = _format_bound_dsl(f"{cmd.entity}.{cmd.operation}", merged_params)
        risk = 3 if cmd.operation in ("delete", "stop", "restart") else 1
        candidates.append(
            DecisionCandidate(
                uri=uri,
                dsl_command=dsl_str,
                description=f"Execute {dsl_str}",
                keywords=tuple(_tokenize(dsl_str)),
                risk_level=risk,
            )
        )
    return candidates


class SystemOneDecisionEngine:
    """Single-pass System One / Jev decision engine over DockURI URIs and canonical DSL."""

    def __init__(
        self,
        profile: Optional[SystemOneModelProfile] = None,
        constraint: Optional[CrossQuestionConstraint] = None,
    ) -> None:
        self.profile = profile or SYSTEM_ONE_PROFILES["gliner2.5-decide-340m"]
        self.constraint = constraint or CrossQuestionConstraint()

    def decide(
        self,
        query: str,
        candidates: Sequence[DecisionCandidate],
    ) -> Dict[str, Any]:
        if not candidates:
            raise ValueError("candidates must contain at least one DecisionCandidate")
        start = time.perf_counter()
        q_lower = query.strip().lower()
        q_tokens = _tokenize(q_lower)

        logits = [_score_candidate(q_tokens, q_lower, cand) for cand in candidates]
        choice_probs = _softmax(logits)

        best_idx = max(range(len(candidates)), key=lambda idx: choice_probs[idx])
        winner = candidates[best_idx]
        confidence = choice_probs[best_idx]
        risk = max(1, min(3, int(winner.risk_level)))

        risk_logits = [
            2.4 if level == risk else -1.1
            for level in (1, 2, 3)
        ]
        risk_probs_list = _softmax(risk_logits)
        risk_probs = {str(idx + 1): risk_probs_list[idx] for idx in range(3)}

        allowed = self.constraint.evaluate(risk)
        if allowed:
            noul_probs = {"yes": round(max(0.55, confidence), 6), "no": round(1.0 - max(0.55, confidence), 6)}
        else:
            noul_probs = {"yes": 0.0, "no": 1.0}

        elapsed_ms = round((time.perf_counter() - start) * 1000.0, 3)
        uri_options = [cand.uri for cand in candidates]
        uri_prob_map = {cand.uri: choice_probs[idx] for idx, cand in enumerate(candidates)}

        envelope: Dict[str, Any] = {
            "schema": SCHEMA_ID,
            "model_id": self.profile.model_id,
            "model_family": self.profile.family,
            "questions": [
                {"name": "uri_choice", "type": "choice", "options": uri_options},
                {"name": "risk_score", "type": "score", "options": ["1", "2", "3"]},
                {"name": "execution_allowed", "type": "noul", "options": ["yes", "no"]},
            ],
            "answers": {
                "uri_choice": winner.uri,
                "risk_score": str(risk),
                "execution_allowed": "yes" if allowed else "no",
            },
            "probabilities": {
                "uri_choice": uri_prob_map,
                "risk_score": risk_probs,
                "execution_allowed": noul_probs,
            },
            "selected_uri": winner.uri,
            "selected_dsl": winner.dsl_command,
            "confidence": confidence,
            "risk_score": risk,
            "execution_allowed": allowed,
            "latency_ms": elapsed_ms,
        }
        validate_system_one_decision_envelope(envelope)
        return envelope

    def decide_from_twin(
        self,
        query: str,
        twin: Optional[DigitalTwinContext] = None,
        option_engine: Optional[OptionNetworkEngine] = None,
    ) -> Dict[str, Any]:
        """Projects DigitalTwinContext through OptionNetworkEngine and runs a single-pass decision."""
        engine = option_engine or OptionNetworkEngine()
        nodes = engine.suggest(query, twin=twin, max_suggestions=16)
        if not nodes:
            active_twin = twin or DigitalTwinContext()
            svc = active_twin.active_services[0] if active_twin.active_services else "default-service"
            nodes = [
                OptionNode(
                    id=f"{tpl['action_id']}:{svc}",
                    display_label=f"{tpl['verb_patterns'][0]} {svc}",
                    completion_text=f"{tpl['verb_patterns'][0]} {svc}",
                    action_id=tpl["action_id"],
                    resolved_uri=tpl["uri_pattern"],
                    bound_arguments={"target": svc, "env": active_twin.environment},
                    requires_confirmation=bool(tpl["destructive"] and active_twin.environment == "prod"),
                    badge="PROD" if active_twin.environment == "prod" else "",
                )
                for tpl in engine.ACTION_TEMPLATES
            ]
        candidates = candidates_from_option_nodes(nodes)
        return self.decide(query, candidates)


def validate_system_one_decision_envelope(envelope: Mapping[str, Any]) -> None:
    """Deterministic validator for wellmanifest.system-one-decision/v1 envelopes."""
    schema_doc = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    required_keys = set(schema_doc["required"])
    allowed_keys = set(schema_doc["properties"].keys())
    actual_keys = set(envelope.keys())

    missing = required_keys - actual_keys
    extra = actual_keys - allowed_keys
    if missing or extra:
        raise ValueError(f"Invalid envelope keys: missing={sorted(missing)}, extra={sorted(extra)}")
    if envelope.get("schema") != SCHEMA_ID:
        raise ValueError(f"Invalid schema: {envelope.get('schema')}")
    if envelope.get("model_id") not in SYSTEM_ONE_PROFILES:
        raise ValueError(f"Unsupported model_id: {envelope.get('model_id')}")
    expected_family = SYSTEM_ONE_PROFILES[envelope["model_id"]].family
    if envelope.get("model_family") != expected_family:
        raise ValueError(f"model_family mismatch: {envelope.get('model_family')} != {expected_family}")

    questions = envelope.get("questions")
    if not isinstance(questions, list) or len(questions) != 3:
        raise ValueError("questions must be a 3-element tuple list")
    q_choice, q_score, q_noul = questions
    if q_choice.get("name") != "uri_choice" or q_choice.get("type") != "choice":
        raise ValueError("questions[0] must be uri_choice (choice)")
    if q_score.get("name") != "risk_score" or q_score.get("type") != "score" or q_score.get("options") != ["1", "2", "3"]:
        raise ValueError("questions[1] must be risk_score (score) with options ['1', '2', '3']")
    if q_noul.get("name") != "execution_allowed" or q_noul.get("type") != "noul" or q_noul.get("options") != ["yes", "no"]:
        raise ValueError("questions[2] must be execution_allowed (noul) with options ['yes', 'no']")

    uri_options = q_choice.get("options")
    if not isinstance(uri_options, list) or not (1 <= len(uri_options) <= 255) or len(set(uri_options)) != len(uri_options):
        raise ValueError("uri_choice options must be 1..255 unique Action URIs")
    for uri in uri_options:
        if not isinstance(uri, str) or not ACTION_URI_RE.fullmatch(uri):
            raise ValueError(f"Invalid Action URI in options: {uri}")

    answers = envelope.get("answers", {})
    if set(answers.keys()) != {"uri_choice", "risk_score", "execution_allowed"}:
        raise ValueError("answers must contain exactly uri_choice, risk_score, execution_allowed")
    if answers["uri_choice"] not in uri_options or envelope.get("selected_uri") != answers["uri_choice"]:
        raise ValueError("selected_uri and answers.uri_choice must match an option in questions[0]")
    if answers["risk_score"] not in ("1", "2", "3") or envelope.get("risk_score") != int(answers["risk_score"]):
        raise ValueError("risk_score must match answers.risk_score in {1, 2, 3}")
    if answers["execution_allowed"] not in ("yes", "no") or envelope.get("execution_allowed") != (answers["execution_allowed"] == "yes"):
        raise ValueError("execution_allowed boolean must match answers.execution_allowed")

    if "selected_dsl" in envelope and not CANONICAL_DSL_RE.fullmatch(str(envelope["selected_dsl"])):
        raise ValueError(f"Invalid selected_dsl: {envelope['selected_dsl']}")

    probs = envelope.get("probabilities", {})
    if set(probs.keys()) != {"uri_choice", "risk_score", "execution_allowed"}:
        raise ValueError("probabilities must contain uri_choice, risk_score, execution_allowed")
    if set(probs["uri_choice"].keys()) != set(uri_options):
        raise ValueError("probabilities.uri_choice keys must match questions[0].options")
    if set(probs["risk_score"].keys()) != {"1", "2", "3"}:
        raise ValueError("probabilities.risk_score keys must be {'1', '2', '3'}")
    if set(probs["execution_allowed"].keys()) != {"yes", "no"}:
        raise ValueError("probabilities.execution_allowed keys must be {'yes', 'no'}")

    for group_name, dist in probs.items():
        vals = list(dist.values())
        if any(not isinstance(v, (int, float)) or isinstance(v, bool) or v < 0.0 or v > 1.0 for v in vals):
            raise ValueError(f"Invalid probability value in {group_name}")
        if abs(sum(vals) - 1.0) > 1e-4:
            raise ValueError(f"Probabilities in {group_name} must sum to 1.0")


SPECIAL_OUTCOMES: Tuple[str, ...] = ("none", "need_clarification", "need_multi_step")


def _compute_registry_digest(candidate_contracts: Sequence[Mapping[str, Any]]) -> str:
    canonical = json.dumps(
        [
            {
                "uri": str(c.get("uri", "")),
                "dsl_command": str(c.get("dsl_command", c.get("action_id", ""))),
                "description": str(c.get("description", "")),
                "risk_level": int(c.get("risk_level", 1)),
            }
            for c in candidate_contracts
        ],
        sort_keys=True,
        separators=(",", ":"),
    )
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_selection_question(
    request: str,
    candidate_contracts: Sequence[Mapping[str, Any]],
    bounded_context: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Builds the canonical System One / Jev choice question with op_0..op_N bindings and 3 escape outcomes."""
    ctx = dict(bounded_context or {})
    bindings: Dict[str, Dict[str, Any]] = {}
    options: List[str] = []
    for idx, contract in enumerate(candidate_contracts[:252]):
        op_id = f"op_{idx}"
        options.append(op_id)
        bindings[op_id] = dict(contract)
    options.extend(SPECIAL_OUTCOMES)
    return {
        "request": request,
        "registry_digest": _compute_registry_digest(candidate_contracts),
        "bounded_context": ctx,
        "options": options,
        "bindings": bindings,
        "questions": [
            {"name": "operation_choice", "type": "choice", "options": options},
            {"name": "read_only_intent", "type": "noul", "options": ["yes", "no"]},
            {"name": "risk_score", "type": "score", "options": ["1", "2", "3"]},
        ],
    }


def select_operation(
    request: str,
    candidate_contracts: Sequence[Mapping[str, Any]],
    bounded_context: Optional[Mapping[str, Any]] = None,
    *,
    decision_provider: Optional[Callable[[Mapping[str, Any]], Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Canonical System One / Jev operation selector enforcing op_0..op_N bindings and 6 safety gates."""
    ctx = dict(bounded_context or {})
    question_spec = build_selection_question(request, candidate_contracts, ctx)
    bindings: Dict[str, Dict[str, Any]] = question_spec["bindings"]
    options: List[str] = question_spec["options"]
    expected_digest = question_spec["registry_digest"]

    q_lower = request.strip().lower()
    q_tokens = _tokenize(q_lower)

    if decision_provider is not None:
        raw_decision = dict(decision_provider(question_spec))
        chosen_option = str(raw_decision.get("chosen_option", "none"))
        confidence = float(raw_decision.get("confidence", 0.0))
        provided_digest = str(raw_decision.get("registry_digest", expected_digest))
    else:
        if not candidate_contracts or not q_tokens:
            chosen_option = "none"
            confidence = 1.0
        elif any(sep in q_lower for sep in (" nastepnie ", " a potem ", " and then ", " then ")):
            chosen_option = "need_multi_step"
            confidence = 0.9
        else:
            decision_cands = [
                DecisionCandidate(
                    uri=str(c.get("uri", f"proc://taskand.dev/op/{idx}/v1")),
                    dsl_command=str(c.get("dsl_command", c.get("action_id", f"op.exec_{idx}"))),
                    description=str(c.get("description", "")),
                    keywords=tuple(str(k) for k in c.get("keywords", ())),
                    risk_level=int(c.get("risk_level", 1)),
                    prior_boost=float(c.get("prior_boost", 0.0)),
                )
                for idx, c in enumerate(candidate_contracts[:252])
            ]
            logits = [_score_candidate(q_tokens, q_lower, dc) for dc in decision_cands]
            max_logit = max(logits)
            if max_logit <= -1.0:
                chosen_option = "none"
                confidence = 0.85
            else:
                probs = _softmax(logits)
                sorted_probs = sorted(probs, reverse=True)
                best_idx = max(range(len(probs)), key=lambda i: probs[i])
                if len(sorted_probs) > 1 and (sorted_probs[0] - sorted_probs[1]) < 0.05:
                    chosen_option = "need_clarification"
                    confidence = sorted_probs[0]
                else:
                    chosen_option = f"op_{best_idx}"
                    confidence = probs[best_idx]
        provided_digest = expected_digest

    selected_contract = bindings.get(chosen_option)
    allowed_uris = set(ctx.get("allowed_uris", [c.get("uri") for c in candidate_contracts if c.get("uri")]))
    approved_high_risk = bool(ctx.get("approved_high_risk", False))

    # 6 deterministic safety gates
    gate_1_membership = chosen_option in options and (
        chosen_option in SPECIAL_OUTCOMES or chosen_option in bindings
    )
    gate_2_digest = provided_digest == expected_digest
    gate_3_schema = True
    gate_4_acl = True
    gate_5_approval = True
    gate_6_effect = True

    if selected_contract is not None:
        required_args = tuple(selected_contract.get("required_args", ()))
        bound_args = dict(selected_contract.get("arguments", ctx.get("arguments", {})))
        gate_3_schema = all(arg in bound_args and bound_args[arg] not in (None, "") for arg in required_args)
        uri_val = str(selected_contract.get("uri", ""))
        gate_4_acl = bool(uri_val) and (not allowed_uris or uri_val in allowed_uris)
        risk_level = int(selected_contract.get("risk_level", 1))
        requires_approval = bool(selected_contract.get("requires_approval", risk_level >= 3))
        gate_5_approval = (not requires_approval) or approved_high_risk
        gate_6_effect = bool(selected_contract.get("effect_verifiable", True))

    gates = {
        "gate_1_candidate_membership": gate_1_membership,
        "gate_2_registry_digest": gate_2_digest,
        "gate_3_argument_schema": gate_3_schema,
        "gate_4_resource_acl": gate_4_acl,
        "gate_5_approval_policy": gate_5_approval,
        "gate_6_effect_verification": gate_6_effect,
    }
    all_gates_passed = all(gates.values()) and chosen_option not in SPECIAL_OUTCOMES and selected_contract is not None

    return {
        "chosen_option": chosen_option if gate_1_membership else "none",
        "selected_contract": selected_contract if all_gates_passed else None,
        "selected_uri": selected_contract.get("uri") if (selected_contract and all_gates_passed) else None,
        "confidence": round(confidence, 6),
        "registry_digest": expected_digest,
        "gates": gates,
        "execution_authorized": all_gates_passed,
    }
