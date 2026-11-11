"""
wellmanifest/nl-dsl-llm — System One / Jev-Compatible Typed Decision Layer (v1.0)

Implements single-pass typed decisions (choice, score, noul) over DockURI Action URIs
and canonical DSL commands conforming to wellmanifest/nohardcode, wellmanifest/ssot,
wellmanifest/dsl, wellmanifest/policy-dsl, and wellmanifest/nl-*.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any
from urllib.parse import quote

from standard.nl_dsl_llm import (
    DigitalTwinContext,
    DSLCommand,
    OptionNetworkEngine,
    OptionNode,
)
from standard.system_one_profiles import (
    ACTION_URI_RE,
    CANONICAL_DSL_RE,
    SCHEMA_ID,
    SCHEMA_PATH,
    SPECIAL_OUTCOMES,
    SYSTEM_ONE_PROFILES,
    CrossQuestionConstraint,
    DecisionCandidate,
    SystemOneModelProfile,
    _score_candidate,
    _softmax,
    _tokenize,
    select_decision_profile,
)

__all__ = [
    "ACTION_URI_RE",
    "CANONICAL_DSL_RE",
    "SCHEMA_ID",
    "SCHEMA_PATH",
    "SPECIAL_OUTCOMES",
    "SYSTEM_ONE_PROFILES",
    "CrossQuestionConstraint",
    "DecisionCandidate",
    "SystemOneDecisionEngine",
    "SystemOneModelProfile",
    "build_selection_question",
    "candidates_from_dsl_commands",
    "candidates_from_option_nodes",
    "select_decision_profile",
    "select_operation",
    "validate_system_one_decision_envelope",
]


@functools.lru_cache(maxsize=1)
def _load_schema_keys() -> tuple[frozenset[str], frozenset[str]]:
    schema_doc = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return frozenset(schema_doc["required"]), frozenset(schema_doc["properties"].keys())


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _format_bound_dsl(action_id: str, bound_arguments: Mapping[str, Any]) -> str:
    base = action_id if "." in action_id else f"{action_id}.execute"
    if not bound_arguments:
        return base
    parts: list[str] = []
    for key in sorted(bound_arguments):
        raw_val = bound_arguments[key]
        if isinstance(raw_val, (dict, list)):
            formatted = json.dumps(raw_val, ensure_ascii=False, separators=(",", ":"))
        else:
            text_val = str(raw_val)
            formatted = (
                json.dumps(text_val, ensure_ascii=False)
                if (not text_val or any(ch.isspace() or ch in ('"', "'") for ch in text_val))
                else text_val
            )
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


def candidates_from_option_nodes(nodes: Sequence[OptionNode]) -> list[DecisionCandidate]:
    """Convert OptionNetworkEngine OptionNodes into System One DecisionCandidates."""
    seen_uris: set[str] = set()
    candidates: list[DecisionCandidate] = []
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
    commands: Sequence[DSLCommand], namespace: str = "taskand.dev"
) -> list[DecisionCandidate]:
    """Project canonical DSLCommand objects into DockURI proc:// DecisionCandidates."""
    candidates: list[DecisionCandidate] = []
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
        profile: SystemOneModelProfile | None = None,
        constraint: CrossQuestionConstraint | None = None,
    ) -> None:
        self.profile = profile or SYSTEM_ONE_PROFILES["gliner2.5-decide-340m"]
        self.constraint = constraint or CrossQuestionConstraint()

    def decide(self, query: str, candidates: Sequence[DecisionCandidate]) -> dict[str, Any]:
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

        risk_probs_list = _softmax([2.4 if level == risk else -1.1 for level in (1, 2, 3)])
        risk_probs = {str(idx + 1): risk_probs_list[idx] for idx in range(3)}

        allowed = self.constraint.evaluate(risk)
        noul_probs = (
            {"yes": round(max(0.55, confidence), 6), "no": round(1.0 - max(0.55, confidence), 6)}
            if allowed
            else {"yes": 0.0, "no": 1.0}
        )

        elapsed_ms = round((time.perf_counter() - start) * 1000.0, 3)
        uri_options = [cand.uri for cand in candidates]
        uri_prob_map = {cand.uri: choice_probs[idx] for idx, cand in enumerate(candidates)}

        envelope: dict[str, Any] = {
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
        twin: DigitalTwinContext | None = None,
        option_engine: OptionNetworkEngine | None = None,
    ) -> dict[str, Any]:
        """Project DigitalTwinContext through OptionNetworkEngine and run a single-pass decision."""
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
        return self.decide(query, candidates_from_option_nodes(nodes))


def validate_system_one_decision_envelope(envelope: Mapping[str, Any]) -> None:
    """Deterministic validator for wellmanifest.system-one-decision/v1 envelopes."""
    invalid_env = not isinstance(envelope, Mapping)
    if invalid_env:
        raise ValueError("Envelope must be a mapping")
    required_keys, allowed_keys = _load_schema_keys()
    actual_keys = set(envelope.keys())

    missing = required_keys - actual_keys
    extra = actual_keys - allowed_keys
    if missing or extra:
        raise ValueError(f"Invalid envelope keys: missing={sorted(missing)}, extra={sorted(extra)}")
    if envelope.get("schema") != SCHEMA_ID:
        raise ValueError(f"Invalid schema: {envelope.get('schema')}")
    model_id = envelope.get("model_id")
    if not isinstance(model_id, str) or model_id not in SYSTEM_ONE_PROFILES:
        raise ValueError(f"Unsupported model_id: {model_id}")
    expected_family = SYSTEM_ONE_PROFILES[model_id].family
    if envelope.get("model_family") != expected_family:
        raise ValueError(f"model_family mismatch: {envelope.get('model_family')} != {expected_family}")

    questions = envelope.get("questions")
    if not isinstance(questions, list) or len(questions) != 3 or not all(isinstance(q, Mapping) for q in questions):
        raise ValueError("questions must be a 3-element list of objects")
    q_choice, q_score, q_noul = questions
    if q_choice.get("name") != "uri_choice" or q_choice.get("type") != "choice":
        raise ValueError("questions[0] must be uri_choice (choice)")
    if q_score.get("name") != "risk_score" or q_score.get("type") != "score" or q_score.get("options") != ["1", "2", "3"]:
        raise ValueError("questions[1] must be risk_score (score) with options ['1', '2', '3']")
    if q_noul.get("name") != "execution_allowed" or q_noul.get("type") != "noul" or q_noul.get("options") != ["yes", "no"]:
        raise ValueError("questions[2] must be execution_allowed (noul) with options ['yes', 'no']")

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
    if not isinstance(answers, Mapping) or set(answers.keys()) != {"uri_choice", "risk_score", "execution_allowed"}:
        raise ValueError("answers must contain exactly uri_choice, risk_score, execution_allowed")
    if answers["uri_choice"] not in uri_options or envelope.get("selected_uri") != answers["uri_choice"]:
        raise ValueError("selected_uri and answers.uri_choice must match an option in questions[0]")
    raw_risk = envelope.get("risk_score")
    if answers["risk_score"] not in ("1", "2", "3") or type(raw_risk) is not int or raw_risk != int(answers["risk_score"]):
        raise ValueError("risk_score must be an integer matching answers.risk_score in {1, 2, 3}")
    raw_allowed = envelope.get("execution_allowed")
    if answers["execution_allowed"] not in ("yes", "no") or not isinstance(raw_allowed, bool) or raw_allowed != (answers["execution_allowed"] == "yes"):
        raise ValueError("execution_allowed boolean must match answers.execution_allowed")

    confidence = envelope.get("confidence")
    if not _is_finite_number(confidence) or not (0.0 <= float(confidence) <= 1.0):
        raise ValueError(f"Invalid confidence: {confidence}")
    latency_ms = envelope.get("latency_ms")
    if not _is_finite_number(latency_ms) or float(latency_ms) < 0.0:
        raise ValueError(f"Invalid latency_ms: {latency_ms}")

    if "selected_dsl" in envelope and (
        not isinstance(envelope["selected_dsl"], str) or not CANONICAL_DSL_RE.fullmatch(envelope["selected_dsl"])
    ):
        raise ValueError(f"Invalid selected_dsl: {envelope['selected_dsl']}")

    probs = envelope.get("probabilities")
    if not isinstance(probs, Mapping) or set(probs.keys()) != {"uri_choice", "risk_score", "execution_allowed"}:
        raise ValueError("probabilities must contain uri_choice, risk_score, execution_allowed")
    if not all(isinstance(probs[g], Mapping) for g in ("uri_choice", "risk_score", "execution_allowed")):
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
    bounded_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build canonical System One / Jev choice question with op_0..op_N bindings and 3 escape outcomes."""
    ctx = dict(bounded_context or {})
    bindings: dict[str, dict[str, Any]] = {}
    options: list[str] = []
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
    bounded_context: Mapping[str, Any] | None = None,
    *,
    decision_provider: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Canonical System One / Jev operation selector enforcing op_0..op_N bindings and 6 safety gates."""
    ctx = dict(bounded_context or {})
    question_spec = build_selection_question(request, candidate_contracts, ctx)
    bindings: dict[str, dict[str, Any]] = question_spec["bindings"]
    options: list[str] = question_spec["options"]
    expected_digest = question_spec["registry_digest"]

    q_lower = request.strip().lower()
    q_tokens = _tokenize(q_lower)
    confidence_valid = True

    if decision_provider is not None:
        raw_decision = dict(decision_provider(question_spec))
        chosen_option = str(raw_decision.get("chosen_option", "none"))
        raw_conf = raw_decision.get("confidence", 0.0)
        if _is_finite_number(raw_conf) and 0.0 <= float(raw_conf) <= 1.0:
            confidence = float(raw_conf)
        else:
            confidence = 0.0
            confidence_valid = False
        provided_digest = str(raw_decision.get("registry_digest", expected_digest))
    else:
        if not candidate_contracts or not q_tokens:
            chosen_option, confidence = "none", 1.0
        elif any(sep in q_lower for sep in (" nastepnie ", " a potem ", " and then ", " then ")):
            chosen_option, confidence = "need_multi_step", 0.9
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
            if max(logits) <= -1.0:
                chosen_option, confidence = "none", 0.85
            else:
                probs = _softmax(logits)
                sorted_probs = sorted(probs, reverse=True)
                best_idx = max(range(len(probs)), key=lambda i: probs[i])
                if len(sorted_probs) > 1 and (sorted_probs[0] - sorted_probs[1]) < 0.05:
                    chosen_option, confidence = "need_clarification", sorted_probs[0]
                else:
                    chosen_option, confidence = f"op_{best_idx}", probs[best_idx]
        provided_digest = expected_digest

    selected_contract = bindings.get(chosen_option)
    explicit_acl = "allowed_uris" in ctx and ctx["allowed_uris"] is not None
    raw_allowed_uris = (
        ctx["allowed_uris"]
        if explicit_acl
        else [c.get("uri") for c in candidate_contracts if isinstance(c.get("uri"), str) and c.get("uri")]
    )
    acl_valid = isinstance(raw_allowed_uris, (list, tuple, set, frozenset)) and all(
        isinstance(item, str) for item in raw_allowed_uris
    )
    allowed_uris = set(raw_allowed_uris) if acl_valid else set()
    approved_high_risk = ctx.get("approved_high_risk") is True

    gate_1_membership = (
        confidence_valid
        and chosen_option in options
        and (chosen_option in SPECIAL_OUTCOMES or chosen_option in bindings)
    )
    gate_2_digest = provided_digest == expected_digest
    gate_3_schema = True
    gate_4_acl = acl_valid
    gate_5_approval = True
    gate_6_effect = True

    if selected_contract is not None:
        required_args = tuple(selected_contract.get("required_args", ()))
        bound_args = dict(selected_contract.get("arguments", ctx.get("arguments", {})))
        gate_3_schema = all(arg in bound_args and bound_args[arg] not in (None, "") for arg in required_args)
        uri_val = str(selected_contract.get("uri", ""))
        gate_4_acl = acl_valid and bool(uri_val) and (uri_val in allowed_uris if explicit_acl or allowed_uris else True)
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
    all_gates_passed = (
        all(gates.values()) and chosen_option not in SPECIAL_OUTCOMES and selected_contract is not None
    )

    return {
        "chosen_option": chosen_option if gate_1_membership else "none",
        "selected_contract": selected_contract if all_gates_passed else None,
        "selected_uri": selected_contract.get("uri") if (selected_contract and all_gates_passed) else None,
        "confidence": round(confidence, 6),
        "registry_digest": expected_digest,
        "gates": gates,
        "execution_authorized": all_gates_passed,
    }
