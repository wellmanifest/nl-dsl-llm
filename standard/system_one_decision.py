"""
wellmanifest/nl-dsl-llm — System One / Jev-Compatible Typed Decision Layer (v1.0)

Implements single-pass typed decisions (choice, score, noul) over DockURI Action URIs
and canonical DSL commands conforming to wellmanifest/nohardcode, wellmanifest/ssot,
wellmanifest/dsl, wellmanifest/policy-dsl, and wellmanifest/nl-*.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

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
    _format_action_uri,
    _format_bound_dsl,
    _is_finite_number,
    _parse_contract_risk,
    _score_candidate,
    _softmax,
    _tokenize,
    select_decision_profile,
    validate_system_one_decision_envelope,
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
        invalid_risk = type(winner.risk_level) is not int
        if invalid_risk:
            raise ValueError("DecisionCandidate.risk_level must be an integer")
        risk = max(1, min(3, winner.risk_level))

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
            svc = (
                active_twin.active_services[0]
                if active_twin.active_services
                else "default-service"
            )
            nodes = [
                OptionNode(
                    id=f"{tpl['action_id']}:{svc}",
                    display_label=f"{tpl['verb_patterns'][0]} {svc}",
                    completion_text=f"{tpl['verb_patterns'][0]} {svc}",
                    action_id=tpl["action_id"],
                    resolved_uri=tpl["uri_pattern"],
                    bound_arguments={"target": svc, "env": active_twin.environment},
                    requires_confirmation=bool(
                        tpl["destructive"] and active_twin.environment == "prod"
                    ),
                    badge="PROD" if active_twin.environment == "prod" else "",
                )
                for tpl in engine.ACTION_TEMPLATES
            ]
        return self.decide(query, candidates_from_option_nodes(nodes))


def _compute_registry_digest(candidate_contracts: Sequence[Mapping[str, Any]]) -> str:
    entries = []
    for c in candidate_contracts:
        risk_val, _ = _parse_contract_risk(c.get("risk_level", 1))
        entries.append(
            {
                "uri": str(c.get("uri", "")),
                "dsl_command": str(c.get("dsl_command", c.get("action_id", ""))),
                "description": str(c.get("description", "")),
                "risk_level": risk_val,
            }
        )
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_selection_question(
    request: str,
    candidate_contracts: Sequence[Mapping[str, Any]],
    bounded_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build canonical System One / Jev choice question with op_0..op_N bindings and 3 escapes."""
    ctx = dict(bounded_context) if isinstance(bounded_context, Mapping) else {}
    bounded_contracts = [
        dict(contract) if isinstance(contract, Mapping) else {}
        for contract in candidate_contracts[:252]
    ]
    bindings: dict[str, dict[str, Any]] = {}
    options: list[str] = []
    for idx, contract in enumerate(bounded_contracts):
        op_id = f"op_{idx}"
        options.append(op_id)
        bindings[op_id] = contract
    options.extend(SPECIAL_OUTCOMES)
    return {
        "request": request if isinstance(request, str) else "",
        "registry_digest": _compute_registry_digest(bounded_contracts),
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
    """Canonical System One / Jev operation selector enforcing op_0..op_N bindings and 6 gates."""
    ctx = dict(bounded_context) if isinstance(bounded_context, Mapping) else {}
    question_spec = build_selection_question(request, candidate_contracts, ctx)
    bindings: dict[str, dict[str, Any]] = question_spec["bindings"]
    options: list[str] = question_spec["options"]
    expected_digest = question_spec["registry_digest"]
    bounded_contracts = list(bindings.values())

    q_lower = request.strip().lower() if isinstance(request, str) else ""
    q_tokens = _tokenize(q_lower)
    confidence_valid = True
    chosen_valid = True
    digest_valid = True

    if decision_provider is not None:
        provider_out = decision_provider(question_spec)
        raw_decision = dict(provider_out) if isinstance(provider_out, Mapping) else {}
        raw_chosen = raw_decision.get("chosen_option", "none")
        if isinstance(raw_chosen, str):
            chosen_option = raw_chosen
        else:
            chosen_option = "none"
            chosen_valid = False
        raw_conf = raw_decision.get("confidence", 0.0)
        if _is_finite_number(raw_conf) and 0.0 <= float(raw_conf) <= 1.0:
            confidence = float(raw_conf)
        else:
            confidence = 0.0
            confidence_valid = False
        raw_digest = raw_decision.get("registry_digest", expected_digest)
        if isinstance(raw_digest, str):
            provided_digest = raw_digest
        else:
            provided_digest = ""
            digest_valid = False
    else:
        if not bounded_contracts or not q_tokens:
            chosen_option, confidence = "none", 1.0
        elif any(sep in q_lower for sep in (" nastepnie ", " a potem ", " and then ", " then ")):
            chosen_option, confidence = "need_multi_step", 0.9
        else:
            decision_cands = []
            for idx, c in enumerate(bounded_contracts):
                risk_val, _ = _parse_contract_risk(c.get("risk_level", 1))
                raw_boost = c.get("prior_boost", 0.0)
                boost_val = float(raw_boost) if _is_finite_number(raw_boost) else 0.0
                raw_kw = c.get("keywords", ())
                kw_seq = raw_kw if isinstance(raw_kw, (list, tuple, set, frozenset)) else ()
                decision_cands.append(
                    DecisionCandidate(
                        uri=str(c.get("uri", f"proc://taskand.dev/op/{idx}/v1")),
                        dsl_command=str(c.get("dsl_command", c.get("action_id", f"op.exec_{idx}"))),
                        description=str(c.get("description", "")),
                        keywords=tuple(str(k) for k in kw_seq),
                        risk_level=risk_val,
                        prior_boost=boost_val,
                    )
                )
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
        else [
            c.get("uri") for c in bounded_contracts if isinstance(c.get("uri"), str) and c.get("uri")
        ]
    )
    acl_valid = isinstance(raw_allowed_uris, (list, tuple, set, frozenset)) and all(
        isinstance(item, str) for item in raw_allowed_uris
    )
    allowed_uris = set(raw_allowed_uris) if acl_valid else set()
    approved_high_risk = ctx.get("approved_high_risk") is True

    gate_1_membership = (
        chosen_valid
        and confidence_valid
        and chosen_option in options
        and (chosen_option in SPECIAL_OUTCOMES or chosen_option in bindings)
    )
    gate_2_digest = digest_valid and provided_digest == expected_digest
    gate_3_schema = True
    gate_4_acl = acl_valid
    gate_5_approval = True
    gate_6_effect = True

    if selected_contract is not None:
        risk_level, risk_ok = _parse_contract_risk(selected_contract.get("risk_level", 1))
        raw_req_args = selected_contract.get("required_args", ())
        req_args_ok = isinstance(raw_req_args, (list, tuple, set, frozenset)) and all(
            isinstance(arg, str) and bool(arg) for arg in raw_req_args
        )
        raw_args = selected_contract.get("arguments", ctx.get("arguments", {}))
        args_ok = isinstance(raw_args, Mapping)
        bound_args = dict(raw_args) if args_ok else {}
        required_args = tuple(raw_req_args) if req_args_ok else ()
        gate_3_schema = (
            risk_ok
            and req_args_ok
            and args_ok
            and all(arg in bound_args and bound_args[arg] not in (None, "") for arg in required_args)
        )
        raw_uri = selected_contract.get("uri")
        uri_val = raw_uri if isinstance(raw_uri, str) else ""
        gate_4_acl = (
            acl_valid
            and bool(uri_val)
            and (uri_val in allowed_uris if explicit_acl or allowed_uris else True)
        )
        raw_req_approval = selected_contract.get("requires_approval")
        requires_approval = (
            raw_req_approval if isinstance(raw_req_approval, bool) else (risk_level >= 3)
        )
        gate_5_approval = (not requires_approval) or approved_high_risk
        gate_6_effect = selected_contract.get("effect_verifiable", True) is True

    gates = {
        "gate_1_candidate_membership": gate_1_membership,
        "gate_2_registry_digest": gate_2_digest,
        "gate_3_argument_schema": gate_3_schema,
        "gate_4_resource_acl": gate_4_acl,
        "gate_5_approval_policy": gate_5_approval,
        "gate_6_effect_verification": gate_6_effect,
    }
    all_gates_passed = (
        all(gates.values())
        and chosen_option not in SPECIAL_OUTCOMES
        and selected_contract is not None
    )

    return {
        "chosen_option": chosen_option if gate_1_membership else "none",
        "selected_contract": selected_contract if all_gates_passed else None,
        "selected_uri": (
            selected_contract.get("uri") if (selected_contract and all_gates_passed) else None
        ),
        "confidence": round(confidence, 6),
        "registry_digest": expected_digest,
        "gates": gates,
        "execution_authorized": all_gates_passed,
    }
