"""
Tests for wellmanifest/nl-dsl-llm System One / Jev-compatible typed decision layer.
"""

from __future__ import annotations

import copy
import json

import pytest

from standard.nl_dsl_llm import DigitalTwinContext, DSLCommand
from standard.system_one_decision import (
    SCHEMA_ID,
    SCHEMA_PATH,
    SYSTEM_ONE_PROFILES,
    CrossQuestionConstraint,
    DecisionCandidate,
    SystemOneDecisionEngine,
    candidates_from_dsl_commands,
     select_decision_profile,
    validate_system_one_decision_envelope,
)


def test_system_one_profiles_cover_all_ten_model_families() -> None:
    expected_families = {
        "Clef",
        "Clef-flash",
        "GLiNER2.5-Decide",
        "Laya",
        "Laya-MLX",
        "Kev",
        "SemIf",
        "NanoJev",
        "Bespoke-Nimble",
        "jevlike",
        "jevos",
    }
    actual_families = {prof.family for prof in SYSTEM_ONE_PROFILES.values()}
    assert actual_families == expected_families
    for prof in SYSTEM_ONE_PROFILES.values():
        assert prof.question_types == ("choice", "score", "noul")
        assert prof.max_candidates >= 64


def test_select_decision_profile_routing() -> None:
    assert select_decision_profile(modality="image", max_latency_ms=20.0).model_id == "clef-flash-9b"
    assert select_decision_profile(modality="video", max_latency_ms=50.0).model_id == "clef-27b"
    assert select_decision_profile(hardware="apple-silicon").model_id == "laya-mlx-322m"
    assert select_decision_profile(prefer_state_prefetch=True).model_id == "semif-qwen3.5-4b"
    assert select_decision_profile(candidate_count=200, require_offline=True).model_id == "nanojev-0.6b"
    assert select_decision_profile(candidate_count=200, hardware="gpu").model_id == "bespoke-nimble-9b"
    assert select_decision_profile(hardware="openvino").model_id == "jevos-v3-cpu"
    assert select_decision_profile(hardware="cpu", max_latency_ms=10.0).model_id == "jevlike-scorer"


def test_decide_from_digital_twin_prioritizes_degraded_and_enforces_constraint() -> None:
    twin = DigitalTwinContext(
        environment="prod",
        active_services=["postgres", "redis-cache"],
        degraded_services=["redis-cache"],
        active_tickets=["ticket-001"],
    )
    engine = SystemOneDecisionEngine(
        profile=SYSTEM_ONE_PROFILES["gliner2.5-decide-340m"],
        constraint=CrossQuestionConstraint(max_unconfirmed_risk=2, autonomy_mode="confirm"),
    )
    env = engine.decide_from_twin("zrestartuj redis-cache", twin=twin)

    assert env["schema"] == SCHEMA_ID
    assert env["selected_uri"] == "proc://taskand.dev/service/restart/v1?env=prod&target=redis-cache"
    assert env["selected_dsl"] == "service.restart env=prod target=redis-cache"
    assert env["risk_score"] == 3
    assert env["execution_allowed"] is False
    assert env["answers"]["execution_allowed"] == "no"
    assert env["probabilities"]["execution_allowed"] == {"yes": 0.0, "no": 1.0}
    for group in ("uri_choice", "risk_score", "execution_allowed"):
        assert abs(sum(env["probabilities"][group].values()) - 1.0) <= 1e-4


def test_decide_from_dsl_commands_allows_low_risk_and_autonomous_high_risk() -> None:
    cmds = [
        DSLCommand(entity="ticket", operation="list", filters={"status": "OPEN"}),
        DSLCommand(entity="service", operation="restart", arguments={"target": "api-gateway"}),
    ]
    candidates = candidates_from_dsl_commands(cmds)

    safe_engine = SystemOneDecisionEngine(
        profile=SYSTEM_ONE_PROFILES["laya-322m-multilingual"],
        constraint=CrossQuestionConstraint(max_unconfirmed_risk=2, autonomy_mode="confirm"),
    )
    safe_env = safe_engine.decide("ticket list open", candidates)
    assert safe_env["selected_uri"] == "proc://taskand.dev/ticket/list/v1?status=OPEN"
    assert safe_env["selected_dsl"] == "ticket.list status=OPEN"
    assert safe_env["risk_score"] == 1
    assert safe_env["execution_allowed"] is True
    assert safe_env["answers"]["execution_allowed"] == "yes"

    auto_engine = SystemOneDecisionEngine(
        profile=SYSTEM_ONE_PROFILES["kev-4b"],
        constraint=CrossQuestionConstraint(max_unconfirmed_risk=2, autonomy_mode="autonomous"),
    )
    auto_env = auto_engine.decide("service restart api-gateway", candidates)
    assert auto_env["selected_uri"] == "proc://taskand.dev/service/restart/v1?target=api-gateway"
    assert auto_env["risk_score"] == 3
    assert auto_env["execution_allowed"] is True


def test_envelope_validation_rejects_cross_field_and_schema_violations() -> None:
    cands = [
        DecisionCandidate(
            uri="proc://taskand.dev/ticket/list/v1?status=OPEN",
            dsl_command="ticket.list status=OPEN",
            description="List open tickets",
            keywords=("ticket", "list", "open"),
            risk_level=1,
        )
    ]
    base = SystemOneDecisionEngine().decide("list open tickets", cands)
    validate_system_one_decision_envelope(base)

    bad_uri = copy.deepcopy(base)
    bad_uri["selected_uri"] = "proc://taskand.dev/other/op/v1"
    with pytest.raises(ValueError, match="selected_uri"):
        validate_system_one_decision_envelope(bad_uri)

    bad_prob = copy.deepcopy(base)
    bad_prob["probabilities"]["risk_score"]["1"] = 0.9
    with pytest.raises(ValueError, match="sum to 1.0"):
        validate_system_one_decision_envelope(bad_prob)

    bad_dsl = copy.deepcopy(base)
    bad_dsl["selected_dsl"] = "invalid_dsl_without_dot"
    with pytest.raises(ValueError, match="selected_dsl"):
        validate_system_one_decision_envelope(bad_dsl)


def test_jsonschema_draft202012_conformance_if_installed() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema_doc = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema_doc)
    validator.check_schema(schema_doc)

    cands = [
        DecisionCandidate(
            uri="proc://taskand.dev/service/status/v1?target=postgres",
            dsl_command="service.status target=postgres",
            description="Check postgres status",
            keywords=("status", "postgres"),
            risk_level=1,
        )
    ]
    envelope = SystemOneDecisionEngine().decide("status postgres", cands)
    validator.validate(envelope)
