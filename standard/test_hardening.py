"""Adversarial hardening tests for NL-DSL-LLM standard modules."""

from __future__ import annotations

import asyncio
import base64
import copy
import struct
from pathlib import Path

import pytest

from standard.nl_dsl_llm import (
    DSLCommand,
    DSLExecutor,
    InterfaceAdapters,
    NLDSLLLMBridge,
    SemanticCache,
)
from standard.nl_dsl_llm_ws import (
    NLStreamFrame,
    NLStreamServer,
    StreamFrameType,
    decode_ws_frame,
    encode_ws_frame,
)
from standard.system_one_decision import (
    SystemOneDecisionEngine,
    candidates_from_dsl_commands,
    select_operation,
    validate_system_one_decision_envelope,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_all_repository_python_modules_stay_under_500_lines() -> None:
    checked = 0
    for py_path in sorted(REPO_ROOT.rglob("*.py")):
        rel_parts = py_path.relative_to(REPO_ROOT).parts
        if any(part in {".git", ".governance", ".worktrees", "__pycache__"} for part in rel_parts):
            continue
        lines = len(py_path.read_text(encoding="utf-8").splitlines())
        assert lines < 500, f"{py_path.relative_to(REPO_ROOT)} has {lines} lines (>= 500)"
        checked += 1
    assert checked >= 15


def test_dsl_command_quoted_whitespace_roundtrip_and_malformed_token_rejection() -> None:
    cmd = DSLCommand.from_text('ticket.create title="Fix login regression" filter:status="IN PROGRESS"')
    assert cmd.entity == "ticket"
    assert cmd.operation == "create"
    assert cmd.arguments["title"] == "Fix login regression"
    assert cmd.filters["status"] == "IN PROGRESS"

    roundtripped = DSLCommand.from_text(cmd.to_canonical_string())
    assert roundtripped.arguments == cmd.arguments
    assert roundtripped.filters == cmd.filters

    for malformed in ("", "   ", 'ticket.create "unclosed', "ticket.create bare_token", "ticket.create =val"):
        with pytest.raises(ValueError):
            DSLCommand.from_text(malformed)


def test_semantic_cache_non_string_slots_and_mcp_strict_validation() -> None:
    cache = SemanticCache(confidence_threshold=0.7)
    cache.register_template(
        "pokaż 10 zadań",
        DSLCommand(entity="ticket", operation="list", arguments={"limit": 10}),
        slots={"limit": 10, "active": True},
    )
    matched = cache.match("pokaż 10 zadań")
    assert matched is not None
    assert matched[0].arguments["limit"] == 10

    bridge = NLDSLLLMBridge(DSLExecutor())
    with pytest.raises(ValueError, match="non-empty string"):
        InterfaceAdapters.handle_mcp_tool_call(bridge, "nl_ask", {"query": ""})
    with pytest.raises(ValueError, match="boolean"):
        InterfaceAdapters.handle_mcp_tool_call(
            bridge, "nl_ask", {"query": "list tickets", "allow_llm_fallback": "no"}
        )


def test_system_one_action_uri_encoding_and_strict_envelope_validation() -> None:
    cmds = [DSLCommand(entity="ticket", operation="create", arguments={"title": "Fix login bug"})]
    cands = candidates_from_dsl_commands(cmds)
    assert cands[0].uri == "proc://taskand.dev/ticket/create/v1?title=Fix%20login%20bug"

    env = SystemOneDecisionEngine().decide("create ticket", cands)
    validate_system_one_decision_envelope(env)

    nan_env = copy.deepcopy(env)
    nan_env["probabilities"]["risk_score"]["1"] = float("nan")
    with pytest.raises(ValueError, match="probability"):
        validate_system_one_decision_envelope(nan_env)

    bool_risk = copy.deepcopy(env)
    bool_risk["risk_score"] = True
    bool_risk["answers"]["risk_score"] = "1"
    with pytest.raises(ValueError, match="risk_score"):
        validate_system_one_decision_envelope(bool_risk)

    int_allowed = copy.deepcopy(env)
    int_allowed["execution_allowed"] = 1
    int_allowed["answers"]["execution_allowed"] = "yes"
    with pytest.raises(ValueError, match="execution_allowed"):
        validate_system_one_decision_envelope(int_allowed)

    unhashable_opts = copy.deepcopy(env)
    unhashable_opts["questions"][0]["options"] = [{"bad": 1}]
    with pytest.raises(ValueError, match="uri_choice"):
        validate_system_one_decision_envelope(unhashable_opts)


def test_select_operation_denies_explicit_empty_allowed_uris_and_nan_confidence() -> None:
    contracts = [
        {
            "uri": "proc://taskand.dev/service/status/v1?target=postgres",
            "dsl_command": "service.status target=postgres",
            "description": "Check postgres status",
            "keywords": ["status", "postgres"],
            "risk_level": 1,
            "required_args": ["target"],
            "arguments": {"target": "postgres"},
        }
    ]
    denied = select_operation("status postgres", contracts, {"allowed_uris": []})
    assert denied["gates"]["gate_4_resource_acl"] is False
    assert denied["execution_authorized"] is False
    assert denied["selected_uri"] is None

    nan_res = select_operation(
        "status postgres",
        contracts,
        decision_provider=lambda _q: {"chosen_option": "op_0", "confidence": float("nan")},
    )
    assert nan_res["gates"]["gate_1_candidate_membership"] is False
    assert nan_res["execution_authorized"] is False
    assert nan_res["confidence"] == 0.0


def test_websocket_bounds_malformed_frames_and_unique_sessions() -> None:
    # 64-bit oversized frame header must be rejected before buffering
    oversized_header = bytearray([0x81, 127]) + struct.pack("!Q", 10 * 1024 * 1024)
    with pytest.raises(ValueError, match="exceeds max"):
        decode_ws_frame(oversized_header, max_payload_bytes=1024)

    with pytest.raises(ValueError, match="sequence"):
        NLStreamFrame.from_json('{"type": "STREAM_START", "sequence": true}')

    async def _exercise_server() -> None:
        server = NLStreamServer(host="127.0.0.1", port=0, max_handshake_bytes=256)
        await server.start()
        assert server.server is not None
        port = server.server.sockets[0].getsockname()[1]
        try:
            # 1. Oversized HTTP header without CRLFCRLF gets HTTP 431
            r_big, w_big = await asyncio.open_connection("127.0.0.1", port)
            w_big.write(b"GET / HTTP/1.1\r\nX-Pad: " + (b"A" * 512))
            await w_big.drain()
            resp_big = await r_big.read(256)
            assert b"431" in resp_big
            w_big.close()
            await w_big.wait_closed()

            # 2. Two simultaneous WS connections get distinct session IDs & malformed JSON emits STREAM_ERROR
            key = base64.b64encode(b"0123456789abcdef").decode("ascii")
            hs = (
                f"GET /ws HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
                f"Upgrade: websocket\r\nConnection: Upgrade\r\n"
                f"Sec-WebSocket-Key: {key}\r\n\r\n"
            ).encode("ascii")

            r1, w1 = await asyncio.open_connection("127.0.0.1", port)
            r2, w2 = await asyncio.open_connection("127.0.0.1", port)
            w1.write(hs)
            w2.write(hs)
            await asyncio.gather(w1.drain(), w2.drain())
            await asyncio.gather(r1.readuntil(b"\r\n\r\n"), r2.readuntil(b"\r\n\r\n"))
            assert len(server.active_sessions) == 2

            # Malformed JSON frame emits STREAM_ERROR instead of silent drop
            w1.write(encode_ws_frame("not-json", opcode=1, mask=True))
            await w1.drain()
            raw = bytearray(await r1.read(512))
            decoded = decode_ws_frame(raw)
            assert decoded is not None
            err_frame = NLStreamFrame.from_json(decoded[1].decode("utf-8"))
            assert err_frame.type == StreamFrameType.STREAM_ERROR

            for w in (w1, w2):
                w.close()
                await w.wait_closed()
        finally:
            await server.stop()

    asyncio.run(_exercise_server())


def test_rfc6455_handshake_control_frames_and_json_depth_nan_hardening() -> None:
    from standard.nl_dsl_llm_ws import make_ws_handshake_response

    # 1. Invalid / non-ASCII Sec-WebSocket-Key raises ValueError instead of UnicodeEncodeError
    for bad_key in ("short", "not-base64-key-at-all!!!", "żółć" * 6, "QUFBQUFBQUFBQUFBQUFBQ==="):
        with pytest.raises(ValueError):
            make_ws_handshake_response(bad_key)

    # 2. Control frame > 125 bytes, fragmented control frame, RSV bits, non-minimal lengths rejected
    with pytest.raises(ValueError, match="125"):
        encode_ws_frame(b"x" * 126, opcode=9)
    with pytest.raises(ValueError, match="125"):
        decode_ws_frame(bytearray([0x89, 126, 0x00, 126]) + (b"x" * 126))
    with pytest.raises(ValueError, match="fragmented"):
        decode_ws_frame(bytearray([0x09, 0x00]))
    with pytest.raises(ValueError, match="RSV"):
        decode_ws_frame(bytearray([0x91, 0x00]))
    with pytest.raises(ValueError, match="Non-minimal"):
        decode_ws_frame(bytearray([0x81, 126, 0x00, 10]) + (b"a" * 10))

    # 3. NLStreamFrame rejects NaN/Infinity and >64 JSON nesting
    with pytest.raises(ValueError):
        NLStreamFrame.from_json('{"type": "STREAM_START", "payload": {"x": NaN}}')
    deep = '{"type": "STREAM_START", "payload": ' + ('{"a": ' * 70) + "1" + ("}" * 70) + "}"
    with pytest.raises(ValueError):
        NLStreamFrame.from_json(deep)
    bad_out = NLStreamFrame(
        type=StreamFrameType.STREAM_START,
        session_id="s1",
        sequence=1,
        timestamp_ms=1.0,
        payload={"bad": float("nan")},
    )
    with pytest.raises(ValueError):
        bad_out.to_json()

    # 4. Non-ASCII Sec-WebSocket-Key over TCP returns HTTP 400 cleanly
    async def _bad_upgrade() -> None:
        server = NLStreamServer(host="127.0.0.1", port=0)
        await server.start()
        assert server.server is not None
        port = server.server.sockets[0].getsockname()[1]
        try:
            r_bad, w_bad = await asyncio.open_connection("127.0.0.1", port)
            w_bad.write(
                b"GET /ws HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                b"Upgrade: websocket\r\nConnection: Upgrade\r\n"
                b"Sec-WebSocket-Key: \xff\xfe\xfd\r\n\r\n"
            )
            await w_bad.drain()
            resp = await r_bad.read(256)
            assert b"400 Bad Request" in resp
            w_bad.close()
            await w_bad.wait_closed()
        finally:
            await server.stop()

    asyncio.run(_bad_upgrade())


def test_select_operation_malformed_contract_fields_and_252_digest_parity() -> None:
    from standard.system_one_decision import build_selection_question

    # Boolean risk_level and non-mapping arguments fail gate_3_argument_schema cleanly without crash
    malformed_contracts = [
        {
            "uri": "proc://taskand.dev/service/status/v1?target=db",
            "dsl_command": "service.status target=db",
            "description": "db status",
            "keywords": ["status", "db"],
            "risk_level": True,
            "arguments": ["not-a-dict"],
        }
    ]
    res = select_operation("status db", malformed_contracts)
    assert res["gates"]["gate_3_argument_schema"] is False
    assert res["execution_authorized"] is False

    # Non-string chosen_option / registry_digest from provider fail closed
    res_provider = select_operation(
        "status db",
        [
            {
                "uri": "proc://taskand.dev/service/status/v1?target=db",
                "dsl_command": "service.status target=db",
                "risk_level": 1,
            }
        ],
        decision_provider=lambda _q: {
            "chosen_option": None,
            "confidence": 0.9,
            "registry_digest": 12345,
        },
    )
    assert res_provider["gates"]["gate_1_candidate_membership"] is False
    assert res_provider["gates"]["gate_2_registry_digest"] is False
    assert res_provider["execution_authorized"] is False

    # 252-contract cap has exact digest parity with sliced bindings
    many = [
        {
            "uri": f"proc://taskand.dev/op/run/v1?i={i}",
            "dsl_command": f"op.run i={i}",
            "risk_level": 1,
        }
        for i in range(260)
    ]
    q_260 = build_selection_question("run", many)
    q_252 = build_selection_question("run", many[:252])
    assert q_260["registry_digest"] == q_252["registry_digest"]
    assert len(q_260["bindings"]) == 252
