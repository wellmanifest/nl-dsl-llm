#!/usr/bin/env python3
"""Conformance tests for NL-DSL-LLM WebSocket Streaming & Option Network Autocomplete."""

import asyncio
import base64
import json
import time
import pytest

from standard.nl_dsl_llm_ws import (
    NLStreamFrame,
    NLStreamServer,
    StreamFrameType,
    StreamingOptionNetworkEngine,
    decode_ws_frame,
    encode_ws_frame,
    make_ws_handshake_response,
)


def test_handshake_response_generation():
    """Verify standard RFC 6455 WebSocket handshake accept key calculation."""
    client_key = "dGhlIHNhbXBsZSBub25jZQ=="
    expected_accept = "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="
    response = make_ws_handshake_response(client_key).decode("ascii")

    assert "HTTP/1.1 101 Switching Protocols" in response
    assert "Upgrade: websocket" in response
    assert f"Sec-WebSocket-Accept: {expected_accept}" in response


def test_frame_encode_decode_roundtrip():
    """Verify framing encode and decode for text, binary, and masked frames."""
    text_msg = "Hello, Option Network!"
    encoded = encode_ws_frame(text_msg, opcode=1)

    # Server to client: unmasked
    buffer = bytearray(encoded)
    decoded = decode_ws_frame(buffer)
    assert decoded is not None
    opcode, payload, consumed = decoded
    assert opcode == 1
    assert payload.decode("utf-8") == text_msg
    assert consumed == len(encoded)

    # Client to server: simulate masking key
    mask = b"\x12\x34\x56\x78"
    raw_payload = text_msg.encode("utf-8")
    masked_payload = bytearray(len(raw_payload))
    for i in range(len(raw_payload)):
        masked_payload[i] = raw_payload[i] ^ mask[i % 4]

    client_header = bytearray([0x81, 0x80 | len(raw_payload)]) + mask
    client_frame = bytearray(client_header + masked_payload)

    decoded_client = decode_ws_frame(client_frame)
    assert decoded_client is not None
    c_opcode, c_payload, c_consumed = decoded_client
    assert c_opcode == 1
    assert c_payload.decode("utf-8") == text_msg
    assert c_consumed == len(client_frame)


def test_streaming_option_network_latency_sla():
    """Verify Option Network autocomplete executes in < 50ms SLA (sub-5ms typical)."""
    engine = StreamingOptionNetworkEngine()

    prefixes = ["z", "zatrz", "dod", "dodaj", "pok", "pokaż", "res", "restart"]
    for prefix in prefixes:
        res = engine.update_prefix(prefix)
        assert res["sla_met"] is True
        assert res["latency_ms"] < 50.0  # Specification SLA
        assert "dag" in res
        assert "candidates" in res
        assert len(res["dag"]["nodes"]) > 0
        assert res["prefix"] == prefix


def test_streaming_option_network_dag_structure():
    """Verify generated DAG nodes and edge weights for graph visualization."""
    engine = StreamingOptionNetworkEngine()
    res = engine.update_prefix("zatrzymaj")

    dag = res["dag"]
    nodes = dag["nodes"]
    edges = dag["edges"]

    # Root node exists
    assert any(n["id"] == "node:root" for n in nodes)

    # Candidate nodes connected to root
    cand_nodes = [n for n in nodes if n.get("type") == "command_candidate"]
    assert len(cand_nodes) > 0

    for cand in cand_nodes:
        assert any(e["from"] == "node:root" and e["to"] == cand["id"] for e in edges)

    # Sub-option child edges exist
    param_nodes = [n for n in nodes if n.get("type") == "param_option"]
    if param_nodes:
        assert any(e["to"] == param_nodes[0]["id"] for e in edges)


@pytest.mark.asyncio
async def test_websocket_streaming_server_session_lifecycle():
    """End-to-end integration test of NLStreamServer with simulated client."""
    server = NLStreamServer(host="127.0.0.1", port=0)
    await server.start()
    assert server.server is not None
    port = server.server.sockets[0].getsockname()[1]

    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)

        # 1. Perform WebSocket Handshake
        client_key = base64.b64encode(b"0123456789abcdef").decode("ascii")
        handshake_req = (
            f"GET /ws/stream HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{port}\r\n"
            f"Upgrade: websocket\r\n"
            f"Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {client_key}\r\n"
            f"Sec-WebSocket-Version: 13\r\n"
            f"\r\n"
        )
        writer.write(handshake_req.encode("ascii"))
        await writer.drain()

        # Read Handshake Response
        handshake_resp = await reader.readuntil(b"\r\n\r\n")
        assert b"101 Switching Protocols" in handshake_resp

        # Helper to send masked text frame
        def send_text_frame(text: str):
            writer.write(encode_ws_frame(text, opcode=1, mask=True))

        rx_buf = bytearray()
        # Helper to read next unmasked frame
        async def read_frame() -> tuple[int, str]:
            while True:
                res = decode_ws_frame(rx_buf)
                if res is not None:
                    opcode, payload, consumed = res
                    del rx_buf[:consumed]
                    return opcode, payload.decode("utf-8")
                chunk = await reader.read(1024)
                assert len(chunk) > 0, "Connection closed unexpectedly"
                rx_buf.extend(chunk)

        # 2. Send STREAM_START
        start_frame = NLStreamFrame(
            type=StreamFrameType.STREAM_START,
            session_id="client_session_1",
            sequence=1,
            timestamp_ms=time.time() * 1000,
            payload={"client": "test-runner", "audio_codec": "pcm16"},
        )
        send_text_frame(start_frame.to_json())
        await writer.drain()

        opcode, resp_raw = await read_frame()
        resp_start = NLStreamFrame.from_json(resp_raw)
        assert resp_start.type == StreamFrameType.STREAM_START
        assert resp_start.payload["status"] == "ready"

        # 3. Send TRANSCRIPTION_PARTIAL (live typing simulation)
        partial_frame = NLStreamFrame(
            type=StreamFrameType.TRANSCRIPTION_PARTIAL,
            session_id="client_session_1",
            sequence=2,
            timestamp_ms=time.time() * 1000,
            payload={"text": "zatrz"},
        )
        send_text_frame(partial_frame.to_json())
        await writer.drain()

        opcode, resp_raw = await read_frame()
        resp_opt = NLStreamFrame.from_json(resp_raw)
        assert resp_opt.type == StreamFrameType.OPTION_NETWORK_UPDATE
        assert resp_opt.payload["prefix"] == "zatrz"
        assert len(resp_opt.payload["candidates"]) > 0
        assert resp_opt.payload["sla_met"] is True

        # 4. Send COMMAND_COMMITTED
        commit_frame = NLStreamFrame(
            type=StreamFrameType.COMMAND_COMMITTED,
            session_id="client_session_1",
            sequence=3,
            timestamp_ms=time.time() * 1000,
            payload={"text": "zamknij zadanie ticket-001"},
        )
        send_text_frame(commit_frame.to_json())
        await writer.drain()

        opcode, resp_raw = await read_frame()
        resp_cmd = NLStreamFrame.from_json(resp_raw)
        assert resp_cmd.type == StreamFrameType.COMMAND_RESULT
        assert resp_cmd.payload["success"] is True
        assert resp_cmd.payload["status"] == "OK"
        assert "execution_ms" in resp_cmd.payload

        # Close client
        writer.close()
        await writer.wait_closed()
    finally:
        await server.stop()
