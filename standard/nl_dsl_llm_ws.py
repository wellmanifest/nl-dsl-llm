#!/usr/bin/env python3
"""Standard reference implementation for NL-DSL-LLM WebSocket Bidirectional Streaming Protocol.

Provides RFC 6455 framing with zero external dependencies, live Option Network
DAG autocomplete projection with sub-50ms latency, and 4-tier pipeline integration.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import math
import os
import re
import struct
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from standard.nl_dsl_llm import (
    DigitalTwinContext,
    DSLExecutor,
    DSLResult,
    NLDSLLLMBridge,
    OptionNetworkEngine,
)

WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
MAX_HANDSHAKE_BYTES = 16 * 1024
MAX_FRAME_PAYLOAD_BYTES = 1024 * 1024


class StreamFrameType(str, Enum):
    STREAM_START = "STREAM_START"
    AUDIO_CHUNK = "AUDIO_CHUNK"
    TRANSCRIPTION_PARTIAL = "TRANSCRIPTION_PARTIAL"
    OPTION_NETWORK_UPDATE = "OPTION_NETWORK_UPDATE"
    COMMAND_COMMITTED = "COMMAND_COMMITTED"
    COMMAND_RESULT = "COMMAND_RESULT"
    STREAM_ERROR = "STREAM_ERROR"


@dataclass
class NLStreamFrame:
    type: StreamFrameType
    session_id: str
    sequence: int
    timestamp_ms: float
    payload: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        data = {
            "type": self.type.value if isinstance(self.type, StreamFrameType) else str(self.type),
            "session_id": self.session_id,
            "sequence": self.sequence,
            "timestamp_ms": self.timestamp_ms,
            "payload": self.payload,
        }
        return json.dumps(data)

    @classmethod
    def from_json(cls, raw: str) -> NLStreamFrame:
        data = json.loads(raw)
        invalid_data = not isinstance(data, dict)
        if invalid_data:
            raise ValueError("NLStreamFrame JSON must be an object")
        raw_seq = data.get("sequence", 0)
        if type(raw_seq) is not int or raw_seq < 0:
            raise ValueError("NLStreamFrame sequence must be a non-negative integer")
        raw_ts = data.get("timestamp_ms", time.time() * 1000)
        if not isinstance(raw_ts, (int, float)) or isinstance(raw_ts, bool) or not math.isfinite(float(raw_ts)):
            raise ValueError("NLStreamFrame timestamp_ms must be a finite number")
        raw_payload = data.get("payload", {})
        invalid_payload = not isinstance(raw_payload, dict)
        if invalid_payload:
            raise ValueError("NLStreamFrame payload must be a JSON object")
        session_id = data.get("session_id", "default")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("NLStreamFrame session_id must be a non-empty string")
        return cls(
            type=StreamFrameType(data["type"]),
            session_id=session_id,
            sequence=raw_seq,
            timestamp_ms=float(raw_ts),
            payload=raw_payload,
        )


def _xor_ws_mask(payload: bytes | bytearray, mask: bytes | bytearray) -> bytes:
    """Fast vectorized RFC 6455 4-byte XOR masking/unmasking."""
    length = len(payload)
    if length == 0:
        return b""
    repeated_mask = (bytes(mask) * ((length // 4) + 1))[:length]
    return (
        int.from_bytes(payload, "big") ^ int.from_bytes(repeated_mask, "big")
    ).to_bytes(length, "big")


def make_ws_handshake_response(key: str) -> bytes:
    """Create HTTP 101 Switching Protocols response for WebSocket upgrade."""
    accept_src = key.strip() + WS_GUID
    accept_val = base64.b64encode(hashlib.sha1(accept_src.encode("ascii")).digest()).decode("ascii")
    response = (
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Accept: {accept_val}\r\n"
        "\r\n"
    )
    return response.encode("ascii")


def encode_ws_frame(
    payload: bytes | str,
    opcode: int = 1,
    mask: bool = False,
    *,
    max_payload_bytes: int = MAX_FRAME_PAYLOAD_BYTES,
) -> bytes:
    """Encode an RFC 6455 WebSocket frame with bounded payload size."""
    payload_bytes = payload.encode("utf-8") if isinstance(payload, str) else payload
    length = len(payload_bytes)
    if length > max_payload_bytes:
        raise ValueError(f"WebSocket payload length {length} exceeds limit {max_payload_bytes}")

    header = bytearray([0x80 | (opcode & 0x0F)])
    mask_bit = 0x80 if mask else 0x00
    if length < 126:
        header.append(mask_bit | length)
    elif length <= 0xFFFF:
        header.append(mask_bit | 126)
        header.extend(struct.pack("!H", length))
    else:
        header.append(mask_bit | 127)
        header.extend(struct.pack("!Q", length))

    if mask:
        masking_key = os.urandom(4)
        header.extend(masking_key)
        return bytes(header) + _xor_ws_mask(payload_bytes, masking_key)

    return bytes(header) + payload_bytes


def decode_ws_frame(
    buffer: bytearray, *, max_payload_bytes: int = MAX_FRAME_PAYLOAD_BYTES
) -> tuple[int, bytes, int] | None:
    """Decode an RFC 6455 WebSocket frame, raising ValueError if payload exceeds max_payload_bytes."""
    if len(buffer) < 2:
        return None

    opcode = buffer[0] & 0x0F
    is_masked = (buffer[1] & 0x80) != 0
    payload_len = buffer[1] & 0x7F

    offset = 2
    if payload_len == 126:
        if len(buffer) < 4:
            return None
        payload_len = struct.unpack("!H", buffer[2:4])[0]
        offset = 4
    elif payload_len == 127:
        if len(buffer) < 10:
            return None
        payload_len = struct.unpack("!Q", buffer[2:10])[0]
        offset = 10

    if payload_len > max_payload_bytes:
        raise ValueError(f"WebSocket frame payload {payload_len} exceeds max {max_payload_bytes}")

    mask: bytes | None = None
    if is_masked:
        if len(buffer) < offset + 4:
            return None
        mask = bytes(buffer[offset : offset + 4])
        offset += 4

    if len(buffer) < offset + payload_len:
        return None

    raw_data = bytes(buffer[offset : offset + payload_len])
    payload = _xor_ws_mask(raw_data, mask) if (is_masked and mask is not None) else raw_data
    return opcode, payload, offset + payload_len


class StreamingOptionNetworkEngine:
    """Compute real-time DAG autocomplete suggestions from streaming prefix tokens (< 50ms SLA)."""

    def __init__(self, context: DigitalTwinContext | None = None) -> None:
        self.context = context or DigitalTwinContext()
        self.base_engine = OptionNetworkEngine()

    def update_prefix(self, prefix: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        clean = prefix.strip() if isinstance(prefix, str) else ""
        nodes = self.base_engine.suggest(clean, twin=self.context, max_suggestions=5)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        root_id = "node:root"
        candidates: list[dict[str, Any]] = []
        dag_nodes: list[dict[str, Any]] = [
            {"id": root_id, "label": clean or "start", "type": "prefix", "status": "active"}
        ]
        dag_edges: list[dict[str, Any]] = []

        for idx, node in enumerate(nodes[:5]):
            cand_id = f"node:cand_{idx}"
            candidates.append(
                {
                    "id": cand_id,
                    "label": node.display_label,
                    "completion_text": node.completion_text,
                    "action_id": node.action_id,
                    "confidence": round(node.confidence, 3),
                    "badge": node.badge,
                    "requires_confirmation": node.requires_confirmation,
                    "bound_arguments": node.bound_arguments,
                }
            )
            dag_nodes.append(
                {
                    "id": cand_id,
                    "label": node.display_label,
                    "action_id": node.action_id,
                    "confidence": round(node.confidence, 3),
                    "badge": node.badge,
                    "type": "command_candidate",
                }
            )
            dag_edges.append({"from": root_id, "to": cand_id, "weight": round(node.confidence, 3)})

            for sub_idx, sub in enumerate(node.next_options[:3]):
                sub_id = f"node:sub_{idx}_{sub_idx}"
                dag_nodes.append({"id": sub_id, "label": sub.display_label, "type": "param_option"})
                dag_edges.append({"from": cand_id, "to": sub_id, "weight": round(sub.confidence, 3)})

        return {
            "prefix": clean,
            "candidates": candidates,
            "dag": {"nodes": dag_nodes, "edges": dag_edges},
            "candidate_count": len(candidates),
            "latency_ms": round(latency_ms, 3),
            "sla_met": latency_ms < 50.0,
        }


class NLStreamServer:
    """Async WebSocket streaming server for NL-DSL-LLM sessions."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        *,
        max_handshake_bytes: int = MAX_HANDSHAKE_BYTES,
        max_frame_payload_bytes: int = MAX_FRAME_PAYLOAD_BYTES,
    ) -> None:
        self.host = host
        self.port = port
        self.max_handshake_bytes = max_handshake_bytes
        self.max_frame_payload_bytes = max_frame_payload_bytes
        self.server: asyncio.Server | None = None
        self.engine = StreamingOptionNetworkEngine()
        self.executor = DSLExecutor()
        self.executor.register("ticket", "list", lambda _cmd: [{"id": "ticket-001", "title": "Example task", "status": "OPEN"}])
        self.executor.register("ticket", "create", lambda cmd: {"id": "ticket-002", "title": cmd.arguments.get("title", "New task"), "status": "OPEN"})
        self.executor.register("ticket", "close", lambda cmd: {"id": cmd.arguments.get("id", "ticket-001"), "status": "CLOSED"})
        self.executor.register("service", "restart", lambda cmd: {"service": cmd.arguments.get("target", "auth-service"), "status": "restarted"})
        self.executor.register("service", "stop", lambda cmd: {"service": cmd.arguments.get("target", "auth-service"), "status": "stopped"})
        self.executor.register("cluster", "status", lambda _cmd: {"cluster": "ready", "env": "prod"})
        self.bridge = NLDSLLLMBridge(executor=self.executor)
        self.active_sessions: dict[str, dict[str, Any]] = {}
        self._client_tasks: set[asyncio.Task[Any]] = set()
        self._seq = 0

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    async def start(self) -> None:
        self.server = await asyncio.start_server(self.handle_client, self.host, self.port)

    async def stop(self) -> None:
        for task in list(self._client_tasks):
            task.cancel()
        if self.server:
            self.server.close()
            await self.server.wait_closed()

    async def _send_error(
        self, writer: asyncio.StreamWriter, session_id: str, code: str, message: str
    ) -> None:
        err_frame = NLStreamFrame(
            type=StreamFrameType.STREAM_ERROR,
            session_id=session_id,
            sequence=self._next_seq(),
            timestamp_ms=time.time() * 1000,
            payload={"error": message, "code": code},
        )
        writer.write(encode_ws_frame(err_frame.to_json()))
        await writer.drain()

    async def handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        task = asyncio.current_task()
        if task:
            self._client_tasks.add(task)
        buffer = bytearray()
        handshake_done = False
        session_id = f"sess_{int(time.time() * 1000)}_{self._next_seq()}"
        self.active_sessions[session_id] = {
            "accumulated_text": "",
            "audio_chunk_count": 0,
            "created_at": time.time(),
        }

        try:
            while True:
                if not handshake_done:
                    header_end = buffer.find(b"\r\n\r\n")
                    if header_end == -1:
                        if len(buffer) > self.max_handshake_bytes:
                            writer.write(b"HTTP/1.1 431 Request Header Fields Too Large\r\n\r\n")
                            await writer.drain()
                            return
                        data = await reader.read(4096)
                        if not data:
                            break
                        buffer.extend(data)
                        continue
                    header_text = buffer[:header_end].decode("latin-1")
                    del buffer[: header_end + 4]

                    key_match = re.search(r"Sec-WebSocket-Key:\s*([^\r\n]+)", header_text, re.IGNORECASE)
                    if not key_match:
                        writer.write(b"HTTP/1.1 400 Bad Request\r\n\r\n")
                        await writer.drain()
                        return

                    writer.write(make_ws_handshake_response(key_match.group(1).strip()))
                    await writer.drain()
                    handshake_done = True

                frame = decode_ws_frame(buffer, max_payload_bytes=self.max_frame_payload_bytes)
                if frame is not None:
                    opcode, payload, consumed = frame
                    del buffer[:consumed]

                    if opcode == 8:
                        writer.write(encode_ws_frame(b"", opcode=8))
                        await writer.drain()
                        return
                    if opcode == 9:
                        writer.write(encode_ws_frame(payload, opcode=10))
                        await writer.drain()
                    elif opcode in (1, 2):
                        await self._dispatch_message(session_id, payload, opcode, writer)
                    continue

                if len(buffer) > self.max_frame_payload_bytes + 16:
                    await self._send_error(writer, session_id, "ERR_FRAME_TOO_LARGE", "Frame buffer exceeded limit")
                    return
                data = await reader.read(4096)
                if not data:
                    break
                buffer.extend(data)

        except asyncio.CancelledError:
            pass
        except Exception as error:  # noqa: BLE001
            try:
                if handshake_done:
                    await self._send_error(writer, session_id, "ERR_STREAM_EXCEPTION", str(error))
            except OSError:
                pass
        finally:
            if task:
                self._client_tasks.discard(task)
            self.active_sessions.pop(session_id, None)
            try:
                writer.close()
                await writer.wait_closed()
            except OSError:
                pass

    async def _dispatch_message(
        self,
        session_id: str,
        payload: bytes,
        opcode: int,
        writer: asyncio.StreamWriter,
    ) -> None:
        session = self.active_sessions.get(session_id, {})
        if opcode == 2:
            session["audio_chunk_count"] = session.get("audio_chunk_count", 0) + 1
            return

        try:
            frame_in = NLStreamFrame.from_json(payload.decode("utf-8"))
        except (UnicodeDecodeError, ValueError, KeyError, TypeError) as exc:
            await self._send_error(writer, session_id, "ERR_INVALID_FRAME", str(exc))
            return

        if frame_in.type == StreamFrameType.STREAM_START:
            resp = NLStreamFrame(
                type=StreamFrameType.STREAM_START,
                session_id=session_id,
                sequence=self._next_seq(),
                timestamp_ms=time.time() * 1000,
                payload={
                    "status": "ready",
                    "server": "nl-dsl-llm-ws/v1",
                    "features": ["audio_stream", "option_network_stream", "digital_twin_projection"],
                    "sla_max_latency_ms": 50.0,
                },
            )
            writer.write(encode_ws_frame(resp.to_json()))
            await writer.drain()

        elif frame_in.type == StreamFrameType.AUDIO_CHUNK:
            session["audio_chunk_count"] = session.get("audio_chunk_count", 0) + 1
            partial_text = frame_in.payload.get("partial_text", "")
            if isinstance(partial_text, str) and partial_text:
                session["accumulated_text"] = partial_text
                trans_frame = NLStreamFrame(
                    type=StreamFrameType.TRANSCRIPTION_PARTIAL,
                    session_id=session_id,
                    sequence=self._next_seq(),
                    timestamp_ms=time.time() * 1000,
                    payload={"text": partial_text, "stability": 0.85, "is_final": False},
                )
                writer.write(encode_ws_frame(trans_frame.to_json()))
                opt_frame = NLStreamFrame(
                    type=StreamFrameType.OPTION_NETWORK_UPDATE,
                    session_id=session_id,
                    sequence=self._next_seq(),
                    timestamp_ms=time.time() * 1000,
                    payload=self.engine.update_prefix(partial_text),
                )
                writer.write(encode_ws_frame(opt_frame.to_json()))
                await writer.drain()

        elif frame_in.type == StreamFrameType.TRANSCRIPTION_PARTIAL:
            raw_prefix = frame_in.payload.get("text", "")
            current_prefix = raw_prefix if isinstance(raw_prefix, str) else ""
            session["accumulated_text"] = current_prefix
            opt_frame = NLStreamFrame(
                type=StreamFrameType.OPTION_NETWORK_UPDATE,
                session_id=session_id,
                sequence=self._next_seq(),
                timestamp_ms=time.time() * 1000,
                payload=self.engine.update_prefix(current_prefix),
            )
            writer.write(encode_ws_frame(opt_frame.to_json()))
            await writer.drain()

        elif frame_in.type == StreamFrameType.COMMAND_COMMITTED:
            raw_cmd = frame_in.payload.get("text")
            command_text = raw_cmd if isinstance(raw_cmd, str) and raw_cmd else session.get("accumulated_text", "")
            t_exec_0 = time.perf_counter()
            result: DSLResult = self.bridge.handle_request(command_text)
            exec_ms = (time.perf_counter() - t_exec_0) * 1000.0
            res_frame = NLStreamFrame(
                type=StreamFrameType.COMMAND_RESULT,
                session_id=session_id,
                sequence=self._next_seq(),
                timestamp_ms=time.time() * 1000,
                payload={
                    "success": result.success,
                    "status": result.status,
                    "data": result.data,
                    "errors": result.errors,
                    "meta": result.meta,
                    "execution_ms": round(exec_ms, 3),
                },
            )
            writer.write(encode_ws_frame(res_frame.to_json()))
            await writer.drain()
