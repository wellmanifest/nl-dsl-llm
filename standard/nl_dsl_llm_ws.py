#!/usr/bin/env python3
"""Standard reference implementation for NL-DSL-LLM WebSocket Bidirectional Streaming Protocol.

Provides RFC 6455 framing with zero external dependencies, live Option Network
DAG autocomplete projection with sub-50ms latency, and 4-tier pipeline integration.
"""

from __future__ import annotations

import asyncio
import sys
import time
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
from standard.nl_dsl_ws_framing import (
    MAX_CONTROL_FRAME_BYTES,
    MAX_FRAME_PAYLOAD_BYTES,
    MAX_HANDSHAKE_BYTES,
    MAX_JSON_DEPTH,
    WS_GUID,
    NLStreamFrame,
    StreamFrameType,
    _xor_ws_mask,
    decode_ws_frame,
    encode_ws_frame,
    make_ws_handshake_response,
    parse_ws_upgrade_key,
)

__all__ = [
    "MAX_CONTROL_FRAME_BYTES",
    "MAX_FRAME_PAYLOAD_BYTES",
    "MAX_HANDSHAKE_BYTES",
    "MAX_JSON_DEPTH",
    "WS_GUID",
    "NLStreamFrame",
    "NLStreamServer",
    "StreamFrameType",
    "StreamingOptionNetworkEngine",
    "_xor_ws_mask",
    "decode_ws_frame",
    "encode_ws_frame",
    "make_ws_handshake_response",
    "parse_ws_upgrade_key",
]


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
                dag_edges.append(
                    {"from": cand_id, "to": sub_id, "weight": round(sub.confidence, 3)}
                )

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
        self.executor.register(
            "ticket",
            "list",
            lambda _cmd: [{"id": "ticket-001", "title": "Example task", "status": "OPEN"}],
        )
        self.executor.register(
            "ticket",
            "create",
            lambda cmd: {
                "id": "ticket-002",
                "title": cmd.arguments.get("title", "New task"),
                "status": "OPEN",
            },
        )
        self.executor.register(
            "ticket",
            "close",
            lambda cmd: {"id": cmd.arguments.get("id", "ticket-001"), "status": "CLOSED"},
        )
        self.executor.register(
            "service",
            "restart",
            lambda cmd: {
                "service": cmd.arguments.get("target", "auth-service"),
                "status": "restarted",
            },
        )
        self.executor.register(
            "service",
            "stop",
            lambda cmd: {
                "service": cmd.arguments.get("target", "auth-service"),
                "status": "stopped",
            },
        )
        self.executor.register(
            "cluster", "status", lambda _cmd: {"cluster": "ready", "env": "prod"}
        )
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

                    try:
                        ws_key = parse_ws_upgrade_key(header_text)
                        resp_bytes = make_ws_handshake_response(ws_key)
                    except ValueError:
                        writer.write(b"HTTP/1.1 400 Bad Request\r\n\r\n")
                        await writer.drain()
                        return

                    writer.write(resp_bytes)
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
                        writer.write(encode_ws_frame(payload[:MAX_CONTROL_FRAME_BYTES], opcode=10))
                        await writer.drain()
                    elif opcode in (1, 2):
                        await self._dispatch_message(session_id, payload, opcode, writer)
                    continue

                if len(buffer) > self.max_frame_payload_bytes + 16:
                    await self._send_error(
                        writer, session_id, "ERR_FRAME_TOO_LARGE", "Frame buffer exceeded limit"
                    )
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
                    "features": [
                        "audio_stream",
                        "option_network_stream",
                        "digital_twin_projection",
                    ],
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
            command_text = (
                raw_cmd
                if isinstance(raw_cmd, str) and raw_cmd
                else session.get("accumulated_text", "")
            )
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
