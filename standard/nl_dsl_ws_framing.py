"""RFC 6455 WebSocket handshake, framing, and strict JSON stream envelopes for NL-DSL-LLM."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import struct
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
MAX_HANDSHAKE_BYTES = 16 * 1024
MAX_FRAME_PAYLOAD_BYTES = 1024 * 1024
MAX_JSON_DEPTH = 64
MAX_CONTROL_FRAME_BYTES = 125
_WS_KEY_RE = re.compile(r"^[A-Za-z0-9+/]{22}==$")


class StreamFrameType(str, Enum):
    STREAM_START = "STREAM_START"
    AUDIO_CHUNK = "AUDIO_CHUNK"
    TRANSCRIPTION_PARTIAL = "TRANSCRIPTION_PARTIAL"
    OPTION_NETWORK_UPDATE = "OPTION_NETWORK_UPDATE"
    COMMAND_COMMITTED = "COMMAND_COMMITTED"
    COMMAND_RESULT = "COMMAND_RESULT"
    STREAM_ERROR = "STREAM_ERROR"


def _reject_json_constant(token: str) -> Any:
    raise ValueError(f"Non-standard JSON constant is forbidden: {token}")


def _assert_json_depth(root: Any, max_depth: int = MAX_JSON_DEPTH) -> None:
    stack: list[tuple[Any, int]] = [(root, 0)]
    while stack:
        current, depth = stack.pop()
        if depth > max_depth:
            raise ValueError(f"JSON nesting exceeds limit of {max_depth}")
        if isinstance(current, dict):
            for val in current.values():
                if isinstance(val, (dict, list)):
                    stack.append((val, depth + 1))
        elif isinstance(current, list):
            for item in current:
                if isinstance(item, (dict, list)):
                    stack.append((item, depth + 1))


@dataclass
class NLStreamFrame:
    type: StreamFrameType
    session_id: str
    sequence: int
    timestamp_ms: float
    payload: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        if not math.isfinite(self.timestamp_ms):
            raise ValueError("NLStreamFrame timestamp_ms must be finite")
        _assert_json_depth(self.payload)
        data = {
            "type": self.type.value if isinstance(self.type, StreamFrameType) else str(self.type),
            "session_id": self.session_id,
            "sequence": self.sequence,
            "timestamp_ms": self.timestamp_ms,
            "payload": self.payload,
        }
        return json.dumps(data, allow_nan=False)

    @classmethod
    def from_json(cls, raw: str) -> NLStreamFrame:
        invalid_raw = not isinstance(raw, str)
        if invalid_raw:
            raise ValueError("NLStreamFrame raw JSON must be a string")
        try:
            data = json.loads(raw, parse_constant=_reject_json_constant)
        except RecursionError as exc:
            raise ValueError("NLStreamFrame JSON exceeds parser recursion limit") from exc
        _assert_json_depth(data)
        invalid_data = not isinstance(data, dict)
        if invalid_data:
            raise ValueError("NLStreamFrame JSON must be an object")
        raw_seq = data.get("sequence", 0)
        if type(raw_seq) is not int or raw_seq < 0:
            raise ValueError("NLStreamFrame sequence must be a non-negative integer")
        raw_ts = data.get("timestamp_ms", time.time() * 1000)
        if (
            not isinstance(raw_ts, (int, float))
            or isinstance(raw_ts, bool)
            or not math.isfinite(float(raw_ts))
        ):
            raise ValueError("NLStreamFrame timestamp_ms must be a finite number")
        raw_payload = data.get("payload", {})
        invalid_payload = not isinstance(raw_payload, dict)
        if invalid_payload:
            raise ValueError("NLStreamFrame payload must be a JSON object")
        session_id = data.get("session_id", "default")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("NLStreamFrame session_id must be a non-empty string")
        raw_type = data.get("type")
        invalid_type = not isinstance(raw_type, str)
        if invalid_type:
            raise ValueError("NLStreamFrame type must be a string")
        return cls(
            type=StreamFrameType(raw_type),
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
    """Create HTTP 101 Switching Protocols response for a valid RFC 6455 Sec-WebSocket-Key."""
    invalid_key = not isinstance(key, str)
    if invalid_key:
        raise ValueError("Sec-WebSocket-Key must be a string")
    cleaned = key.strip()
    if _WS_KEY_RE.fullmatch(cleaned) is None:
        raise ValueError("Invalid RFC 6455 Sec-WebSocket-Key")
    decoded = base64.b64decode(cleaned, validate=True)
    if len(decoded) != 16:
        raise ValueError("Sec-WebSocket-Key must decode to 16 bytes")
    accept_src = cleaned + WS_GUID
    accept_val = base64.b64encode(hashlib.sha1(accept_src.encode("ascii")).digest()).decode("ascii")
    response = (
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Accept: {accept_val}\r\n"
        "\r\n"
    )
    return response.encode("ascii")


def parse_ws_upgrade_key(header_text: str) -> str:
    """Validate an HTTP/1.1 WebSocket upgrade header block and return Sec-WebSocket-Key."""
    lines = header_text.split("\r\n")
    if not lines or not re.match(r"^GET\s+\S+\s+HTTP/1\.1$", lines[0]):
        raise ValueError("Invalid WebSocket HTTP/1.1 GET request line")
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if not line:
            continue
        if ":" not in line:
            raise ValueError("Malformed HTTP header line")
        name, val = line.split(":", 1)
        headers[name.strip().lower()] = val.strip()
    upgrade_tokens = {tok.strip().lower() for tok in headers.get("upgrade", "").split(",")}
    conn_tokens = {tok.strip().lower() for tok in headers.get("connection", "").split(",")}
    if "websocket" not in upgrade_tokens or "upgrade" not in conn_tokens:
        raise ValueError("Missing Upgrade: websocket or Connection: Upgrade header")
    ws_key = headers.get("sec-websocket-key", "")
    if _WS_KEY_RE.fullmatch(ws_key) is None:
        raise ValueError("Invalid Sec-WebSocket-Key header")
    return ws_key


def encode_ws_frame(
    payload: bytes | str,
    opcode: int = 1,
    mask: bool = False,
    *,
    max_payload_bytes: int = MAX_FRAME_PAYLOAD_BYTES,
) -> bytes:
    """Encode an RFC 6455 WebSocket frame with bounded payload and control-frame checks."""
    if not (0 <= opcode <= 0x0F):
        raise ValueError(f"Invalid WebSocket opcode: {opcode}")
    payload_bytes = payload.encode("utf-8") if isinstance(payload, str) else bytes(payload)
    length = len(payload_bytes)
    if opcode >= 0x08 and length > MAX_CONTROL_FRAME_BYTES:
        raise ValueError("WebSocket control frame payload cannot exceed 125 bytes")
    if length > max_payload_bytes:
        raise ValueError(f"WebSocket payload length {length} exceeds limit {max_payload_bytes}")

    header = bytearray([0x80 | opcode])
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
    """Decode an RFC 6455 WebSocket frame, enforcing RSV, control-frame, and size invariants."""
    if len(buffer) < 2:
        return None

    first_byte, second_byte = buffer[0], buffer[1]
    if (first_byte & 0x70) != 0:
        raise ValueError("Unsupported non-zero WebSocket RSV bits")
    fin = (first_byte & 0x80) != 0
    opcode = first_byte & 0x0F
    is_masked = (second_byte & 0x80) != 0
    payload_len = second_byte & 0x7F

    if opcode >= 0x08:
        if not fin:
            raise ValueError("WebSocket control frames must not be fragmented")
        if payload_len > MAX_CONTROL_FRAME_BYTES:
            raise ValueError("WebSocket control frame payload exceeds 125 bytes")

    offset = 2
    if payload_len == 126:
        if len(buffer) < 4:
            return None
        payload_len = struct.unpack("!H", buffer[2:4])[0]
        if payload_len < 126:
            raise ValueError("Non-minimal 16-bit WebSocket payload length encoding")
        offset = 4
    elif payload_len == 127:
        if len(buffer) < 10:
            return None
        payload_len = struct.unpack("!Q", buffer[2:10])[0]
        if (payload_len >> 63) != 0 or payload_len <= 0xFFFF:
            raise ValueError("Invalid or non-minimal 64-bit WebSocket payload length encoding")
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
