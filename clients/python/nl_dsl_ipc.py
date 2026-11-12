"""Low-level IPC framing, socket safety, polarity helpers, and self-test for NLClient."""

from __future__ import annotations

import json
import os
import re
import socket
import stat
from pathlib import Path
from typing import Any

MAX_IPC_RESPONSE_BYTES = 1_048_576
MAX_CLIENT_QUERY_CHARS = 4096
MAX_KORU_ITERATIONS = 1000

_NEGATION_TOKENS = frozenset(
    {
        "nie",
        "bez",
        "żadnych",
        "zadnych",
        "nigdy",
        "no",
        "not",
        "never",
        "without",
        "dont",
        "don't",
    }
)
_ASCII_TOKEN_SPLIT_RE = re.compile(r"[^\w']+", re.UNICODE)
_ASCII_INT_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9_-])([0-9]+)(?![A-Za-z0-9_-])")
_ASCII_TICKET_ID_RE = re.compile(
    r"\b([A-Za-z]{2,16}-[0-9]{1,8}|[0-9]{1,8})\b", re.ASCII
)


def reject_nonfinite(constant: str) -> Any:
    raise ValueError(f"Non-finite JSON constant in IPC payload: {constant}")


def has_negation(text: str) -> bool:
    lowered = text.lower()
    if "do not " in lowered or "nie " in f"{lowered} ":
        return True
    tokens = [tok for tok in _ASCII_TOKEN_SPLIT_RE.split(lowered) if tok]
    return any(tok in _NEGATION_TOKENS for tok in tokens)


def is_safe_unix_socket(socket_path: str) -> bool:
    if not isinstance(socket_path, str) or not socket_path or "\x00" in socket_path:
        return False
    path_obj = Path(socket_path)
    cur = Path(path_obj.anchor) if path_obj.is_absolute() else Path.cwd()
    parts = path_obj.parts[1:] if path_obj.is_absolute() else path_obj.parts
    try:
        for part in parts[:-1]:
            if part in ("", "."):
                continue
            if part == "..":
                cur = cur.parent
                continue
            cur = cur / part
            if len(cur.parts) > 2 and cur.is_symlink():
                return False
        st = os.lstat(socket_path)
    except OSError:
        return False
    return stat.S_ISSOCK(st.st_mode)


def recv_json_line(
    sock: socket.socket, max_bytes: int = MAX_IPC_RESPONSE_BYTES
) -> dict[str, Any]:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            break
        nl_idx = chunk.find(b"\n")
        if nl_idx != -1:
            head = chunk[:nl_idx]
            if total + len(head) > max_bytes:
                raise ValueError("IPC response exceeded MAX_IPC_RESPONSE_BYTES")
            chunks.append(head)
            break
        total += len(chunk)
        if total > max_bytes:
            raise ValueError("IPC response exceeded MAX_IPC_RESPONSE_BYTES")
        chunks.append(chunk)
    raw_line = b"".join(chunks).strip()
    if not raw_line:
        raise ValueError("Empty IPC response")
    decoded = json.loads(raw_line.decode("utf-8"), parse_constant=reject_nonfinite)
    if not isinstance(decoded, dict):
        raise TypeError("IPC response must be a JSON object")
    return decoded


def run_client_self_test(nl_client_cls: Any, koru_client_cls: Any) -> None:
    import tempfile
    import threading

    for bad_kwargs in (
        {"port": 0},
        {"port": 70000},
        {"port": True},
        {"timeout_seconds": 0},
        {"timeout_seconds": float("nan")},
    ):
        try:
            nl_client_cls(**bad_kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected ValueError for {bad_kwargs}")

    responses = [
        b'{"status":"READY","action_id":"sys.cpu.status"}\n{"trailing":true}\n',
        b'{"status":NaN}\n',
        b"x" * (MAX_IPC_RESPONSE_BYTES + 2048),
    ]

    def _serve_once(payload_bytes: bytes) -> int:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        bound = srv.getsockname()[1]

        def _worker() -> None:
            try:
                conn, _ = srv.accept()
                with conn:
                    conn.recv(4096)
                    conn.sendall(payload_bytes)
            finally:
                srv.close()

        threading.Thread(target=_worker, daemon=True).start()
        return bound

    p1 = _serve_once(responses[0])
    c1 = nl_client_cls(
        socket_path="/nonexistent/nl.sock",
        host="127.0.0.1",
        port=p1,
        timeout_seconds=2.0,
    )
    r1 = c1.interpret("pokaż status cpu")
    assert r1["status"] == "READY" and r1["action_id"] == "sys.cpu.status"

    for bad_wire in responses[1:]:
        port_n = _serve_once(bad_wire)
        cn = nl_client_cls(
            socket_path="/nonexistent/nl.sock",
            host="127.0.0.1",
            port=port_n,
            timeout_seconds=2.0,
        )
        rn = cn.interpret("pokaż status cpu")
        assert rn["action_id"] == "monitor.cpu_status"
        assert cn._fallback_engine is not None

    with tempfile.TemporaryDirectory() as tmp:
        real_file = Path(tmp) / "fake.sock"
        real_file.write_text("not-a-socket", encoding="utf-8")
        sym_sock = Path(tmp) / "sym.sock"
        os.symlink(real_file, sym_sock)
        assert not is_safe_unix_socket(str(sym_sock))
        assert not is_safe_unix_socket(str(real_file))

    koru = koru_client_cls(socket_path="/nonexistent/nl.sock")
    for negated in (
        "nie naprawiaj projektu",
        "don't run doctor",
        "nigdy nie uruchamiaj pętli",
        "without doctor",
    ):
        neg_res = koru.dispatch_natural_command(negated)
        assert neg_res["success"] is False and neg_res["cli_command"] == ""
    huge_loop = koru.dispatch_natural_command(
        "uruchom pętlę dla " + ("9" * 500) + " zadań"
    )
    assert (
        huge_loop["cli_command"]
        == f"koru --loop --max-iterations {MAX_KORU_ITERATIONS}"
    )
    over_budget = koru.dispatch_natural_command(
        "uruchom pętlę dla " + ("9" * 5000) + " zadań"
    )
    assert over_budget["success"] is False and over_budget["status"] == "CLARIFY"
    zero_loop = koru.dispatch_natural_command("uruchom pętlę dla 0 zadań")
    assert zero_loop["cli_command"] == "koru --loop --max-iterations 1"
    unicode_ticket = koru.dispatch_natural_command(
        "pokaż kontekst dla ticketa \u0661\u0662"
    )
    assert unicode_ticket["cli_command"] == "koru --context --ticket NEXT --format json"
    ascii_ticket = koru.dispatch_natural_command("pokaż kontekst dla ticketa plf-042")
    assert (
        ascii_ticket["cli_command"] == "koru --context --ticket PLF-042 --format json"
    )
    print("CLIENT-SELF-TEST-PASS")
