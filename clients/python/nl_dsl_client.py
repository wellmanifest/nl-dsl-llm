#!/usr/bin/env python3
"""
Python client for paxlet-com/nl-dsl-llm runtime daemon.
Enables sub-millisecond IPC for nl-dsl-sh, Clonerd, and Python scripts.
"""

from __future__ import annotations

import json
import os
import socket
import sys
from typing import Any, Dict, Optional

# Prevent .pyc pollution in git checkouts
sys.dont_write_bytecode = True

# Auto-discover local src directory if not installed as system package
_repo_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if os.path.isdir(_repo_src) and _repo_src not in sys.path:
    sys.path.insert(0, _repo_src)


class NLClient:
    """Client for NL-DSL-LLM IPC daemon over Unix Domain Socket or TCP."""

    def __init__(
        self,
        socket_path: Optional[str] = None,
        host: Optional[str] = None,
        port: int = 50051,
        client_id: str = "nl-dsl-sh",
        timeout_seconds: float = 5.0,
    ):
        self.socket_path = socket_path or os.environ.get("NL_DSL_SOCKET", "/run/paxlet/nl-dsl-llm.sock")
        self.host = host or os.environ.get("NL_DSL_HOST")
        self.port = int(os.environ.get("NL_DSL_PORT", port))
        self.client_id = client_id
        self.timeout = timeout_seconds

    def interpret(
        self,
        query: str,
        allow_fallback: bool = True,
        model: str = "qwen_2b",
        locale: str = "pl",
    ) -> Dict[str, Any]:
        """Send query to daemon for intent interpretation."""
        payload = {
            "action": "interpret",
            "query": query.strip(),
            "client_id": self.client_id,
            "locale": locale,
            "allow_llm_fallback": allow_fallback,
            "model_target": model,
        }
        return self._send_request(payload)

    def execute(
        self,
        action_id: str,
        arguments: Optional[Dict[str, Any]] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Execute a validated action via the daemon execution boundary."""
        payload = {
            "action": "execute",
            "action_id": action_id,
            "arguments": arguments or {},
            "dry_run": dry_run,
            "client_id": self.client_id,
        }
        return self._send_request(payload)

    def _send_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        # 1. Try Unix Domain Socket
        if os.path.exists(self.socket_path):
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                    sock.settimeout(self.timeout)
                    sock.connect(self.socket_path)
                    sock.sendall((json.dumps(payload) + "\n").encode("utf-8"))
                    data = b""
                    while b"\n" not in data:
                        chunk = sock.recv(4096)
                        if not chunk:
                            break
                        data += chunk
                    return json.loads(data.decode("utf-8").strip())
            except Exception:
                pass

        # 2. Try TCP fallback if host is set
        if self.host:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.settimeout(self.timeout)
                    sock.connect((self.host, self.port))
                    sock.sendall((json.dumps(payload) + "\n").encode("utf-8"))
                    data = b""
                    while b"\n" not in data:
                        chunk = sock.recv(4096)
                        if not chunk:
                            break
                        data += chunk
                    return json.loads(data.decode("utf-8").strip())
            except Exception:
                pass

        # 3. Direct in-memory engine fallback (local mode)
        try:
            from paxlet_nl_dsl_llm.server import NLRuntimeEngine
            engine = NLRuntimeEngine()
            if payload.get("action") == "interpret":
                return engine.interpret(
                    payload["query"],
                    client_id=self.client_id,
                    allow_llm_fallback=payload.get("allow_llm_fallback", True),
                    model_target=payload.get("model_target", "qwen_2b"),
                )
            elif payload.get("action") == "execute":
                return engine.execute(
                    payload.get("action_id", ""),
                    payload.get("arguments", {}),
                    dry_run=payload.get("dry_run", False),
                )
        except ImportError:
            pass

        return {
            "status": "UNAVAILABLE",
            "action_id": "",
            "arguments": {},
            "resolved_procedure_uri": "",
            "matched_layer": "SOURCE_LAYER_UNSPECIFIED",
            "error": "Daemon unavailable and in-memory engine fallback not found.",
        }


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "pokaż status cpu"
    client = NLClient()
    res = client.interpret(q)
    print(json.dumps(res, indent=2, ensure_ascii=False))
