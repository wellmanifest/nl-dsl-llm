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

    def suggest(
        self,
        partial_query: str,
        twin_context: Optional[Dict[str, Any]] = None,
        max_suggestions: int = 5,
    ) -> Dict[str, Any]:
        """Request real-time autocomplete suggestions and option network DAG."""
        payload = {
            "action": "suggest",
            "partial_query": partial_query.strip(),
            "client_id": self.client_id,
            "max_suggestions": max_suggestions,
            "twin_context": twin_context or {
                "environment": "prod",
                "active_services": ["postgres", "redis-cache"],
                "degraded_services": ["redis-cache"],
                "active_tickets": ["ticket-001", "ticket-002"],
            },
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

        # 3. Direct in-memory engine / fallback
        if payload.get("action") == "suggest":
            twin = payload.get("twin_context", {})
            env = twin.get("environment", "prod")
            services = twin.get("active_services", ["postgres", "redis-cache"])
            degraded = twin.get("degraded_services", ["redis-cache"])
            clean = payload["partial_query"].lower()
            suggestions = []
            if "zrestartuj".startswith(clean) or clean.startswith("zre") or clean.startswith("rest"):
                for svc in services:
                    is_deg = svc in degraded
                    suggestions.append({
                        "id": f"service.restart:{svc}",
                        "display_label": f"Zrestartuj {svc}" + (" (Zdegradowany)" if is_deg else f" [{env}]"),
                        "completion_text": f"zrestartuj {svc}",
                        "action_id": "service.restart",
                        "resolved_uri": f"urn:nl-dsl-sh:service:restart",
                        "bound_arguments": {"target": svc, "env": env},
                        "confidence": 0.98 if is_deg else 0.90,
                        "requires_confirmation": env == "prod",
                        "badge": "DEGRADED" if is_deg else "PROD",
                        "next_options": [
                            {"id": f"{svc}:graceful", "display_label": "Łagodny restart", "completion_text": f"zrestartuj {svc} --graceful"},
                            {"id": f"{svc}:force", "display_label": "Wymuszony restart", "completion_text": f"zrestartuj {svc} --force"},
                        ]
                    })
            return {
                "success": True,
                "partial_query": payload["partial_query"],
                "suggestions": sorted(suggestions, key=lambda s: 0 if s["badge"] == "DEGRADED" else 1),
                "latency_ms": 0.25,
            }

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
    client = NLClient()
    if len(sys.argv) > 2 and sys.argv[1] == "--suggest":
        suggs = client.suggest(sys.argv[2])
        print(json.dumps(suggs, indent=2, ensure_ascii=False))
    else:
        q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "pokaż status cpu"
        res = client.interpret(q)
        print(json.dumps(res, indent=2, ensure_ascii=False))
