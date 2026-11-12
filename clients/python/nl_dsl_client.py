"""
Python client for paxlet-com/nl-dsl-llm runtime daemon.
Enables sub-millisecond IPC for nl-dsl-sh, Clonerd, and Python scripts.
"""

from __future__ import annotations

import json
import math
import os
import socket
import sys
from typing import Any

# Prevent .pyc pollution in git checkouts
sys.dont_write_bytecode = True

_client_dir = os.path.abspath(os.path.dirname(__file__))
if _client_dir not in sys.path:
    sys.path.insert(0, _client_dir)

# Auto-discover local src directory if not installed as system package
_repo_src = os.path.abspath(os.path.join(_client_dir, "..", "..", "src"))
if os.path.isdir(_repo_src) and _repo_src not in sys.path:
    sys.path.insert(0, _repo_src)

from nl_dsl_ipc import (
    _ASCII_INT_TOKEN_RE,
    _ASCII_TICKET_ID_RE,
    MAX_CLIENT_QUERY_CHARS,
    MAX_IPC_RESPONSE_BYTES,
    MAX_KORU_ITERATIONS,
    has_negation,
    is_safe_unix_socket,
    recv_json_line,
    run_client_self_test,
)


class NLClient:
    """Client for NL-DSL-LLM IPC daemon over Unix Domain Socket or TCP."""

    def __init__(
        self,
        socket_path: str | None = None,
        host: str | None = None,
        port: int = 50051,
        client_id: str = "nl-dsl-sh",
        timeout_seconds: float = 5.0,
    ):
        raw_port = os.environ.get("NL_DSL_PORT")
        resolved_port = int(raw_port) if raw_port is not None else port
        if (
            isinstance(resolved_port, bool)
            or not isinstance(resolved_port, int)
            or not (1 <= resolved_port <= 65535)
        ):
            raise ValueError("port must be an integer between 1 and 65535")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(float(timeout_seconds))
            or float(timeout_seconds) <= 0.0
        ):
            raise ValueError("timeout_seconds must be a positive finite number")
        self.socket_path = socket_path or os.environ.get(
            "NL_DSL_SOCKET", "/run/paxlet/nl-dsl-llm.sock"
        )
        self.host = host if host is not None else os.environ.get("NL_DSL_HOST")
        self.port = resolved_port
        self.client_id = client_id
        self.timeout = float(timeout_seconds)
        self._fallback_engine: Any = None

    def interpret(
        self,
        query: str,
        allow_fallback: bool = True,
        model: str = "qwen_2b",
        locale: str = "pl",
    ) -> dict[str, Any]:
        """Send query to daemon for intent interpretation."""
        if (
            not isinstance(query, str)
            or not query.strip()
            or len(query) > MAX_CLIENT_QUERY_CHARS
        ):
            return {
                "status": "CLARIFY",
                "action_id": "",
                "arguments": {},
                "resolved_procedure_uri": "",
                "matched_layer": "SOURCE_LAYER_UNSPECIFIED",
                "error": "Query empty or exceeded maximum length",
            }
        payload = {
            "action": "interpret",
            "query": query.strip(),
            "client_id": self.client_id,
            "locale": locale,
            "allow_llm_fallback": bool(allow_fallback),
            "model_target": model,
        }
        return self._send_request(payload)

    def execute(
        self,
        action_id: str,
        arguments: dict[str, Any] | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Execute a validated action via the daemon execution boundary."""
        if arguments is not None and not isinstance(arguments, dict):
            return {
                "success": False,
                "error_code": "INVALID_ARGUMENTS",
                "message": "arguments must be a dict",
            }
        payload = {
            "action": "execute",
            "action_id": action_id,
            "arguments": arguments or {},
            "dry_run": bool(dry_run),
            "client_id": self.client_id,
        }
        return self._send_request(payload)

    def suggest(
        self,
        partial_query: str,
        twin_context: dict[str, Any] | None = None,
        max_suggestions: int = 5,
    ) -> dict[str, Any]:
        """Request real-time autocomplete suggestions and option network DAG."""
        if (
            not isinstance(partial_query, str)
            or len(partial_query) > MAX_CLIENT_QUERY_CHARS
        ):
            return {
                "success": False,
                "partial_query": "",
                "suggestions": [],
                "latency_ms": 0.0,
            }
        capped_suggestions = max(1, min(int(max_suggestions), 50))
        payload = {
            "action": "suggest",
            "partial_query": partial_query.strip(),
            "client_id": self.client_id,
            "max_suggestions": capped_suggestions,
            "twin_context": twin_context
            or {
                "environment": "prod",
                "active_services": ["postgres", "redis-cache"],
                "degraded_services": ["redis-cache"],
                "active_tickets": ["ticket-001", "ticket-002"],
            },
        }
        return self._send_request(payload)

    def _get_fallback_engine(self) -> Any:
        if self._fallback_engine is None:
            from paxlet_nl_dsl_llm.server import NLRuntimeEngine

            self._fallback_engine = NLRuntimeEngine()
        return self._fallback_engine

    def _send_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        wire_frame = (json.dumps(payload, allow_nan=False) + "\n").encode("utf-8")

        if is_safe_unix_socket(self.socket_path):
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                    sock.settimeout(self.timeout)
                    sock.connect(self.socket_path)
                    sock.sendall(wire_frame)
                    return recv_json_line(sock, MAX_IPC_RESPONSE_BYTES)
            except (OSError, ValueError, TypeError):
                pass

        if self.host:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.settimeout(self.timeout)
                    sock.connect((self.host, self.port))
                    sock.sendall(wire_frame)
                    return recv_json_line(sock, MAX_IPC_RESPONSE_BYTES)
            except (OSError, ValueError, TypeError):
                pass

        if payload.get("action") == "suggest":
            twin = (
                payload.get("twin_context")
                if isinstance(payload.get("twin_context"), dict)
                else {}
            )
            env = twin.get("environment", "prod")
            services = twin.get("active_services", ["postgres", "redis-cache"])
            degraded = twin.get("degraded_services", ["redis-cache"])
            clean = str(payload.get("partial_query", "")).lower()
            suggestions = []
            if (
                clean
                and not has_negation(clean)
                and (
                    "zrestartuj".startswith(clean) or clean.startswith(("zre", "rest"))
                )
            ):
                for svc in services[: payload.get("max_suggestions", 5)]:
                    is_deg = svc in degraded
                    suggestions.append(
                        {
                            "id": f"service.restart:{svc}",
                            "display_label": f"Zrestartuj {svc}"
                            + (" (Zdegradowany)" if is_deg else f" [{env}]"),
                            "completion_text": f"zrestartuj {svc}",
                            "action_id": "service.restart",
                            "resolved_uri": "urn:nl-dsl-sh:service:restart",
                            "bound_arguments": {"target": svc, "env": env},
                            "confidence": 0.98 if is_deg else 0.90,
                            "requires_confirmation": env == "prod",
                            "badge": "DEGRADED" if is_deg else "PROD",
                            "next_options": [
                                {
                                    "id": f"{svc}:graceful",
                                    "display_label": "Łagodny restart",
                                    "completion_text": f"zrestartuj {svc} --graceful",
                                },
                                {
                                    "id": f"{svc}:force",
                                    "display_label": "Wymuszony restart",
                                    "completion_text": f"zrestartuj {svc} --force",
                                },
                            ],
                        }
                    )
            return {
                "success": True,
                "partial_query": payload.get("partial_query", ""),
                "suggestions": sorted(
                    suggestions, key=lambda s: 0 if s["badge"] == "DEGRADED" else 1
                ),
                "latency_ms": 0.25,
            }

        try:
            engine = self._get_fallback_engine()
            if payload.get("action") == "interpret":
                return engine.interpret(
                    payload["query"],
                    client_id=self.client_id,
                    allow_llm_fallback=payload.get("allow_llm_fallback", True),
                    model_target=payload.get("model_target", "qwen_2b"),
                )
            if payload.get("action") == "execute":
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


class KoruNLClient(NLClient):
    """Specialized client for Koru closed-loop automation and planfile queue control."""

    def __init__(self, **kwargs: Any):
        kwargs.setdefault("client_id", "koru")
        super().__init__(**kwargs)

    def dispatch_natural_command(self, natural_command: str) -> dict[str, Any]:
        """Translate a natural language developer command to canonical koru CLI execution parameters."""
        if (
            not isinstance(natural_command, str)
            or not natural_command.strip()
            or len(natural_command) > MAX_CLIENT_QUERY_CHARS
        ):
            return {
                "success": False,
                "status": "CLARIFY",
                "action": "",
                "cli_command": "",
                "reason": "Command empty or exceeded maximum length",
            }
        clean = natural_command.lower().strip()
        if has_negation(clean):
            return {
                "success": False,
                "status": "CLARIFY",
                "action": "",
                "cli_command": "",
                "reason": "Negated command blocked by client polarity guard",
            }
        if (
            "diagnostyk" in clean
            or "doctor" in clean
            or "napraw" in clean
            or "sprawdź projekt" in clean
        ):
            parts = ["koru", "--doctor"]
            if "napraw" in clean:
                parts.append("--repair")
            parts.extend(["--project", "."])
            return {
                "success": True,
                "action": "koru.doctor",
                "cli_command": " ".join(parts),
                "resolved_uri": "urn:koru:diagnostics:doctor",
                "confidence": 1.0,
            }
        if (
            "następne" in clean
            or "weź zadanie" in clean
            or "kolejka" in clean
            or "next" in clean
        ):
            queue = "development" if "dev" in clean else "default"
            return {
                "success": True,
                "action": "koru.queue.next",
                "cli_command": f"koru --queue --queue-name {queue}",
                "resolved_uri": f"urn:koru:queue:{queue}:next",
                "confidence": 1.0,
            }
        if "pętl" in clean or "loop" in clean or "automatycz" in clean:
            num = _ASCII_INT_TOKEN_RE.search(clean)
            if num is None:
                max_iter = 10
            elif len(num.group(1)) > 6:
                max_iter = MAX_KORU_ITERATIONS
            else:
                max_iter = max(1, min(int(num.group(1)), MAX_KORU_ITERATIONS))
            return {
                "success": True,
                "action": "koru.queue.loop",
                "cli_command": f"koru --loop --max-iterations {max_iter}",
                "resolved_uri": "urn:koru:queue:loop",
                "confidence": 0.95,
            }
        if "kontekst" in clean or "context" in clean or "brief" in clean:
            match = _ASCII_TICKET_ID_RE.search(clean)
            t_id = match.group(1).upper() if match else "NEXT"
            return {
                "success": True,
                "action": "koru.context",
                "cli_command": f"koru --context --ticket {t_id} --format json",
                "resolved_uri": f"urn:koru:context:{t_id}",
                "confidence": 0.95,
            }
        return self.interpret(natural_command)


def run_self_test() -> None:
    run_client_self_test(NLClient, KoruNLClient)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        run_self_test()
    elif len(sys.argv) > 2 and sys.argv[1] == "--suggest":
        client = NLClient()
        suggs = client.suggest(sys.argv[2])
        print(json.dumps(suggs, indent=2, ensure_ascii=False))
    else:
        client = NLClient()
        q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "pokaż status cpu"
        res = client.interpret(q)
        print(json.dumps(res, indent=2, ensure_ascii=False))
