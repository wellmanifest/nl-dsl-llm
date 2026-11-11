#!/usr/bin/env python3
"""
wellmanifest/nl-dsl-llm reference standard implementation.
Implements the 3-layer architecture:
  Layer 1: Deterministic NL Pattern Parser (Fast Path, PL & EN)
  Layer 2: Canonical DSL Engine (Single Execution Boundary & Validator)
  Layer 3: LLM Translation Fallback (Adaptive Path)
Plus interface parity adapters: CLI, Web REST API, MCP Server.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
try:
    from standard.nl_dsl_core import (
        CachedIntent,
        DigitalTwinContext,
        DSLCommand,
        DSLExecutor,
        DSLResult,
        NLIntentParser,
        OptionNetworkEngine,
        OptionNode,
        SemanticCache,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from standard.nl_dsl_core import (
        CachedIntent,
        DigitalTwinContext,
        DSLCommand,
        DSLExecutor,
        DSLResult,
        NLIntentParser,
        OptionNetworkEngine,
        OptionNode,
        SemanticCache,
    )

__all__ = [
    "CachedIntent",
    "DSLCommand",
    "DSLExecutor",
    "DSLResult",
    "DigitalTwinContext",
    "InterfaceAdapters",
    "LLMTranslator",
    "NLDSLLLMBridge",
    "NLIntentParser",
    "OptionNetworkEngine",
    "OptionNode",
    "SemanticCache",
    "run_self_test",
]


class LLMTranslator:
    """Adaptive semantic fallback compiler into canonical DSL statements."""

    def __init__(self, model_name: str = "default", api_client: Any = None) -> None:
        self.model_name = model_name
        self.api_client = api_client

    def translate(
        self, prompt: str, schema_hints: dict[str, Any] | None = None
    ) -> DSLCommand | None:
        if not isinstance(prompt, str):
            return None
        clean = prompt.strip()
        if not clean:
            return None
        lower = clean.lower()
        extra_args = dict(schema_hints) if schema_hints else {}
        if "przetestuj" in lower or "test" in lower:
            return DSLCommand(entity="test", operation="run", arguments={"query": clean, **extra_args})
        if "statystyki" in lower or "metrics" in lower:
            return DSLCommand(entity="metrics", operation="list", arguments={"subject": clean, **extra_args})
        return None


class NLDSLLLMBridge:
    """Coordinates Option Network, Fast Path, Semantic Cache, LLM Fallback, and DSL Executor."""

    def __init__(
        self,
        executor: DSLExecutor,
        parser: NLIntentParser | None = None,
        translator: LLMTranslator | None = None,
        cache: SemanticCache | None = None,
        option_engine: OptionNetworkEngine | None = None,
    ) -> None:
        self.executor = executor
        self.parser = parser or NLIntentParser()
        self.translator = translator or LLMTranslator()
        self.cache = cache or SemanticCache()
        self.option_engine = option_engine or OptionNetworkEngine()

    def suggest_options(
        self,
        partial_query: str,
        twin_context: DigitalTwinContext | None = None,
        max_suggestions: int = 5,
    ) -> dict[str, Any]:
        """Real-time autocomplete & contextual option network for text/voice input."""
        start = time.perf_counter()
        options = self.option_engine.suggest(
            partial_query, twin=twin_context, max_suggestions=max_suggestions
        )
        elapsed = (time.perf_counter() - start) * 1000
        return {
            "success": True,
            "partial_query": partial_query,
            "suggestions": [opt.to_dict() for opt in options],
            "latency_ms": round(elapsed, 3),
        }

    def handle_request(self, input_text: str, allow_llm_fallback: bool = True) -> DSLResult:
        if not isinstance(input_text, str) or not input_text.strip():
            return DSLResult(
                success=False,
                status="VALIDATION_ERROR",
                data=None,
                errors=[{"code": "NL_EMPTY_QUERY", "message": "Input query must be a non-empty string"}],
                meta={
                    "executionTimeMs": 0.0,
                    "sourceLayer": "nl_fast_path",
                    "canonicalDsl": None,
                    "traceId": str(uuid.uuid4())[:8],
                },
            )

        cmd = self.parser.parse(input_text)
        if cmd:
            source = "direct_dsl" if cmd.raw_dsl else "nl_fast_path"
            return self.executor.execute(cmd, source_layer=source)

        cached_result = self.cache.match(input_text)
        if cached_result:
            matched_cmd, _score = cached_result
            return self.executor.execute(matched_cmd, source_layer="semantic_cache")

        if allow_llm_fallback is True:
            fallback_cmd = self.translator.translate(input_text)
            if fallback_cmd:
                return self.executor.execute(fallback_cmd, source_layer="llm_fallback")

        return DSLResult(
            success=False,
            status="VALIDATION_ERROR",
            data=None,
            errors=[
                {
                    "code": "NL_UNRECOGNIZED_INTENT",
                    "message": f"Could not parse natural language query into valid DSL: '{input_text}'",
                }
            ],
            meta={
                "executionTimeMs": 0.0,
                "sourceLayer": "nl_fast_path",
                "canonicalDsl": None,
                "traceId": str(uuid.uuid4())[:8],
            },
        )


class InterfaceAdapters:
    """Unified adapters for CLI Shell, Web REST API, and Model Context Protocol."""

    @staticmethod
    def cli_format(result: DSLResult, output_format: str = "text") -> str:
        """Format DSLResult for CLI display."""
        if output_format == "json":
            return json.dumps(result.to_dict(), indent=2, ensure_ascii=False)

        if not result.success:
            errs = "; ".join(e.get("message", "") for e in result.errors)
            return f"Error ({result.status}): {errs}"

        if output_format == "markdown":
            lines = [f"### Result ({result.status})"]
            if isinstance(result.data, list):
                if not result.data:
                    lines.append("*Empty list.*")
                else:
                    keys = list(result.data[0].keys()) if isinstance(result.data[0], dict) else ["Value"]
                    lines.append("| " + " | ".join(keys) + " |")
                    lines.append("| " + " | ".join(["---"] * len(keys)) + " |")
                    for row in result.data:
                        if isinstance(row, dict):
                            lines.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
                        else:
                            lines.append(f"| {row} |")
            else:
                lines.append(f"```json\n{json.dumps(result.data, indent=2, ensure_ascii=False)}\n```")
            return "\n".join(lines)

        if isinstance(result.data, list):
            if not result.data:
                return "No items found."
            return json.dumps(result.data, indent=2, ensure_ascii=False)
        return str(result.data)

    @staticmethod
    def mcp_tools_manifest() -> list[dict[str, Any]]:
        """Declare standard MCP tools for agents."""
        return [
            {
                "name": "nl_ask",
                "description": "Ask a natural language query (PL or EN); parses into canonical DSL and executes safely.",
                "inputSchema": {
                    "type": "object",
                    "required": ["query"],
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language query, e.g. 'pokaż otwarte zadania' or 'list tickets'",
                        },
                        "allow_llm_fallback": {"type": "boolean", "default": True},
                    },
                },
            },
            {
                "name": "execute_dsl",
                "description": "Execute a canonical domain DSL statement directly.",
                "inputSchema": {
                    "type": "object",
                    "required": ["dsl_command"],
                    "properties": {
                        "dsl_command": {
                            "type": "string",
                            "description": "Canonical DSL string, e.g. 'ticket.list status=OPEN'",
                        }
                    },
                },
            },
        ]

    @staticmethod
    def handle_mcp_tool_call(
        bridge: NLDSLLLMBridge, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Dispatch MCP tool execution with strict input validation."""
        invalid_args = not isinstance(arguments, dict)
        if invalid_args:
            raise ValueError("MCP tool arguments must be a JSON object")
        if tool_name == "nl_ask":
            query = arguments.get("query")
            if not isinstance(query, str) or not query.strip():
                raise ValueError("nl_ask requires a non-empty string 'query'")
            allow_fallback = arguments.get("allow_llm_fallback", True)
            if not isinstance(allow_fallback, bool):
                raise ValueError("allow_llm_fallback must be a boolean")
            res = bridge.handle_request(query, allow_llm_fallback=allow_fallback)
            return {"content": [{"type": "text", "text": json.dumps(res.to_dict(), indent=2)}]}
        if tool_name == "execute_dsl":
            dsl_text = arguments.get("dsl_command")
            if not isinstance(dsl_text, str) or not dsl_text.strip():
                raise ValueError("execute_dsl requires a non-empty string 'dsl_command'")
            cmd = DSLCommand.from_text(dsl_text)
            res = bridge.executor.execute(cmd, source_layer="direct_dsl")
            return {"content": [{"type": "text", "text": json.dumps(res.to_dict(), indent=2)}]}
        raise ValueError(f"Unknown tool: {tool_name}")


def _emit_self_test(line: str = "") -> None:
    sys.stdout.write(f"{line}\n")
    sys.stdout.flush()


def run_self_test() -> int:
    """Execute normative self-tests verifying the standard implementation."""
    _emit_self_test("Running wellmanifest/nl-dsl-llm standard conformance self-test...")

    executor = DSLExecutor()
    tickets = [
        {"id": "001", "title": "Setup repository", "status": "CLOSED"},
        {"id": "002", "title": "Add NL-DSL-LLM parser", "status": "OPEN"},
        {"id": "003", "title": "Deploy MCP server", "status": "OPEN"},
    ]

    def handle_ticket_list(cmd: DSLCommand) -> list[dict[str, str]]:
        status = cmd.get_param("status")
        return [t for t in tickets if t["status"] == status] if status else list(tickets)

    def handle_ticket_create(cmd: DSLCommand) -> dict[str, str]:
        item = {"id": f"{len(tickets) + 1:03d}", "title": cmd.arguments.get("title", "Untitled"), "status": "OPEN"}
        tickets.append(item)
        return item

    executor.register("ticket", "list", handle_ticket_list)
    executor.register("ticket", "create", handle_ticket_create)
    bridge = NLDSLLLMBridge(executor)

    res1 = bridge.handle_request("ticket.list status=OPEN")
    assert res1.success is True and len(res1.data) == 2 and res1.meta["sourceLayer"] == "direct_dsl"
    _emit_self_test("✓ Direct DSL execution verified")

    res2 = bridge.handle_request("pokaż otwarte zadania")
    assert res2.success is True and len(res2.data) == 2 and res2.meta["sourceLayer"] == "nl_fast_path"
    _emit_self_test("✓ Layer 1 Polish query ('pokaż otwarte zadania') verified")

    res3 = bridge.handle_request("show open tickets")
    assert res3.success is True and len(res3.data) == 2 and res3.meta["sourceLayer"] == "nl_fast_path"
    _emit_self_test("✓ Layer 1 English query ('show open tickets') verified")

    res4 = bridge.handle_request("dodaj zadanie 'Implement tests'")
    assert res4.success is True and res4.data["title"] == "Implement tests"
    _emit_self_test("✓ Layer 1 Creation ('dodaj zadanie ...') verified")

    mcp_call = InterfaceAdapters.handle_mcp_tool_call(bridge, "nl_ask", {"query": "wypisz tickety"})
    mcp_res = json.loads(mcp_call["content"][0]["text"])
    assert mcp_res["success"] is True and len(mcp_res["data"]) == 4
    _emit_self_test("✓ Model Context Protocol (MCP) tool call parity verified")

    md_output = InterfaceAdapters.cli_format(res1, output_format="markdown")
    assert "### Result (OK)" in md_output and "| id | title | status |" in md_output
    _emit_self_test("✓ CLI formatting (markdown table) verified")

    executor.register("cluster", "status", lambda cmd: {"cluster": "ready", "env": cmd.get_param("env", "prod")})
    bridge.cache.register_template(
        "stan klastra produkcyjnego",
        DSLCommand(entity="cluster", operation="status", filters={"env": "prod"}),
        slots={"env": "prod"},
    )
    res_cache = bridge.handle_request("stan klastra produkcyjnego")
    assert res_cache.success is True and res_cache.meta["sourceLayer"] == "semantic_cache"
    res_neg = bridge.handle_request("nie sprawdzaj stanu klastra produkcyjnego", allow_llm_fallback=False)
    assert res_neg.success is False and res_neg.status == "VALIDATION_ERROR"
    _emit_self_test("✓ Layer 1.5 Semantic Cache and Polarity Guard verified")

    twin = DigitalTwinContext(
        environment="prod",
        active_services=["postgres", "redis-cache"],
        degraded_services=["redis-cache"],
        active_tickets=["ticket-001", "ticket-002"],
    )
    sugg_resp = bridge.suggest_options("zre", twin_context=twin)
    assert sugg_resp["success"] is True and len(sugg_resp["suggestions"]) >= 2
    assert sugg_resp["suggestions"][0]["bound_arguments"]["target"] == "redis-cache"
    assert sugg_resp["suggestions"][0]["badge"] == "DEGRADED"
    assert sugg_resp["suggestions"][0]["requires_confirmation"] is True
    assert "--graceful" in sugg_resp["suggestions"][0]["next_options"][0]["completion_text"]
    assert sugg_resp["latency_ms"] < 10.0
    _emit_self_test("✓ Layer 0.5 Real-time Autocomplete & Option Network verified")

    _emit_self_test("\nALL STANDARD CONFORMANCE CHECKS PASSED (8/8).")
    return 0


if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser(description="wellmanifest/nl-dsl-llm Reference Tool")
    cli_parser.add_argument("--self-test", action="store_true", help="Execute built-in conformance self-test suite")
    cli_args = cli_parser.parse_args()
    if cli_args.self_test:
        sys.exit(run_self_test())
    cli_parser.print_help()
    sys.exit(0)
