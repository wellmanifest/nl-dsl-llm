"""
Pytest test suite for wellmanifest/nl-dsl-llm standard conformance.
"""

import json
from standard.nl_dsl_llm import (
    DSLCommand,
    DSLExecutor,
    DigitalTwinContext,
    InterfaceAdapters,
    NLDSLLLMBridge,
    NLIntentParser,
)


def create_test_fixture():
    executor = DSLExecutor()
    store = [
        {"id": "t-1", "title": "Setup repository", "status": "CLOSED"},
        {"id": "t-2", "title": "Implement NL parser", "status": "OPEN"},
        {"id": "t-3", "title": "Configure MCP server", "status": "OPEN"},
    ]

    def list_tickets(cmd: DSLCommand):
        status = cmd.get_param("status")
        if status:
            return [t for t in store if t["status"] == status]
        return list(store)

    def create_ticket(cmd: DSLCommand):
        new_item = {
            "id": f"t-{len(store) + 1}",
            "title": cmd.arguments.get("title", "Untitled"),
            "status": "OPEN",
        }
        store.append(new_item)
        return new_item

    executor.register("ticket", "list", list_tickets)
    executor.register("ticket", "create", create_ticket)
    bridge = NLDSLLLMBridge(executor)
    return bridge, store


def test_direct_dsl_execution():
    bridge, _ = create_test_fixture()
    res = bridge.handle_request("ticket.list status=OPEN")
    assert res.success is True
    assert res.status == "OK"
    assert len(res.data) == 2
    assert res.meta["sourceLayer"] == "direct_dsl"


def test_polish_nl_fast_path():
    bridge, _ = create_test_fixture()
    queries = [
        "pokaż otwarte zadania",
        "wypisz otwarte tickety",
        "wyświetl zadania",
    ]
    for q in queries:
        res = bridge.handle_request(q)
        assert res.success is True
        assert res.meta["sourceLayer"] == "nl_fast_path"


def test_english_nl_fast_path():
    bridge, _ = create_test_fixture()
    queries = [
        "show open tickets",
        "list open tasks",
        "display tickets",
    ]
    for q in queries:
        res = bridge.handle_request(q)
        assert res.success is True
        assert res.meta["sourceLayer"] == "nl_fast_path"


def test_creation_via_nl():
    bridge, store = create_test_fixture()
    initial_count = len(store)
    res = bridge.handle_request("dodaj zadanie 'Test standard conformance'")
    assert res.success is True
    assert res.data["title"] == "Test standard conformance"
    assert len(store) == initial_count + 1


def test_mcp_tool_parity():
    bridge, _ = create_test_fixture()
    call = InterfaceAdapters.handle_mcp_tool_call(
        bridge, "nl_ask", {"query": "pokaż zadania"}
    )
    payload = json.loads(call["content"][0]["text"])
    assert payload["success"] is True
    assert len(payload["data"]) >= 3


def test_cli_adapter_formats():
    bridge, _ = create_test_fixture()
    res = bridge.handle_request("ticket.list")
    json_out = InterfaceAdapters.cli_format(res, output_format="json")
    parsed = json.loads(json_out)
    assert parsed["success"] is True

    md_out = InterfaceAdapters.cli_format(res, output_format="markdown")
    assert "### Result (OK)" in md_out
    assert "| id | title | status |" in md_out


def test_semantic_cache_polarity_guard():
    bridge, _ = create_test_fixture()
    bridge.executor.register(
        "cluster", "status", lambda cmd: {"cluster": "ready", "env": cmd.get_param("env", "prod")}
    )
    bridge.cache.register_template(
        "stan klastra produkcyjnego",
        DSLCommand(entity="cluster", operation="status", filters={"env": "prod"}),
        slots={"env": "prod"},
    )

    # Positive match
    res_pos = bridge.handle_request("stan klastra produkcyjnego")
    assert res_pos.success is True
    assert res_pos.meta["sourceLayer"] == "semantic_cache"

    # Inverted query rejected by polarity guard
    res_neg = bridge.handle_request("nie sprawdzaj stanu klastra produkcyjnego", allow_llm_fallback=False)
    assert res_neg.success is False
    assert res_neg.status == "VALIDATION_ERROR"


def test_option_network_digital_twin_projection():
    bridge, _ = create_test_fixture()
    twin = DigitalTwinContext(
        environment="prod",
        active_services=["postgres", "redis-cache"],
        degraded_services=["redis-cache"],
        active_tickets=["ticket-001"],
    )

    # Suggestion on partial token
    res = bridge.suggest_options("zre", twin_context=twin)
    assert res["success"] is True
    suggestions = res["suggestions"]
    assert len(suggestions) >= 2

    # Degraded entity prioritized
    first = suggestions[0]
    assert first["bound_arguments"]["target"] == "redis-cache"
    assert first["badge"] == "DEGRADED"
    assert first["requires_confirmation"] is True
    assert len(first["next_options"]) >= 2
    assert "--graceful" in first["next_options"][0]["completion_text"]
    assert res["latency_ms"] < 10.0


