# Adoption Case Study: semcod/planfile (NL-DSL-LLM Pattern)

This document provides a concrete reference implementation walkthrough of the **NL-DSL-LLM** pattern adopted in `semcod/planfile`.

---

## 1. Architectural Mapping

In `semcod/planfile`, the three-layer pattern is mapped as follows:

| Layer | Component | Description |
|---|---|---|
| **Layer 1: Fast-Path NL Parser** | `planfile/dsl/parser.py` | Multilingual (PL/EN) deterministic regex and keyword parser. Latency < 1ms, 0 external calls. |
| **Layer 2: Canonical DSL Engine** | `planfile/dsl/executor.py` (`DSLExecutor`) | Validates and executes `DSLCommand(verb, target, params)` against the Planfile store. |
| **Layer 3: Adaptive LLM Fallback** | `DSLExecutor._translate_with_llm` | Translates unrecognized natural language into canonical DSL grammar using SubLLM/LiteLLM. |
| **Interface Parity** | CLI (`planfile ask`), REST API (`POST /query`), MCP Server (`planfile_ask`) | Exposes identical functionality across developer terminal, web services, and AI agents. |

---

## 2. Layer 1: Multilingual Keyword & Regex Dictionary

The parser maintains deterministic synonym maps for verbs, nouns, and attribute filters:

```python
# planfile/dsl/parser.py

VERBS = {
    # Polish
    "pokaz": "show", "pokaż": "show", "wyswietl": "show", "wyświetl": "show", "wypisz": "show",
    "utworz": "create", "utwórz": "create", "dodaj": "create", "nowy": "create", "nowa": "create",
    "wyczysc": "clear", "wyczyść": "clear", "usun": "delete", "usuń": "delete", "skasuj": "delete",
    "edytuj": "edit", "zmien": "edit", "zmień": "edit", "ustaw": "edit",
    "znajdz": "find", "znajdź": "find", "szukaj": "find",
    "waliduj": "validate", "sprawdz": "validate", "sprawdź": "validate",
    # English
    "show": "show", "list": "show", "get": "show", "ls": "show",
    "create": "create", "add": "create", "new": "create",
    "delete": "delete", "remove": "delete", "rm": "delete", "clear": "clear",
    "edit": "edit", "update": "edit", "set": "edit", "modify": "edit",
    "find": "find", "search": "find", "grep": "find",
    "validate": "validate", "check": "validate", "lint": "validate",
}

STATUS_MAP = {
    "otwarte": "todo", "otwarty": "todo", "otwarta": "todo", "open": "todo", "todo": "todo",
    "zamkniete": "done", "zamknięte": "done", "zamkniety": "done", "closed": "done", "done": "done",
    "w toku": "in_progress", "wtoku": "in_progress", "trwajace": "in_progress", "in_progress": "in_progress",
    "zablokowane": "blocked", "blocked": "blocked",
}
```

### Fast-Path Regex Rules

Simple regex matches convert natural language requests directly into a typed `DSLCommand`:
- `"pokaż otwarte zadania"` / `"show open tickets"` -> `DSLCommand(verb="show", target="tickets", params={"status": "todo"})`
- `"dodaj zadanie Naprawić błąd"` / `"create ticket Fix bug"` -> `DSLCommand(verb="create", target="ticket", params={"title": "Fix bug"})`
- `"waliduj planfile"` / `"validate strategy"` -> `DSLCommand(verb="validate", target="planfile", params={})`

---

## 3. Layer 3: Controlled LLM Fallback

If Layer 1 fails to match any deterministic rule:

```python
# planfile/dsl/executor.py

class DSLExecutor:
    def run(self, input_text: str, allow_llm_fallback: bool = True) -> DSLResult:
        # 1. Attempt raw canonical DSL parsing
        cmd = parse_dsl_line(input_text)
        if cmd is not None:
            res = self.execute(cmd)
            res.source_layer = "direct_dsl"
            return res

        # 2. Attempt Layer 1 Fast-Path Regex
        cmd = parse_natural_language(input_text)
        if cmd is not None:
            res = self.execute(cmd)
            res.source_layer = "fast_path_regex"
            return res

        # 3. Layer 3 Adaptive LLM Fallback (if enabled)
        if allow_llm_fallback:
            translated = self._translate_with_llm(input_text)
            if translated:
                cmd = parse_dsl_line(translated)
                if cmd:
                    res = self.execute(cmd)
                    res.source_layer = "llm_fallback"
                    return res

        return DSLResult(
            ok=False,
            command={"raw": input_text},
            error=f"Unrecognized query: '{input_text}'. Run 'planfile dsl help' for grammar.",
            source_layer="rejected"
        )
```

The system prompt for translation explicitly provides the grammar and rejects arbitrary code execution.

---

## 4. Universal Interface Parity

### 4.1 CLI Shell (`planfile ask` and `planfile mcp`)

```bash
# Natural Language queries (PL/EN)
$ planfile ask "pokaż otwarte zadania" --format json
{
  "ok": true,
  "command": {"verb": "show", "target": "tickets", "params": {"status": "todo"}},
  "data": [...],
  "source_layer": "fast_path_regex"
}

$ planfile ask "create sprint Sprint-Alpha with 14 days"
✓ Created sprint 'Sprint-Alpha' (14 days)

# Run MCP Server over stdio for AI coding agents
$ planfile mcp --project-root .
```

### 4.2 Web REST API (`POST /query`)

```bash
$ curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "pokaż zamknięte tickety", "allow_llm_fallback": true}'
```

Response envelope:
```json
{
  "ok": true,
  "command": {
    "verb": "show",
    "target": "tickets",
    "params": {"status": "done"}
  },
  "data": [],
  "error": null,
  "message": "Found 0 tickets",
  "source_layer": "fast_path_regex"
}
```

### 4.3 Model Context Protocol (MCP Server)

AI agents (Cursor, Claude Desktop, Antigravity) consume the MCP server over stdio:
- `planfile_ask`: Natural language query execution.
- `planfile_describe_grammar`: Returns grammar reference for autonomous agent planning.

---

## 5. Conformance Verification

Test coverage guarantees all three layers and interfaces remain strictly conformant:
- `tests/test_nl_dsl_llm.py`
  - `test_layer1_fast_path_polish_and_english()`
  - `test_layer1_ticket_creation_pl_en()`
  - `test_layer1_sprint_management_pl_en()`
  - `test_cli_ask_command()`
  - `test_rest_api_query_endpoint()`
  - `test_mcp_tools()`
  - `test_layer3_fallback_disabled_rejects_unknown()`
