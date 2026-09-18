# Wellmanifest Standard: NL-DSL-LLM Architecture Specification

- **Standard ID**: `wellmanifest/nl-dsl-llm`
- **Version**: `0.1.0`
- **Status**: `Draft / Candidate`
- **Authors**: Wellmanifest Architecture Workgroup

---

## 1. Executive Summary & Architectural Motivation

Modern agentic and developer tooling faces a fundamental trade-off:
1. **Natural Language (NL) Interfaces**: Flexible, accessible, and intuitive for humans and conversational AI agents, but nondeterministic, prone to hallucinations, and difficult to audit or sandbox safely.
2. **Domain-Specific Languages (DSL)**: Strictly typed, deterministic, parseable, transactionally sound, and auditable, but steep learning curves and rigid syntax make them cumbersome for ad-hoc natural queries.
3. **Direct LLM Tool Execution**: Calling APIs or tools directly from LLM completions introduces security vulnerabilities, unvetted parameter passing, and unbounded API costs.

The **NL-DSL-LLM Pattern** resolves this trade-off by establishing a layered, fail-closed architecture where:
- **DSL is the sole execution surface and single source of truth (SSOT).**
- **Natural Language is an intake boundary, never an executor.**
- **Rule-based parsing handles 90%+ of standard operations with 0ms latency and 0 token cost.**
- **LLM translation serves exclusively as an adaptive fallback compiler** when rule-based patterns do not match.
- **Interface Parity guarantees identical semantic behavior across CLI, Web REST API, and Model Context Protocol (MCP).**

```
                     ┌─────────────────────────────────────────┐
                     │           Incoming Requests             │
                     │  (CLI Shell / Web REST API / MCP Tool)  │
                     └────────────────────┬────────────────────┘
                                          │
                        Is input canonical DSL or NL?
                                          │
                      ┌───────────────────┴───────────────────┐
                      │ NL Query                              │ Raw DSL
                      ▼                                       │
         ┌─────────────────────────┐                          │
         │ Layer 1: Rule-Based NL  │                          │
         │ Fast-Path Parser        │                          │
         │ (PL/EN Synonyms & Regex)│                          │
         └────────────┬────────────┘                          │
                      │                                       │
              Pattern matched?                                │
              ├── YES ──────────────────┐                     │
              │                         ▼                     │
              │             ┌───────────────────────┐         │
              └── NO ──────►│ Layer 3: LLM Fallback │         │
                            │ Translation Compiler  │         │
                            │ (NL -> Canonical DSL) │         │
                            └───────────┬───────────┘         │
                                        │                     │
                                        ▼                     ▼
                             ┌───────────────────────────────────┐
                             │ Layer 2: Canonical DSL Engine     │
                             │ - Schema Validation               │
                             │ - Fencing & Permission Check      │
                             │ - Transactional Execution         │
                             │ - Audit & Event Log Emission      │
                             └──────────────────┬────────────────┘
                                                │
                                                ▼
                             ┌───────────────────────────────────┐
                             │ Unified Response Envelope         │
                             │ (JSON / Markdown / Plaintext)     │
                             └───────────────────────────────────┘
```

---

## 2. Core Architectural Layers

### 2.1 Layer 1: Deterministic NL Pattern Parser (Fast Path)

The deterministic NL parser is a localized, rule-driven intake engine that translates recognized linguistic patterns into canonical DSL structures without network requests or model inference.

- **Characteristics**:
  - **Latency**: Sub-millisecond (< 1ms).
  - **Cost**: Zero tokens, zero external API fees.
  - **Availability**: 100% offline and air-gapped capable.
  - **Deterministic**: Given identical string inputs, produces identical DSL commands.
- **Multilingual Normalization**:
  - Requires support for both Polish (PL) and English (EN) root verbs and entities.
  - Case-insensitive, whitespace-tolerant, punctuation-stripped.
  - Verb Synonym Table mapping:
    - *Query/Read*: `pokaż`, `wypisz`, `wyświetl`, `pobierz`, `sprawdź`, `znajdź`, `show`, `list`, `view`, `get`, `query`, `search`.
    - *Create/Add*: `dodaj`, `utwórz`, `stwórz`, `nowy`, `zarejestruj`, `add`, `create`, `new`, `register`.
    - *Update/Modify*: `zmień`, `ustaw`, `edytuj`, `zaktualizuj`, `update`, `set`, `edit`, `modify`.
    - *Close/Complete*: `zamknij`, `zakończ`, `oznacz jako zrobione`, `gotowe`, `done`, `close`, `finish`, `complete`.
    - *Delete/Remove*: `usuń`, `skasuj`, `odrzuć`, `delete`, `remove`, `drop`.
- **Confidence Scoring**:
  - Pattern matches emit a confidence rating (`1.0` for full grammar match, `0.0` for unrecognized).
  - Matches with confidence `< 1.0` or syntax ambiguities fall through to Layer 3.

### 2.2 Layer 2: Canonical DSL Engine (Single Execution Boundary)

The DSL is the **sole executable representation** within the system. No agent, user, or LLM may invoke internal business logic, mutations, or mutations without constructing a validated DSL command envelope.

- **Principles**:
  - **Single Source of Truth**: All operations (CRUD, queries, workflow transitions) are formally declared DSL commands.
  - **Transactional & Idempotent**: Where supported, commands accept idempotency keys or leases.
  - **Strict Schema Enforcement**: Every command is validated against a normative JSON Schema before handler dispatch.
  - **Execution Envelope**: Every execution returns a standardized envelope:
    ```json
    {
      "success": true,
      "command": "ticket.list",
      "status": "COMPLETED",
      "data": [...],
      "errors": [],
      "meta": {
        "executionTimeMs": 1.42,
        "traceId": "req-98fbc102"
      }
    }
    ```

### 2.3 Layer 3: LLM Translation Fallback (Adaptive Path)

When Layer 1 fails to match an incoming natural language prompt with sufficient confidence, Layer 3 acts as an **adaptive semantic compiler**.

- **Fail-Closed Boundary**:
  - The LLM **never** receives direct execution permissions, tools, or shell access.
  - The LLM's **sole task** is translating the conversational prompt into one or more canonical DSL statements.
- **Strict Prompt Specification**:
  - The system prompt presents only the formal DSL grammar and schemas.
  - Expected completion format is strict JSON adhering to `schemas/dsl-command.schema.json` or plain canonical DSL statements.
  - Free-form prose explanation is discarded or confined to an advisory metadata field.
- **Post-Translation Validation**:
  - The generated DSL output from Layer 3 is fed back into Layer 2 for standard schema validation.
  - If the LLM generates an invalid command or illegal parameters, execution fails gracefully with clear validation diagnostics.

---

## 3. Unified Interface Parity

Every adopting system must provide three synchronous interface surfaces sharing the identical NL-DSL-LLM engine:

### 3.1 CLI Shell (`<tool> shell`, `<tool> ask`, `<tool> dsl`)
- Interactive REPL and command-line entry point.
- Single command execution:
  - `<tool> ask "pokaż otwarte zadania"` -> NL parser -> DSL -> output.
  - `<tool> dsl "task.list status=OPEN"` -> direct DSL execution -> output.
- Formats: Pretty terminal table (default), JSON (`--json`), Markdown (`--markdown`).

### 3.2 Web REST API (`/dsl`, `/query`, `/schema`)
- Standardized HTTP endpoints:
  - `POST /api/v1/dsl`: Accepts canonical DSL strings or structured command envelopes.
  - `POST /api/v1/query`: Accepts natural language query strings with optional locale (`pl`, `en`).
  - `GET /api/v1/schema`: Introspects supported DSL entities, operations, and grammar.
- Uniform HTTP status codes and standard execution envelopes in response bodies.

### 3.3 Model Context Protocol (MCP Server)
- Standardized integration for AI agents (Cursor, Claude Desktop, Antigravity, OpenCode).
- Exposed MCP Tools:
  - `execute_dsl`: Executes validated canonical DSL statements.
  - `nl_ask`: Resolves natural language queries (Layer 1 + Layer 3) and executes resulting DSL.
  - `describe_grammar`: Returns current DSL vocabulary, entities, operations, and examples.
- Exposed MCP Resources:
  - `schema://current`: Full JSON Schema of domain DSL and operations.

---

## 4. Conformance Criteria & Checkpoints

A repository or service claiming compliance with `wellmanifest/nl-dsl-llm` must satisfy:

1. **[CONF-01] Multilingual Fast Path**: Implements deterministic Layer 1 regex/synonym parsing for Polish and English without external network or LLM calls.
2. **[CONF-02] Canonical DSL Enforcement**: All mutations and queries route exclusively through validated Layer 2 DSL commands.
3. **[CONF-03] Sandboxed LLM Fallback**: LLM integration is restricted to translation of unrecognized NL prompts into canonical DSL; LLM output is strictly validated by Layer 2 before execution.
4. **[CONF-04] Three-Interface Parity**: CLI (`ask`/`dsl`), REST API (`POST /dsl`, `POST /query`), and MCP Server (`nl_ask`, `execute_dsl`) are implemented and share the same core parser/executor.
5. **[CONF-05] Deterministic Conformance Verification**: Self-testing test suite executable via CLI (`standard/test_conformance.py` or equivalent) validating the full pipeline.
