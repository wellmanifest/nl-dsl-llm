# Wellmanifest Standard: NL-DSL-LLM Architecture Specification

- **Standard ID**: `wellmanifest/nl-dsl-llm`
- **Version**: `0.2.0-candidate`
- **Status**: `Draft / Candidate`
- **Authors**: Wellmanifest Architecture Workgroup

---

## 1. Executive Summary & Architectural Motivation

Modern agentic and developer tooling faces a fundamental trade-off:
1. **Natural Language (NL) Interfaces**: Flexible, accessible, and intuitive for humans and conversational AI agents, but nondeterministic, prone to hallucinations, and difficult to audit or sandbox safely.
2. **Domain-Specific Languages (DSL)**: Strictly typed, deterministic, parseable, transactionally sound, and auditable, but steep learning curves and rigid syntax make them cumbersome for ad-hoc natural queries.
3. **Direct LLM Tool Execution**: Calling APIs or tools directly from LLM completions introduces security vulnerabilities, unvetted parameter passing, and unbounded API costs.

The **NL-DSL-LLM Pattern** resolves this trade-off by establishing a 4-tier, fail-closed architecture where:
- **DSL is the sole execution surface and single source of truth (SSOT).**
- **Natural Language is an intake boundary, never an executor.**
- **Layer 1: Rule-based parsing** handles standard operations with 0ms latency and 0 token cost.
- **Layer 1.5: Semantic Intent Cache** matches recurring intents using embeddings with strict negation and slot isolation guards.
- **Layer 3: Constrained LLM translation** serves exclusively as an adaptive fallback compiler with token-level grammar masking.
- **Universal Interface Parity** guarantees identical semantic behavior across CLI, Web REST API, MCP Server, and binary gRPC.

```
                     ┌─────────────────────────────────────────┐
                     │           Incoming Requests             │
                     │  (CLI Shell / Web REST / MCP / gRPC)    │
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
              └── NO ──────►│ Layer 1.5: Semantic   │         │
                            │ Cache & Slot Guard    │         │
                            │ (multilingual-e5)     │         │
                            └───────────┬───────────┘         │
                                        │                     │
                                Cache match verified?         │
                                ├── YES ┐                     │
                                │       ▼                     │
                                │   ┌───────────────────────┐ │
                                └──►│ Layer 3: LLM Fallback │ │
                                    │ Schema-Constrained    │ │
                                    │ (NL -> Canonical DSL) │ │
                                    └───────────┬───────────┘ │
                                                │             │
                                                ▼             ▼
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
                             │ (JSON / Markdown / Protobuf)      │
                             └───────────────────────────────────┘
```

---

## 2. Core Architectural Layers

### 2.1 Layer 1: Deterministic NL Pattern Parser (Fast Path)

The deterministic NL parser is a localized, rule-driven intake engine that translates recognized linguistic patterns into canonical DSL structures without network requests or model inference.
- **Characteristics**: Sub-millisecond latency (< 1ms), zero tokens, 100% offline capable.
- **Multilingual Normalization**: Support for Polish (PL) and English (EN) root verbs and entities.

### 2.2 Layer 1.5: Semantic Intent Cache & Candidate Matcher

Provides vector-based intent reuse while enforcing strict safeguards against fuzzy matching traps:
- **Explicit Polarity Guard**: Inversion check for negation particles (`nie`, `not`, `don't`, `bez`, `never`). Never maps negated queries to positive action templates.
- **Target Slot Verification**: Environment scopes (`prod` vs `test`) and numeric bounds must match exactly.
- **Fail-Closed Policy**: Any ambiguity immediately falls through to Layer 3.

### 2.3 Layer 2: Canonical DSL Engine (Single Execution Boundary)

The DSL is the **sole executable representation** within the system. No agent, user, or LLM may invoke internal business logic without constructing a validated DSL command envelope.
- **Single Source of Truth**: All operations are formally declared DSL commands.
- **Transactional & Idempotent**: Commands accept idempotency keys and leases.

### 2.4 Layer 3: Schema-Constrained LLM Translation Compiler

Acts as an adaptive semantic compiler when Layers 1 and 1.5 fail to resolve the prompt:
- **Token-Level Masking**: Enforces strict GBNF or JSON schema decoding (`response_format`).
- **Action ID Extraction**: Extracts concise `action_id` and arguments (30-60 tokens) rather than raw URIs or shell scripts.
- **Clarification State**: Emits `status: "clarify"` when prompt is ambiguous, returning structured options.

### 2.5 Layer 0.5: Real-time Autocomplete & Contextual Option Network (Digital Twin Projection)

Enables keystroke-by-keystroke and streaming voice suggestion chips without requiring the user to type or pronounce full commands:
- **Prefix & Inverted Trie (<2ms)**: Fast prefix indexing for command verbs and entities.
- **Digital Twin State Projection**: Contextually filters and populates candidate slots strictly based on live environment entities (active/degraded services, containers, open tickets, nodes). Non-existent resources are pruned from suggestions.
- **Option Network DAG**: Returning options generate connected child nodes (sub-options, flags like `--graceful`, `--force`, or parameter values) forming an interactive selection network.
- **Safety Invariants**: Actions targeting `prod` environments are flagged with `requires_confirmation = true` and `badge = "PROD"`.

---

## 3. Unified Interface Parity

Every adopting system provides four synchronous interface surfaces sharing the identical engine:

### 3.1 CLI Shell (`<tool> shell`, `<tool> ask`, `<tool> dsl`, `<tool> suggest`)
- Interactive REPL, real-time tab-completion, and command-line entry point.

### 3.2 Web REST API (`/dsl`, `/query`, `/suggest`, `/schema`)
- Standardized HTTP endpoints for web dashboards and external automation.

### 3.3 Model Context Protocol (MCP Server)
- Standardized tool integration for AI agents (`nl_ask`, `execute_dsl`, `suggest_options`).

### 3.4 High-Performance gRPC IPC (`paxlet.nl_dsl_llm.v1.NLRuntimeService`)
- Standardized binary RPC over Unix Domain Sockets (`/run/...`) and TCP (`InterpretIntent`, `SuggestOptions`).
- Provides sub-millisecond local IPC (<0.5ms) for containerized and daemon services.

### 3.5 Bidirectional WebSocket Streaming Protocol (`/ws/stream`)
- Real-time bidirectional streaming transport for voice audio chunks (PCM 16kHz, Opus) and interactive UI inputs.
- Streaming frame envelopes (`NLStreamFrame`) supporting:
  - `STREAM_START`: Client handshake specifying schema dialect, session ID, audio format, and feature flags (`option_network_stream: true`).
  - `AUDIO_CHUNK`: Chunked binary/base64 audio frames (20ms-100ms) for streaming STT processing.
  - `TRANSCRIPTION_PARTIAL`: Continuous interim transcription updates with stability scores (`0.0`-`1.0`).
  - `OPTION_NETWORK_UPDATE`: Real-time candidate graph updates pushed to the client with sub-50ms latency as prefix tokens evolve.
  - `COMMAND_COMMITTED`: User utterance or chip selection confirmed; triggers deterministic 4-tier pipeline execution.
  - `COMMAND_RESULT`: Execution completion payload containing canonical DSL, status, latency telemetry, and UI feedback.
  - `STREAM_ERROR`: Structured diagnostic errors with failure category and recovery action.

### 3.6 Option Network Live Autocomplete Engine
- Fast Trie and DAG path-finding engine executing within <5ms in-memory.
- Given streaming prefix $P$, computes the active Option Network $\mathcal{G} = (V, E)$:
  - Vertices $V$: Candidate command tokens, parameters, and entities.
  - Edges $E$: Valid semantic transitions governed by the domain grammar.
- Prunes branches violating current Digital Twin state or permission boundaries.
- Emits ranked suggestions with confidence weights, badges, and parameter defaults.

### 3.7 Conversational Stream Purity & Dedicated Process View (NUL-007 Binding)
In interactive chat, conversational agent palettes, or conversational CLI shells:
- **Strict Conversational Purity**: The chat stream is reserved for dialogue, user utterances, and structured execution notifications.
- **Process Isolation**: Raw subprocess stdout/stderr, multiline dumps, and ANSI terminal sequences must not pollute the conversational transcript.
- **Canonical Addressing**: Executions are referenced by RFC 3986 Action URI (`process://<host>/<bin>?<params>`) and Resource URN (`urn:<domain>:proc:<id>`).
- **Dedicated Artifact View**: Execution output, buffers, exit codes, and interactive CLI inputs are displayed in a separate, isolated Process/Terminal Artifact View (`.pal-term`).

### 3.8 Bidirectional Interactive State URL Synchronization (NUL-008 Binding)
Every web-based or dashboard interface adopting the NL-DSL-LLM pattern must maintain full state synchronization with RFC 3986 query parameters via debounced `history.replaceState`:
- **Real-Time Serialization**: Active layout, pane configurations, focus, open modal/chat, active artifact tab, search/command query, and last interaction event are reflected in URL query parameters.
- **Deterministic Reconstruction**: Loading or deep-linking to an URL with query parameters faithfully restores the exact workspace, panes, focus, and input state without loss.

---

## 4. Conformance Criteria & Checkpoints

1. **[CONF-01] Multilingual Fast Path**: Implements deterministic Layer 1 regex/synonym parsing for Polish and English.
2. **[CONF-02] Canonical DSL Enforcement**: All mutations and queries route exclusively through validated Layer 2 DSL commands.
3. **[CONF-03] Sandboxed LLM Fallback**: LLM integration is restricted to translating unrecognized NL prompts into canonical DSL with schema constraints.
4. **[CONF-04] Unified Interface Parity**: CLI (`ask`/`dsl`/`suggest`), REST API, MCP Server (`nl_ask`), gRPC (`InterpretIntent`, `SuggestOptions`), and WebSocket streaming are supported.
5. **[CONF-05] Semantic Intent Cache Integrity**: Explicit polarity check and slot verification prevent negation inversion errors.
6. **[CONF-06] Contextual Option Network**: Prefix suggestions dynamically prune non-existent entities and project Digital Twin resource graphs within <5ms.
7. **[CONF-07] Deterministic Conformance Verification**: Self-testing test suite executable via CLI.
8. **[CONF-08] WebSocket Bidirectional Streaming Parity**: Streaming chunked audio/tokens with sub-50ms live Option Network autocomplete and structured frame events.
9. **[CONF-09] Conversational Stream Purity**: Chat log nodes exclude raw process stdout/stderr streams, emitting only dialogue and canonical URI/URN receipts.
10. **[CONF-10] Deterministic Bidirectional State URL Sync**: Deep linking and reloading URL with query parameters restores exact UI layout, focus, dialog, and active tab.
