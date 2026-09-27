# Adoption Case Study: paxlet-com/nl-dsl-llm (gRPC Runtime & Dual-Qwen Pipeline)

This document provides a concrete reference walkthrough of the **NL-DSL-LLM** pattern adopted in
`paxlet-com/nl-dsl-llm` for low-latency inter-process communication on Linux VPS servers.

---

## 1. Architectural Mapping

In `paxlet-com/nl-dsl-llm`, the four-tier tripartite pattern is mapped as follows:

| Layer | Component | Description |
|---|---|---|
| **Layer 1: Fast-Path NL Parser** | `paxlet_nl_dsl_llm/layers.py` (`FastPathParser`) | Multilingual (PL/EN) deterministic regex and keyword parser. Latency < 1ms, 0 external tokens. |
| **Layer 1.5: Semantic Intent Cache** | `paxlet_nl_dsl_llm/layers.py` (`SemanticCache`) | Embedding and template cache with explicit polarity/negation guard and slot collision detection. |
| **Layer 2: Canonical DSL & Registry** | `paxlet_nl_dsl_llm/registry.py` (`ProcedureRegistry`) | Validates and resolves abstract `action_id` into procedure URIs (`proc://`, `urn://`, `cluster.`). |
| **Layer 3: Constrained LLM Fallback** | `paxlet_nl_dsl_llm/layers.py` (`LLMCompiler`) | Local inference running Qwen3.5-2B / 0.8B via `llama.cpp` + LiteLLM with token-level GBNF grammar constraints. |
| **Interface Parity: gRPC & CLI** | `paxlet_nl_dsl_llm/server.py` | Exposes `NLRuntimeService` over Unix Domain Sockets (`/run/paxlet/nl-dsl-llm.sock`) and TCP (`:50051`). |

---

## 2. Low-Latency gRPC Protocol

The service exposes the standard Protobuf contract defined in `proto/wellmanifest/nl_dsl_llm/v1/runtime.proto`:

```protobuf
service NLRuntimeService {
  rpc InterpretIntent(InterpretIntentRequest) returns (InterpretIntentResponse);
  rpc ExecuteAction(ExecuteActionRequest) returns (ExecuteActionResponse);
  rpc BenchmarkIntent(BenchmarkIntentRequest) returns (BenchmarkIntentResponse);
  rpc GetActionCatalog(GetActionCatalogRequest) returns (GetActionCatalogResponse);
}
```

By binding to a local Unix Domain Socket (`unix:///run/paxlet/nl-dsl-llm.sock`), IPC overhead between
local containers (Taskand, Clonerd, nl-dsl-sh) is reduced to sub-millisecond ranges (<0.5ms).

---

## 3. Dual-Model Evaluation: Qwen3.5-2B vs Qwen3.5-0.8B

To achieve fast inference on a VPS without GPU (8 vCPU, 8 GB RAM), the service supports dual-model evaluation:

| Metric | Qwen3.5-2B (Q4_K_M) | Qwen3.5-0.8B (Q4_K_M) | Target Recommendation |
|---|---|---|---|
| **RAM RSS** | ~1.78 GB | ~0.84 GB | Both fit simultaneously in 8 GB RAM alongside apps. |
| **P50 Latency** | ~0.82 s | ~0.40 s | 0.8B achieves sub-second routing (<500ms). |
| **P95 Latency** | ~1.36 s | ~0.75 s | Both comfortably meet the target P95 ≤ 5s. |
| **Polish Inflection** | High | Moderate | 2B recommended as default for natural Polish phrasing. |
| **Negation Guard** | Strict | Strict (via Layer 1.5) | Explicit polarity check prevents dangerous regressions. |

---

## 4. Downstream Client Integration

### Taskand (`dev/chat/...` and `dev/llm/...`)
Replaces 90-second remote timeout with low-latency local gRPC call:

```javascript
// taskand client integration
const client = new proto.NLRuntimeService('unix:///run/paxlet/nl-dsl-llm.sock', grpc.credentials.createInsecure());
const res = await client.interpretIntent({ query: "status procesora", client_id: "taskand" });
// res.resolved_procedure_uri -> "proc://taskand.dev/monitor/cpu/v1"
```

### nl-dsl-sh
Resolves natural language commands directly into verified executable URNs (`urn:nl-dsl-sh:...`):

```python
# nl-dsl-sh runtime integration
urn = client.interpret("wyczyść cache", client_id="nl-dsl-sh")["resolved_procedure_uri"]
# urn -> "urn:nl-dsl-sh:cache:purge"
```
