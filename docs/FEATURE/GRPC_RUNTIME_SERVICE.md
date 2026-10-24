# Feature: High-Performance gRPC Runtime Service

- **Repository**: `paxlet-com/nl-dsl-llm`
- **Kind**: `feature`
- **Standard**: `wellmanifest/nl-dsl-llm`
- **Scope**: `repository`
- **Status**: `PROPOSED`

---

## 1. Context & Purpose

The `paxlet-com/nl-dsl-llm` service provides an ultra-low-latency inter-process communication (IPC)
boundary for downstream systems (Taskand, Clonerd, nl-dsl-sh). Natural language queries are translated
into deterministic domain actions via a 4-layer resolution engine.

## 2. Specification & Contracts

The normative service contract is declared in `proto/paxlet/nl_dsl_llm/v1/runtime.proto`:
- `InterpretIntent`: Translates natural language prompts to `action_id` and structured arguments.
- `ExecuteAction`: Executes validated actions with idempotency keys and transactional envelopes.
- `BenchmarkIntent`: Evaluates identical queries across dual models (Qwen3.5-2B vs 0.8B).

Transport supports both Unix Domain Sockets (`unix:///run/paxlet/nl-dsl-llm.sock`) and TCP (`:50051`).

## 3. Verification & Observability

- Conformance verified via `tests/test_runtime_engine.py` and `tests/test_fast_path.py`.
- Metrics captured per RPC: `prefill_latency_ms`, `generation_latency_ms`, token counts, trace IDs.
- Deterministic Layer 1 achieves sub-millisecond latency (<1ms) with zero token cost.

## 4. References & Next Steps

- [NL-DSL-LLM Specification](../../spec/NL_DSL_LLM_SPECIFICATION.md)
- [Qwen Dual-Model Evaluation](../ANALYSIS/QWEN_DUAL_MODEL_EVALUATION.md)
- [VPS Deployment Guide](../SERVICE/DEPLOYMENT_VPS.md)
