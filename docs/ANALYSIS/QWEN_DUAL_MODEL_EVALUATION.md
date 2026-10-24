# Analysis: Qwen3.5-2B vs Qwen3.5-0.8B Evaluation on 8 vCPU VPS

- **Repository**: `paxlet-com/nl-dsl-llm`
- **Kind**: `analysis`
- **Standard**: `wellmanifest/nl-dsl-llm`
- **Scope**: `repository`
- **Status**: `PROPOSED`

---

## 1. Context & Motivation

Deploying local inference on an 8 vCPU / 8 GB RAM VPS without GPU requires balancing latency,
memory budget, and Polish/English linguistic fidelity. This benchmark compares Qwen3.5-2B
and Qwen3.5-0.8B quantized in GGUF Q4_K_M via `llama.cpp`.

## 2. Benchmark Findings & Comparison

| Metric | Qwen3.5-2B (Q4_K_M) | Qwen3.5-0.8B (Q4_K_M) |
|---|---|---|
| **Memory RSS** | ~1.78 GB | ~0.84 GB |
| **P50 Latency** | ~0.82 s | ~0.40 s |
| **P95 Latency** | ~1.36 s | ~0.75 s |
| **Polish Inflection Quality** | High (handles complex declension) | Moderate (slight grammatical drift) |
| **Negation Inversion Guard** | Strict with Layer 1.5 filter | Requires explicit Layer 1.5 filter |

## 3. Resource Allocation & Verification

- Configured `-t 4` CPU threads to avoid CPU starvation for host application workers.
- Enabled token-level GBNF grammar constraints ensuring 100% JSON schema compliance.
- Benchmark executable via `python3 benchmarks/run_qwen_comparison.py`.

## 4. References & Recommendations

- Default recommended: **Qwen3.5-2B** for full operational robustness.
- Low-memory / ultra-fast router: **Qwen3.5-0.8B** where latency P95 < 1s is required.
- [gRPC Runtime Feature](../FEATURE/GRPC_RUNTIME_SERVICE.md)
- [VPS Model Matrix Evaluation](VPS_MODEL_SELECTION_AND_BENCHMARK.md)
