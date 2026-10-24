# Analysis: VPS Model Selection & Benchmark Evaluation on 8 vCPU / 8 GB RAM

- **Repository**: `paxlet-com/nl-dsl-llm`
- **Kind**: `analysis`
- **Standard**: `wellmanifest/nl-dsl-llm`
- **Scope**: `repository`
- **Status**: `PROPOSED`

---

## 1. Context & Motivation

Deploying an autonomous NL-to-DSL runtime on a single VPS with 8 vCPU and 8 GB RAM (CPU-only, no GPU)
imposes strict memory and compute constraints. The system must host Taskand, PostgreSQL, Redis, and
Playwright/noVNC, requiring >= 5.6 GB RAM reserved for host services. This benchmark evaluates a
prioritized matrix of compact LLMs under a single-resident model execution invariant.

## 2. Prioritized Model Matrix & Resource Allocation

| Priority | Model Profile | Role | Quant | RAM RSS | Langs | Polish Fidelity |
|---|---|---|---|---|---|---|
| **1** | `Qwen3-1.7B-Instruct` | Default production router (`enable_thinking=false`) | Q4_K_M | ~1.25 GB | 100+ | Strong (tool calling) |
| **2** | `Qwen3.5-2B` | Quality multilingual fallback & multi-step planning | Q4_K_M | ~1.78 GB | 201 | Official (agentic) |
| **3** | `Granite-4.0-Micro-3B` | Enterprise function calling & strict JSON schema | Q4_K_M | ~2.20 GB | 12 | Unofficial |
| **4** | `SmolLM3-3B` | Permissive tool calling (Apache 2.0) | Q4_K_M | ~2.10 GB | 6 | Limited |
| **5** | `LFM2.5-1.2B-Instruct` | Ultra-fast edge CPU extractor & classifications | Q4_K_M | ~0.89 GB | 8 | Unofficial |
| **6** | `Qwen3.5-0.8B` | Ultra-fast suggestion chips (<1s P95) | Q4_K_M | ~0.72 GB | 201 | Official |

## 3. Benchmark Findings & Performance Verification

Evaluation on 8 vCPU host across 17 battery tests (standard commands, polarity traps, scoping, disambiguation):

- **P50 Latency**: Qwen3.5-0.8B achieves ~334 ms; Qwen3-1.7B and Qwen3.5-2B achieve ~538 ms; 3B models achieve ~850 ms.
- **Accuracy**: Fast-path and semantic cache resolve 76.5% of routine queries in <1 ms without LLM invocation.
- **Digital Twin Pruning**: `DigitalTwinPromptSummarizer` condenses the catalog to top 5-10 actions, prioritizing degraded entities (`redis-cache`) at top priority before CPU prefill.
- **Single-Resident Invariant**: Only one model process runs at any time, guaranteeing >= 5.6 GB RAM headroom for Taskand, Postgres, Redis, and Playwright.

## 4. Architectural Roles & Recommendations

- **Primary Default Router**: `Qwen3-1.7B-Instruct` with `enable_thinking=false`, `-t 4`, and `max_tokens=128`.
- **Quality Multilingual Profile**: `Qwen3.5-2B` for complex multi-step planning and rich Polish inflection.
- **Gateway Switching**: Profile switching managed via LiteLLM or single-resident runner `scripts/run_llama_server.sh`.
- **Dual-Model Reference**: [Qwen Dual-Model Evaluation](QWEN_DUAL_MODEL_EVALUATION.md)
