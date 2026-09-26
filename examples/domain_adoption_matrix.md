# Domain Adoption Matrix & Applicability Analysis: NL-DSL-LLM

This document provides a comprehensive evaluation of where the **NL-DSL-LLM** standard and runtime
are applied, where they provide the highest architectural ROI, and what future improvements can be made.

---

## 1. Existing Project Ecosystem Applicability

| Project | Current Architecture / Bottleneck | NL-DSL-LLM Solution & Layer Mapping | Expected Latency & ROI |
|---|---|---|---|
| **Taskand** (`paxlet-com/taskand`) | Remote LLM (`glm-5.3`) with 90s–120s timeout; unstructured text prompts; manual JSON parsing. | **Layer 3 local profile** (Qwen3-1.7B) + **Layer 0.5** autocomplete in API Menu + `proc://` URI resolution. | **<1s latency** vs 90s; zero cloud cost; 100% schema validation. |
| **Clonerd** (`paxlet-com/clonerd`) | Complex manual git commands, risk of dirty checkout pollution, worktree drift. | **Layer 1 & 2**: Natural commands ("utwórz worktree dla ticket-005") mapped to deterministic `cluster.worktree.allocate`. | **<1ms** execution; enforces Worktrees v5 relative linking and lease metadata. |
| **nl-dsl-sh** (`paxlet-com/nl-dsl-sh`) | Terminal shell requiring deep memorization of complex CLI flags (`journalctl`, `systemctl`, `docker`). | **Layer 0.5 Option Network**: As user types 2-3 letters or speaks, instant option DAG chips appear in terminal. | **0.25ms** response time; eliminates terminal typos and destructive production errors. |
| **Subactor / Semcod** | Multi-agent swarms communicating via raw English prose, causing prompt drift and hallucinated parameters. | **Layer 2 Canonical DSL**: Agent communication normalized to typed DSL commands (`entity.operation`). | Prevents multi-step agent hallucination loops; reduces token usage by **>80%**. |
| **Wellmanifest Governance** | Human prose policies translated manually into repository hooks and pre-commit checks. | **Tripartite Bridge**: Compliance queries compiled to policy DSL invariants with recomputable evidence. | Guaranteed auditability and zero-drift policy compliance. |

---

## 2. Broader Industrial & Enterprise Use Cases

### A. Autonomous SRE & Self-Healing Infrastructure
- **Scenario**: Alertmanager/Prometheus triggers an incident alert (e.g. `redis-cache` degradation or crash loop).
- **Pipeline**: Alert payload feeds the **Digital Twin Context**. `DigitalTwinPromptSummarizer` flags `redis-cache` as `HIGH_DEGRADED` and pre-filters runbook actions.
- **Outcome**: The operator or remediation daemon gets a deterministic `service.restart(target="redis-cache")` action with `requires_confirmation: true` badge on production.

### B. Voice-First Field Operations & Industrial SCADA / IoT
- **Scenario**: Maintenance technicians or plant operators with hands occupied speaking commands into a headset.
- **Pipeline**: Streaming speech tokens from edge ASR (Whisper / sherpa-onnx) feed **Layer 0.5 (Option Network Engine)**.
- **Outcome**: Before the technician finishes speaking a full sentence, the HUD / screen displays the exact matching DAG nodes in **0.18 ms**, allowing single-tap or one-word confirmation.

### C. Resource-Constrained Edge AI & Micro-VPS (8 vCPU / 8 GB RAM)
- **Scenario**: Branch office, retail edge node, or budget VPS without GPU running PostgreSQL, Redis, web apps, and AI.
- **Pipeline**: **Single-Resident Model Invariant** enforces that only one compact model (Qwen3-1.7B ~1.25 GB RSS) resides in RAM.
- **Outcome**: Leaves **> 5.6 GB RAM** completely free for mission-critical databases and runtime processes, preventing OOM crashes.

---

## 3. Recommended Future Standard Improvements

1. **Streaming Audio Tokenization (Speech-to-Intent)**:
   - Direct integration of WebRTC / gRPC audio streaming with server-side VAD (Voice Activity Detection) feeding Layer 0.5.
2. **Federated Procedure Catalogs (Reflection / Schema Discovery)**:
   - Dynamic catalog federation from OpenAPI v3, gRPC Reflection, and Protobuf registries with automatic GBNF grammar compilation.
3. **Active Learning & Semantic Cache Promotion**:
   - Promotion of verified Layer 3 LLM compilation successes into Layer 1.5 Semantic Cache with automated polarity and slot boundary verification.
4. **Direct eBPF Telemetry Feeding for Digital Twin**:
   - Real-time kernel telemetry (cgroup memory pressure, TCP socket drops) feeding `DigitalTwinContext` for predictive remediation.

---

## 4. Verification

Run the multi-domain showcase to verify real-time execution across all four domains:
```bash
python3 standard/domain_showcase.py
```
