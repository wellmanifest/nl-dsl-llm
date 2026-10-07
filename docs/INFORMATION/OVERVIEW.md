---
schema: wellmanifest.docs/document/v2
title: NL-DSL-LLM Architecture and Operational Overview
description: Multi-layered natural language to deterministic DSL intent translation runtime with local CPU inference and gRPC IPC.
category: architecture
tags:
  - architecture
  - nlp
  - dsl
  - runtime
  - grpc
---

# NL-DSL-LLM Architecture & Operational Overview

## summary

`nl-dsl-llm` provides deterministic, ultra-low-latency translation of natural language queries
into strongly typed DSL actions and structured execution envelopes across the Willman and Paxlet ecosystem.

## details

### Resolution Pipeline

```
[ User Input / Natural Language Prompt ]
                   │
                   ▼
┌───────────────────────────────────────┐
│ Layer 1: Deterministic Fast-Path      │ < 1ms, regex & keyword matching
└──────────────────┬────────────────────┘
                   │ (miss / ambiguous)
                   ▼
┌───────────────────────────────────────┐
│ Layer 1.5: Semantic Embedding Cache   │ Cosine similarity & slot validation
└──────────────────┬────────────────────┘
                   │ (miss)
                   ▼
┌───────────────────────────────────────┐
│ Layer 2 / 3: Local LLM Compiler       │ Schema-constrained decoding (Qwen3.5)
└──────────────────┬────────────────────┘
                   │
                   ▼
[ Validated Domain Action / DSL Envelope ]
```

### Core Capabilities

1. **Deterministic Fast-Path**: Sub-millisecond keyword and regex intent matching with zero token overhead.
2. **Semantic Cache**: Embedding-based candidate selection with strict slot and negation validation.
3. **Local LLM Engine**: CPU inference via `llama.cpp` / LiteLLM with schema-constrained JSON grammar generation.
4. **High-Performance gRPC**: Unix domain sockets (`unix:///run/paxlet/nl-dsl-llm.sock`) and TCP transport.

### Ecosystem Integration

- Direct client adapters for **Taskand**, **Clonerd**, and **nl-dsl-sh**.
- Documentation standard: `wellmanifest.docs/document/v2`.
