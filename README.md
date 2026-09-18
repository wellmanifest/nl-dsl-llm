# wellmanifest/nl-dsl-llm

**Standard and Reference Implementation for Tripartite Natural Language, Domain Specific Language, and Large Language Model (NL-DSL-LLM) Architectures.**

## Overview

The `nl-dsl-llm` standard establishes a unified, deterministic, and safe pattern for building developer and agent tools that expose:
1. **Natural Language (NL) Layer**: Multi-lingual intent parsing (Polish & English) with zero-cost, local deterministic matching for common commands.
2. **Domain Specific Language (DSL) Layer**: Canonical, strictly typed, executable, reversible, and validated domain verbs and grammar.
3. **LLM Translation Fallback Layer**: Provider-agnostic LLM compilation for unbounded, free-form natural language prompts into valid, constrained DSL statements.
4. **Universal Interface Parity**: Identical capabilities exposed across CLI Shell (REPL & one-shot), Web REST API, and Model Context Protocol (MCP stdio/SSE) for AI coding agents.

## Standard Structure

- `spec/`: Normative specification and architectural contracts.
- `schemas/`: Machine-verifiable JSON schemas for NL queries, DSL AST, and MCP tool definitions.
- `src/nl_dsl_llm/`: Python reference implementation of the tripartite pipeline.
- `tests/`: Conformance test suite.
- `examples/`: Integration patterns for real-world projects (e.g. `semcod/planfile`, `semcod/monag`).

## Governance

Adopts `wellmanifest/new-project` policy-as-code standard.
