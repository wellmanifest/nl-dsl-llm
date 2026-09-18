# Ticket 001: normative specification and reference implementation for NL-DSL-LLM

- **ID**: ticket-001
- **Owner**: human:tom
- **Status**: IN_PROGRESS
- **Workflow state**: VALIDATION
- **Created**: 2026-09-18

## Goal and scope

Define the normative specification, JSON schemas, standard profiles, and reference implementation for the NL-DSL-LLM pattern across interfaces (CLI shell, Web REST API, MCP).

## Acceptance criteria

- [x] AC-01: Normative specification document `spec/NL_DSL_LLM_SPECIFICATION.md` detailing the 3-layer pattern (NL, DSL, LLM fallback) and interface parity (CLI, REST API, MCP).
- [x] AC-02: JSON Schemas in `schemas/` for DSL request, execution result, MCP tools, and REST API contracts.
- [x] AC-03: Self-testing Python reference standard/conformance suite in `standard/` and `examples/`.
- [x] AC-04: Governance checks pass (`./project/governance-check.sh`).

## Tracking boundary

This directory contains the minimal reviewed intent. Optional participant prose
and raw command logs are not required delivery output.
