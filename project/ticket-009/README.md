# Ticket 009: Wire optional schema_hints parameter in LLMTranslator.translate

- **ID**: ticket-009
- **Owner**: tom
- **Status**: IN_PROGRESS
- **Workflow state**: EDIT
- **Created**: 2026-10-17

## Goal and scope

Wire optional schema_hints parameter in LLMTranslator.translate so the repository passes AST performance audit and standard pack conformance checks with zero findings.

## Acceptance criteria

- [x] AC-01: Remediate all `PERF-AUDIT-*` findings in `nl-dsl-llm` (`audit_repository` returns `0` findings).
- [x] AC-02: Align `.governance/standard-adoption.json` (`standard_pack_check.py` returns `findings=0`) and pass `pytest -q`.

## Tracking boundary

This directory contains the minimal reviewed intent. Optional participant prose
and raw command logs are not required delivery output.
