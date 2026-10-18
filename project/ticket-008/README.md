# Ticket 008: decompose layers module under 500 lines

- **ID**: ticket-008
- **Owner**: unresolved:human
- **Status**: IN_PROGRESS
- **Workflow state**: EDIT
- **Created**: 2026-10-18

## Goal and scope

Decompose `src/paxlet_nl_dsl_llm/layers.py` into `layers.py` and `layers_compiler.py` so every Python module in `src/paxlet_nl_dsl_llm/` is <= 500 lines while preserving all public symbols and behavior.

## Acceptance criteria

- [x] AC-01: Decompose `src/paxlet_nl_dsl_llm/layers.py` so every module in `src/paxlet_nl_dsl_llm/` is <= 500 lines with all tests passing.

## Tracking boundary

This directory contains the minimal reviewed intent. Optional participant prose
and raw command logs are not required delivery output.
