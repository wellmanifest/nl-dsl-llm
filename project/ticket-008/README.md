# Ticket 008: standardize pytest pythonpath and packaging configuration

- **ID**: ticket-008
- **Owner**: agent:gemini
- **Status**: IN_PROGRESS
- **Workflow state**: VALIDATION
- **Created**: 2026-10-05

## Goal and scope

Standardize pytest testpaths and pythonpath configuration and package `__init__.py` markers across `wellmanifest/nl-dsl-llm` to ensure seamless standalone pytest execution matching child standard standards (`wellmanifest/nl-api-llm` and `wellmanifest/nl-uri-dsl-llm`).

## Acceptance criteria

- [x] AC-01: `standard/__init__.py` package marker created.
- [x] AC-02: `pyproject.toml` configures `testpaths = ["standard"]` and `pythonpath = ["."]`.
- [x] AC-03: Full test suite passes directly with `pytest`.
- [x] AC-04: `./project/governance-check.sh` passes with 0 errors and 0 warnings.

## SESSION_EXECUTION_AUTHORIZATION

User: "scalaj integruj , zadbaj o standaryzacje zaleznosci na poziomie wellmanifest/nl-*standardow".

## Tracking boundary

This directory contains the minimal reviewed intent. Optional participant prose
and raw command logs are not required delivery output.
