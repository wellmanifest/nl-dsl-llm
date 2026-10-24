# Ticket 004: standardize standard adoption and pytest configuration

- **ID**: ticket-004
- **Owner**: agent:gemini
- **Status**: IN_PROGRESS
- **Workflow state**: VALIDATION
- **Created**: 2026-10-05

## Goal and scope

Declare formal adoption of parent standard `wellmanifest/nl-dsl-llm@0.1.0` in `pyproject.toml` under `[tool.wellmanifest]`, and configure `pythonpath = ["src"]` and `cache_dir = ".subactor/cache/pytest"` under `[tool.pytest.ini_options]` so that tests run directly out of the box with zero external environment overrides.

## Acceptance criteria

- [x] AC-01: `pyproject.toml` declares `adopts = ["wellmanifest/nl-dsl-llm@0.1.0"]`.
- [x] AC-02: `pyproject.toml` configures `pythonpath = ["src"]` and `cache_dir = ".subactor/cache/pytest"`.
- [x] AC-03: Full test suite passes directly with `pytest` (11/11 passed).
- [x] AC-04: `./project/governance-check.sh` passes with 0 errors and 0 warnings.

## SESSION_EXECUTION_AUTHORIZATION

User: "kontynuuj poprawe i zbadaj co jescze zostało do poprawniea rowniez w azlenzosciahc i porpawiaj".

## Tracking boundary

This directory contains the minimal reviewed intent. Optional participant prose
and raw command logs are not required delivery output.
