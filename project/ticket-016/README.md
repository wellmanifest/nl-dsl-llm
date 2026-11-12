# Ticket 016: Fix paxlet_nl_dsl_llm json import, restore tests suite in pytest, and harden layers, registry, and server

- **ID**: ticket-016
- **Owner**: agent:antigravity
- **Status**: IN_PROGRESS
- **Workflow state**: EDIT
- **Created**: 2026-11-11

## Session authorization

- `SESSION_EXECUTION_AUTHORIZATION`: Autonomous execution, testing, and protected publication requested by the repository owner.

## Goal and scope

Fix the broken `paxlet_nl_dsl_llm` package import (`NameError: name 'json' is not defined` in `src/paxlet_nl_dsl_llm/layers.py`), restore `tests/test_suite.py` execution under `pytest` (`testpaths = ["standard", "tests"]`), and harden the 4-layer NL runtime, registry, and server modules while keeping every repository Python file strictly below 500 lines:

1. Add missing `import json` and `O_NOFOLLOW | O_NONBLOCK` regular-file checks in `src/paxlet_nl_dsl_llm/layers.py`, replace `O(N*M)` dict allocations in `damerau_levenshtein()` with a 3-row DP + `max_dist` fast exit, bound `FastPathParser.parse()` query/token lengths, cap `SemanticCache` capacity (`max_entries = 512`), and bound integer slot digits (`<= 18` digits) so `> 4300`-digit inputs cannot raise `ValueError`.
2. Extract `src/paxlet_nl_dsl_llm/model_profiles.py` to remove circular imports and keep all modules `< 500` lines, unify Polish/English negation detection (`NEGATION_PATTERN`) across `FastPathParser`, `SemanticCache`, and `LLMCompiler.compile()` in `src/paxlet_nl_dsl_llm/layers_compiler.py` so negated commands (`"nie czyść cache"`, `"don't clean cache"`, `"nigdy nie restartuj serwisu"`) fail closed to `clarify` instead of matching destructive actions, and guard `VoiceTokenStreamHandler.process_token()` bounds.
3. Harden `ProcedureRegistry.import_openapi_schema()` in `src/paxlet_nl_dsl_llm/registry.py` against non-dict payloads, `..` path traversal, and unsafe `operationId` characters, and harden `NLRuntimeEngine` in `src/paxlet_nl_dsl_llm/server.py` against empty/unknown `benchmark_matrix()` `IndexError`, non-mapping `execute()` arguments, and `ActionDef.handler` execution (`dry_run`-safe).
4. Add `"tests"` to `pyproject.toml` `testpaths`, make `tests/test_suite.py` work under CI `-o pythonpath=.`, expand regression coverage, and mark merged `ticket-013`, `ticket-014`, and `ticket-015` as `DONE`.

## Acceptance criteria

- [x] AC-01: Importing `paxlet_nl_dsl_llm.layers` and `paxlet_nl_dsl_llm.server` succeeds cleanly and both `pytest -q` and `python3 -B -m pytest -o pythonpath=. -p no:cacheprovider` execute `standard/` and `tests/test_suite.py` with 0 failures.
- [x] AC-02: Oversized tokens/queries, `> 4300`-digit numbers, unbounded cache promotions, negated Layer-3 commands, malformed OpenAPI schemas, empty `benchmark_matrix` profile lists, and non-dict `execute()` arguments fail closed without unhandled exceptions, and every Python module stays `< 500` lines after `ruff format`.
- [x] AC-03: `ruff check --no-cache`, `python3 standard/nl_dsl_llm.py --self-test`, `./project/governance-check.sh --base origin/main`, and `wellman check` pass with 0 errors.

## Tracking boundary

This directory contains the minimal reviewed intent. Optional participant prose
and raw command logs are not required delivery output.
