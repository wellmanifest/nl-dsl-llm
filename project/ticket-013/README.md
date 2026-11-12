# Ticket 013: Externalize NL intent lexicon into semantic-profile.json and remove pipe-string hardcodes

- **ID**: ticket-013
- **Owner**: agent:antigravity
- **Status**: DONE
- **Workflow state**: DONE
- **Created**: 2026-11-02

## Goal and scope

Externalize the natural-language intent lexicon (`verbs`, `entities`, `status_filters`, `fast_path_patterns`) into `standard/semantic-profile.json` and eliminate all remaining `NOHARDCODE-001`, `NOHARDCODE-006`, and `NOHARDCODE-007` findings across `standard/nl_dsl_llm.py`, `src/paxlet_nl_dsl_llm/layers.py`, `src/paxlet_nl_dsl_llm/server.py`, `clients/python/nl_dsl_client.py`, and `scripts/run_llama_servers.sh`.

## Acceptance criteria

- [x] AC-01: `nohardcode_check.py --target . --profile strict` passes with 0 findings (`GOV-PASS`).
- [x] AC-02: `standard/nl_dsl_llm.py` and `src/paxlet_nl_dsl_llm/layers.py` load their NL lexicons and fast-path patterns from `standard/semantic-profile.json` with all unit tests passing.
- [x] AC-03: `./project/governance-check.sh` and `wellman check` pass with 0 errors.

## SESSION_EXECUTION_AUTHORIZATION

User authorized autonomous execution across `wellmanifest/*` repositories to eliminate hardcoded heuristics, externalize NL catalogs, and verify conformance.

## Tracking boundary

This directory contains the minimal reviewed intent. Optional participant prose
and raw command logs are not required delivery output.
