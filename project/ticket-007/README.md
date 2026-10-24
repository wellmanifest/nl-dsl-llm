# ticket-007: Wire query and client_id kwargs and adopt Wellmanifest standards

- **ID**: ticket-007
- **Owner**: agent:antigravity
- **Status**: IN_PROGRESS
- **Workflow state**: EDIT
- **Created**: 2026-10-17

## Goal and scope

Wire `query` in `ContextPruningSummaryLayer.summarize` and `client_id` in `benchmark_both_models` (`PERF-AUDIT-KWARG-001`), and populate `.governance/standard-adoption.json`.

SESSION_EXECUTION_AUTHORIZATION: User authorized autonomous execution of PERF-AUDIT remediation and Wellmanifest standard adoption in session 57c04ad1-83e7-411b-86da-41eeeeb0570b.

## Acceptance criteria

- [x] AC-01: Wire `query` in `src/paxlet_nl_dsl_llm/layers.py` and `client_id` in `src/paxlet_nl_dsl_llm/server.py` (`PERF-AUDIT-KWARG-001`).
- [x] AC-02: Populate `.governance/standard-adoption.json` so `python3 .governance/standard_pack_check.py --root . --strict` reports `findings=0`.
- [x] AC-03: Pass `./project/governance-check.sh` (`GOV-PASS`).
