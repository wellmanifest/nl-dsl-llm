# Ticket 011: Eliminate NOHARDCODE-001/007 findings and add canonical System One selector contract (#10)

- **ID**: ticket-011
- **Owner**: agent:antigravity
- **Status**: IN_PROGRESS
- **Workflow state**: EDIT
- **Created**: 2026-10-25

## Goal and scope

Eliminate all `NOHARDCODE-001` and `NOHARDCODE-007` findings across `wellmanifest/nl-dsl-llm` (`examples/websocket_streaming_client.py`, `standard/domain_showcase.py`, `standard/nl_dsl_llm.py`, `standard/test_websocket_streaming.py`) and expose the canonical `build_selection_question` and `select_operation` System One / Jev selector contract in `standard/system_one_decision.py` (with `op_0..op_N` candidate bindings, `none`, `need_clarification`, `need_multi_step`, `registry_digest`, and 6 deterministic safety gates). Closes #10.

## Acceptance criteria

- [x] AC-01: `nohardcode_check.py --target . --profile balanced` reports 0 violations (`0 errors, 0 warnings`).
- [x] AC-02: `standard/system_one_decision.py` exposes `build_selection_question` and `select_operation` with `op_0..op_N` candidate bindings and 6 safety gates, verified in `standard/test_system_one_decision.py`.
- [x] AC-03: `pytest`, `./project/governance-check.sh`, and `python3 .governance/standard_pack_check.py --root .` pass with 0 errors.

## SESSION_EXECUTION_AUTHORIZATION

User authorized autonomous execution across all repositories to remediate `wellmanifest/nohardcode` findings and implement the System One / Jev selector contract (`Closes #10`).

## Tracking boundary

This directory contains the minimal reviewed intent. Optional participant prose
and raw command logs are not required delivery output.
