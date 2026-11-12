# Ticket 017: Harden Python IPC client socket framing, byte bounds, and Koru polarity guard

- **ID**: ticket-017
- **Owner**: agent:antigravity
- **Status**: IN_PROGRESS
- **Workflow state**: EDIT
- **Created**: 2026-11-12

## Session authorization

- `SESSION_EXECUTION_AUTHORIZATION`: Autonomous execution, testing, and protected publication requested by the repository owner.

## Goal and scope

Harden `clients/python/nl_dsl_client.py` and `clients/python/nl_dsl_ipc.py` (`< 500` lines each) across IPC framing, filesystem socket checks, and `KoruNLClient` safety:

1. **Bounded single-frame IPC & `NaN`/`Infinity` rejection**: Cap `_recv_json_line()` at `MAX_IPC_RESPONSE_BYTES = 1_048_576`, extract strictly the first `\n`-delimited JSON line so pipelined frames do not fail with `Extra data`, enforce `allow_nan=False` and `parse_constant=_reject_nonfinite`, validate `port` (`1..65535`) and positive finite `timeout_seconds`, verify `socket_path` via `os.lstat` (`S_ISSOCK`) and ancestor symlink checks, and cache the in-memory `NLRuntimeEngine` fallback per `NLClient` instance.
2. **`KoruNLClient` polarity, 4300-digit integer, & ASCII ticket guards**: Block negated commands (`"nie naprawiaj projektu"`, `"don't run doctor"`, `"nigdy nie uruchamiaj pętli"`) before synthesizing `koru --doctor --repair` or queue loops, cap loop iteration parsing to `[1, 1000]` without `> 4300`-digit `ValueError`, restrict `koru.context` ticket IDs to ASCII, add `--self-test`, and mark merged `ticket-016` as `DONE`.

## Acceptance criteria

- [x] AC-01: `python3 clients/python/nl_dsl_client.py --self-test` and `ruff check --no-cache clients/python/nl_dsl_client.py clients/python/nl_dsl_ipc.py` pass with 0 errors and both client modules remain `< 500` lines after `ruff format`.
- [x] AC-02: `pytest -q`, `./project/governance-check.sh --base origin/main`, and `wellman check` pass with 0 errors.

## Tracking boundary

This directory contains the minimal reviewed intent. Optional participant prose
and raw command logs are not required delivery output.
