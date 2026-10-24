# Ticket 002: Client adapters for Taskand, Clonerd, and nl-dsl-sh

- **ID**: ticket-002
- **Owner**: human:tom
- **Status**: IN_PROGRESS
- **Workflow state**: VALIDATION
- **Created**: 2026-09-26

## Goal and scope

Provide production-grade client adapters for downstream consumers (Taskand, Clonerd, and `nl-dsl-sh`) to interact with the `paxlet-com/nl-dsl-llm` daemon over Unix Domain Sockets (`/run/paxlet/nl-dsl-llm.sock`) and TCP.

## Acceptance criteria

- [x] AC-01: Native Node.js stream client `clients/node/taskand_nl_client.mjs` for Taskand `dev/chat` and `dev/llm`.
- [x] AC-02: Standalone Python client `clients/python/nl_dsl_client.py` for Python daemons and Clonerd.
- [x] AC-03: Executable Bash CLI `clients/bash/nl-dsl` for `nl-dsl-sh` interactive dispatch and benchmarking.
- [x] AC-04: Documentation in `clients/README.md` compliant with `wellmanifest/docs` v2 Compact Profile.
- [x] AC-05: Governance checks pass (`./project/governance-check.sh`).

## Tracking boundary

This directory contains the minimal reviewed intent. Optional participant prose
and raw command logs are not required delivery output.
