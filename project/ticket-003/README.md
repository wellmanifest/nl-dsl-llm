# Ticket 003: Docker and systemd deployment for shared nl-runtime daemon

- **ID**: ticket-003
- **Owner**: human:tom
- **Status**: IN_PROGRESS
- **Workflow state**: VALIDATION
- **Created**: 2026-09-26

## Goal and scope

Provide production-grade deployment artifacts (Dockerfile, compose.yml, and systemd service unit) for running the `paxlet-com/nl-dsl-llm` daemon with shared `/run/paxlet` socket volume access across Taskand, Clonerd, and `nl-dsl-sh`.

## Acceptance criteria

- [x] AC-01: Production multi-stage `Dockerfile` with non-root runtime and socket volume exposure.
- [x] AC-02: `compose.yml` defining `nl-runtime` container with CPU/memory limits and host gateway access.
- [x] AC-03: `infra/systemd/nl-runtime.service` unit file for host-level daemon execution.
- [x] AC-04: Documentation in `infra/README.md` compliant with `wellmanifest/docs` v2 Compact Profile.
- [x] AC-05: Governance checks pass (`./project/governance-check.sh`).

## Tracking boundary

This directory contains the minimal reviewed intent. Optional participant prose
and raw command logs are not required delivery output.
