# Ticket 005: Fix pycache gitignore leak and pytest sessionstart failure

- **ID**: ticket-005
- **Owner**: unresolved:human
- **Status**: IN_PROGRESS
- **Workflow state**: EDIT
- **Created**: 2026-09-28

## Goal and scope

Prevent Python bytecode (`__pycache__/`, `*.pyc`) from leaking into git status and breaking `wellmanifest_governance.py` during `pytest_sessionstart`.

## Acceptance criteria

- [x] AC-01: `.gitignore` ignores `__pycache__/` and Python bytecode files.
- [x] AC-02: Running `pytest` does not produce dirty untracked git files triggering GOV-TICKET-001.

## Tracking boundary

This directory contains the minimal reviewed intent. Optional participant prose
and raw command logs are not required delivery output.
