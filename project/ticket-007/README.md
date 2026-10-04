# Ticket 007: Conversational stream purity and state URL sync conformance

- **ID**: ticket-007
- **Owner**: agent:antigravity
- **Status**: IN_PROGRESS
- **Workflow state**: EDIT
- **Created**: 2026-10-04

## Goal and scope

SESSION_EXECUTION_AUTHORIZATION: The user requested extending wellmanifest/nl-* standards with conversational stream purity, process view separation, and bidirectional URL state synchronization tested in practice.
In `wellmanifest/nl-dsl-llm`, update `spec/NL_DSL_LLM_SPECIFICATION.md`:
1. Add section 3.7: Conversational Stream Purity & Dedicated Process View (NUL-007 Binding).
2. Add section 3.8: Bidirectional Interactive State URL Synchronization (NUL-008 Binding).
3. Add conformance criteria [CONF-09] (Conversational Stream Purity) and [CONF-10] (Deterministic Bidirectional State URL Sync).

## Acceptance criteria

- [x] AC-01: Update `spec/NL_DSL_LLM_SPECIFICATION.md` with sections 3.7, 3.8 and conformance criteria CONF-09, CONF-10.
- [x] AC-02: Native governance check (`./project/governance-check.sh`) passes without errors.
