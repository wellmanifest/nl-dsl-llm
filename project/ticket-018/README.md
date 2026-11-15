# Ticket 018: Adopt published governance standard 0.20.151 and errors catalog

- **ID**: ticket-018
- **Owner**: agent:antigravity
- **Status**: IN_PROGRESS
- **Workflow state**: PUBLICATION
- **Created**: 2026-11-15

## Goal and scope

`SESSION_EXECUTION_AUTHORIZATION`: User authorized Wave 2 Scope 3 alignment of `wellmanifest/nl-dsl-llm` to published `wellmanifest/new-project` `0.20.151` (`7ac3106162c9e7a9761b4ae6726511eb8935f87a`) and `wellman v0.23.0` (`wellmanifest/docs` and `wellmanifest/logs`). Opt in to `.governance/ticket-activity.override.json` (`git-ancestry`) so already-merged historical `IN_PROGRESS` tickets resolve cleanly via Git ancestry, add `errors/README.md`, and refresh `.governance/standard-adoption.json`.

## Acceptance criteria

- [x] AC-01: Complete immutable `0.20.151` package adopted, `.governance/ticket-activity.override.json` and `errors/README.md` added, and `./project/governance-check.sh`, `standard_pack_check.py`, and `wellman check` pass.
