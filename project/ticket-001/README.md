# Ticket 001: gRPC runtime and dual Qwen benchmark suite

- **ID**: ticket-001
- **Owner**: unresolved:human
- **Status**: IN_PROGRESS
- **Workflow state**: EDIT
- **Created**: 2026-09-26

## Goal and scope

Implement the production gRPC runtime service and evaluation harness for `paxlet-com/nl-dsl-llm`.
Provides low-latency IPC for Taskand, Clonerd, and nl-dsl-sh, with comparative benchmarking
for Qwen3.5-2B and Qwen3.5-0.8B running on local CPU via `llama.cpp` + LiteLLM.

## Acceptance criteria

- [x] AC-01: Normative Protobuf contract defined in `proto/paxlet/nl_dsl_llm/v1/runtime.proto`.
- [x] AC-02: Python gRPC server implementing `NLRuntimeService` with UDS and TCP support.
- [x] AC-03: 4-layer resolution engine (Regex Fast-Path, Semantic Cache with slot/negation guard, LLM fallback).
- [x] AC-04: Dual-model benchmark harness comparing Qwen3.5-2B vs Qwen3.5-0.8B on VPS CPU.
- [x] AC-05: Documentation adhering to `wellmanifest/docs` v2 compact profile.
- [x] AC-06: Verification passes `./project/governance-check.sh` and test suite.

## Tracking boundary

This directory contains the minimal reviewed intent. Optional participant prose
and raw command logs are not required delivery output.
