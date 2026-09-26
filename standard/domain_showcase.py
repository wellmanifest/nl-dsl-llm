#!/usr/bin/env python3
"""
Multi-Domain Showcase for wellmanifest/nl-dsl-llm standard:
Demonstrates where the standard and runtime are applied across four concrete domains:
  1. Taskand (Task Planning, API Menu Context Routing, proc:// URIs)
  2. Clonerd (Git Worktree Allocation, Repo Orchestration, cluster. URIs)
  3. SRE / Autonomous DevOps (Self-Healing, Digital Twin Degraded Entity Pruning)
  4. nl-dsl-sh / Voice Terminal (Streaming Token Autocomplete & Option Network DAG)
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time
from typing import Any, Dict, List

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from standard.nl_dsl_llm import (
    DSLCommand,
    DSLExecutor,
    DigitalTwinContext,
    NLDSLLLMBridge,
    NLIntentParser,
    OptionNetworkEngine,
)


def run_showcase() -> Dict[str, Any]:
    print("=" * 80)
    print("NL-DSL-LLM MULTI-DOMAIN APPLICABILITY & EXECUTION SHOWCASE")
    print("Standard: wellmanifest/nl-dsl-llm v0.2.1-candidate")
    print("=" * 80)

    executor = DSLExecutor()
    results: Dict[str, Any] = {}

    # Register Domain Handlers in Canonical DSL Executor (Layer 2)
    # 1. Taskand Domain Handler
    def handle_task_schedule(cmd: DSLCommand):
        return {
            "entity": "task",
            "operation": "schedule",
            "procedure_uri": "proc://taskand.dev/planner/schedule/v1",
            "scheduled_title": cmd.arguments.get("title", "Routine health scan"),
            "environment": cmd.get_param("environment", "prod"),
            "status": "ENQUEUED"
        }
    executor.register("task", "schedule", handle_task_schedule)
    executor.register("task", "list", lambda cmd: [{"id": "task-001", "status": "OPEN"}, {"id": "task-002", "status": "DONE"}])

    # 2. Clonerd Domain Handler
    def handle_worktree_allocate(cmd: DSLCommand):
        ticket_id = cmd.arguments.get("ticket", "ticket-001")
        return {
            "entity": "worktree",
            "operation": "allocate",
            "procedure_uri": "cluster.worktree.allocate",
            "ticket": ticket_id,
            "path": f".worktrees/{ticket_id}--isolated",
            "link_mode": "relative",
            "status": "ALLOCATED"
        }
    executor.register("worktree", "allocate", handle_worktree_allocate)

    # 3. SRE Remediation Handler
    def handle_service_restart(cmd: DSLCommand):
        target = cmd.arguments.get("target", "redis-cache")
        env = cmd.get_param("environment", "prod")
        return {
            "entity": "service",
            "operation": "restart",
            "procedure_uri": "proc://ops.sre/remediate/v1",
            "target": target,
            "environment": env,
            "requires_confirmation": env == "prod",
            "status": "RESTARTED"
        }
    executor.register("service", "restart", handle_service_restart)

    bridge = NLDSLLLMBridge(executor)

    # =========================================================================
    # Domain 1: Taskand Task Planning & API Menu Context Routing
    # =========================================================================
    print("\n--- [DOMAIN 1: Taskand (Task Planning & API Menu Context Routing)] ---")
    q1 = "task.schedule title='Audit system memory' environment=prod"
    t0 = time.perf_counter()
    res1 = bridge.handle_request(q1)
    lat1 = (time.perf_counter() - t0) * 1000
    print(f"DSL Input:    '{q1}'")
    print(f"Layer:        {res1.meta['sourceLayer']} (Latency: {lat1:.2f}ms)")
    print(f"Result URI:   {res1.data['procedure_uri']}")
    print(f"Status:       {res1.data['status']}")
    assert res1.success is True
    assert res1.data["procedure_uri"] == "proc://taskand.dev/planner/schedule/v1"
    results["domain_1_taskand"] = {"input": q1, "uri": res1.data["procedure_uri"], "latency_ms": lat1}

    # =========================================================================
    # Domain 2: Clonerd Git Worktree & Repository Orchestration
    # =========================================================================
    print("\n--- [DOMAIN 2: Clonerd (Git Worktree & Repo Orchestration)] ---")
    q2 = "worktree.allocate ticket=ticket-005"
    t0 = time.perf_counter()
    res2 = bridge.handle_request(q2)
    lat2 = (time.perf_counter() - t0) * 1000
    print(f"DSL Input:    '{q2}'")
    print(f"Layer:        {res2.meta['sourceLayer']} (Latency: {lat2:.2f}ms)")
    print(f"Result Path:  {res2.data['path']}")
    print(f"Link Mode:    {res2.data['link_mode']}")
    assert res2.success is True
    assert res2.data["procedure_uri"] == "cluster.worktree.allocate"
    results["domain_2_clonerd"] = {"input": q2, "path": res2.data["path"], "latency_ms": lat2}

    # =========================================================================
    # Domain 3: Autonomous SRE & Self-Healing with Digital Twin Projection
    # =========================================================================
    print("\n--- [DOMAIN 3: SRE Self-Healing & Digital Twin Degraded Entity Pruning] ---")
    twin_ctx = DigitalTwinContext(
        environment="prod",
        active_services=["postgres", "redis-cache", "gateway"],
        degraded_services=["redis-cache"],
        active_tickets=["ticket-001", "ticket-002"]
    )
    t0 = time.perf_counter()
    sugg_res = bridge.suggest_options("zre", twin_context=twin_ctx, max_suggestions=3)
    lat3 = (time.perf_counter() - t0) * 1000
    top_sre = sugg_res["suggestions"][0]
    print(f"Input Token:  'zre' (Streaming typing / partial speech)")
    print(f"Top Option:   {top_sre['display_label']} (Badge: {top_sre['badge']}, Conf: {top_sre['confidence']})")
    print(f"Resolved URI: {top_sre['resolved_uri']}")
    print(f"Latency:      {lat3:.2f}ms (<1ms deterministic target)")
    print(f"Next Options: {[n['display_label'] for n in top_sre.get('next_options', [])]}")
    assert top_sre["badge"] == "DEGRADED"
    assert "redis-cache" in top_sre["completion_text"]
    assert top_sre["requires_confirmation"] is True
    results["domain_3_sre"] = {"input": "zre", "top_option": top_sre["display_label"], "badge": top_sre["badge"], "latency_ms": lat3}

    # =========================================================================
    # Domain 4: Voice / Terminal (nl-dsl-sh) Real-Time Autocomplete DAG
    # =========================================================================
    print("\n--- [DOMAIN 4: Voice / Terminal nl-dsl-sh Real-Time Autocomplete DAG] ---")
    t0 = time.perf_counter()
    voice_res = bridge.suggest_options("pok", twin_context=twin_ctx, max_suggestions=5)
    lat4 = (time.perf_counter() - t0) * 1000
    print(f"Speech Token: 'pok' (Partial speech token from Whisper/VAD)")
    print(f"DAG Count:    {len(voice_res['suggestions'])} options generated in {lat4:.2f}ms")
    for idx, opt in enumerate(voice_res["suggestions"][:3], 1):
        print(f"  {idx}. {opt['display_label']:<35} -> {opt['completion_text']}")
    results["domain_4_voice_cli"] = {"input": "pok", "count": len(voice_res["suggestions"]), "latency_ms": lat4}

    print("\n" + "=" * 80)
    print("ALL DOMAIN SHOWCASES COMPLETED SUCCESSFULLY UNDER STRICT GOVERNANCE")
    print("=" * 80)
    return results


if __name__ == "__main__":
    run_showcase()
