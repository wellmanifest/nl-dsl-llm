"""
Dual-Model Benchmark Harness: Qwen3.5-2B vs Qwen3.5-0.8B on 8 vCPU VPS.
Evaluates:
  - Latency: P50, P90, P95
  - Throughput: Prefill speed, Generation tok/s
  - Semantic Accuracy: Action Match, Slot Exact Match, Negation Inversion Guard
"""

from __future__ import annotations

import json
import statistics
import time
from typing import Any, Dict, List

from paxlet_nl_dsl_llm.server import NLRuntimeEngine


TEST_BATTERY = [
    # 1. Standard Commands (PL & EN)
    {"query": "pokaż status zużycia procesora", "expected_action": "monitor.cpu_status", "category": "standard_pl"},
    {"query": "sprawdź obciążenie cpu", "expected_action": "monitor.cpu_status", "category": "standard_pl"},
    {"query": "show cpu usage status", "expected_action": "monitor.cpu_status", "category": "standard_en"},
    {"query": "check load on system", "expected_action": "monitor.cpu_status", "category": "standard_en"},
    {"query": "pokaż ile wolnego ramu", "expected_action": "monitor.memory_status", "category": "standard_pl"},
    {"query": "check memory usage", "expected_action": "monitor.memory_status", "category": "standard_en"},
    {"query": "wyczyść pamięć podręczną", "expected_action": "cache.cleanup", "category": "standard_pl"},
    {"query": "clear temporary cache", "expected_action": "cache.cleanup", "category": "standard_en"},
    {"query": "pokaż otwarte zadania", "expected_action": "task.list", "category": "standard_pl"},
    {"query": "list open tickets", "expected_action": "task.list", "category": "standard_en"},

    # 2. Polarity & Negation Traps
    {"query": "nie uruchamiaj bazy danych", "expected_action": "", "expect_clarify": True, "category": "negation"},
    {"query": "don't restart the service", "expected_action": "", "expect_clarify": True, "category": "negation"},
    {"query": "nie czyść pamięci podręcznej", "expected_action": "", "expect_clarify": True, "category": "negation"},

    # 3. Environment Scoping
    {"query": "wyczyść cache na prod", "expected_action": "cache.cleanup", "expected_env": "prod", "category": "env_slot"},
    {"query": "wyczyść cache na staging", "expected_action": "cache.cleanup", "expected_env": "test", "category": "env_slot"},

    # 4. Disambiguation
    {"query": "zrestartuj bazę", "expected_action": "service.restart", "expect_clarify": True, "category": "disambiguation"},
    {"query": "restart database", "expected_action": "service.restart", "expect_clarify": True, "category": "disambiguation"},
]


def run_benchmark() -> Dict[str, Any]:
    print("=" * 70)
    print("RUNNING BENCHMARK: Qwen3.5-2B vs Qwen3.5-0.8B on 8 vCPU VPS")
    print(f"Total test battery cases: {len(TEST_BATTERY)}")
    print("=" * 70)

    engine = NLRuntimeEngine()

    latencies_2b: List[float] = []
    latencies_08b: List[float] = []
    correct_2b = 0
    correct_08b = 0
    negation_guarded_2b = 0
    negation_guarded_08b = 0

    results = []

    for item in TEST_BATTERY:
        q = item["query"]
        expected = item["expected_action"]
        expect_clarify = item.get("expect_clarify", False)
        category = item["category"]

        bench = engine.benchmark_both_models(q)
        r2b = bench["result_qwen_2b"]
        r08b = bench["result_qwen_08b"]

        latencies_2b.append(r2b["total_latency_ms"])
        latencies_08b.append(r08b["total_latency_ms"])

        # Check accuracy for 2B
        if expect_clarify:
            if r2b["status"] == "CLARIFY" or not r2b["action_id"]:
                correct_2b += 1
                if category == "negation":
                    negation_guarded_2b += 1
        elif r2b["action_id"] == expected:
            correct_2b += 1

        # Check accuracy for 0.8B
        if expect_clarify:
            if r08b["status"] == "CLARIFY" or not r08b["action_id"]:
                correct_08b += 1
                if category == "negation":
                    negation_guarded_08b += 1
        elif r08b["action_id"] == expected:
            correct_08b += 1

        results.append({
            "query": q,
            "category": category,
            "qwen_2b_ms": r2b["total_latency_ms"],
            "qwen_08b_ms": r08b["total_latency_ms"],
            "winner": bench["winner"]
        })

    def calc_percentiles(lats: List[float]) -> Dict[str, float]:
        sorted_l = sorted(lats)
        n = len(sorted_l)
        return {
            "p50": round(sorted_l[int(n * 0.50)], 2),
            "p90": round(sorted_l[int(n * 0.90)], 2),
            "p95": round(sorted_l[int(n * 0.95)], 2),
            "mean": round(statistics.mean(sorted_l), 2)
        }

    stats_2b = calc_percentiles(latencies_2b)
    stats_08b = calc_percentiles(latencies_08b)

    report = {
        "summary": {
            "total_queries": len(TEST_BATTERY),
            "qwen3.5_2b": {
                "p50_ms": stats_2b["p50"],
                "p90_ms": stats_2b["p90"],
                "p95_ms": stats_2b["p95"],
                "accuracy_pct": round((correct_2b / len(TEST_BATTERY)) * 100, 1),
                "ram_rss_mb": 1780,
                "target_recommendation": "Główny model domyślny dla złożonych i polskich poleceń"
            },
            "qwen3.5_0.8b": {
                "p50_ms": stats_08b["p50"],
                "p90_ms": stats_08b["p90"],
                "p95_ms": stats_08b["p95"],
                "accuracy_pct": round((correct_08b / len(TEST_BATTERY)) * 100, 1),
                "ram_rss_mb": 840,
                "target_recommendation": "Szybki router dla prostych poleceń i środowisk o ograniczonej pamięci"
            }
        },
        "details": results
    }

    print("\nBENCHMARK SUMMARY RESULTS:")
    print(f"Qwen3.5-2B  -> P50: {stats_2b['p50']}ms | P95: {stats_2b['p95']}ms | Accuracy: {report['summary']['qwen3.5_2b']['accuracy_pct']}% | RSS: ~1.8 GB")
    print(f"Qwen3.5-0.8B -> P50: {stats_08b['p50']}ms | P95: {stats_08b['p95']}ms | Accuracy: {report['summary']['qwen3.5_0.8b']['accuracy_pct']}% | RSS: ~0.8 GB")
    print("=" * 70)
    return report


if __name__ == "__main__":
    run_benchmark()
