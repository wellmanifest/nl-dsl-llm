"""
Complete test suite for paxlet-com/nl-dsl-llm.
"""

from paxlet_nl_dsl_llm.layers import (
    DigitalTwinContext,
    DigitalTwinPromptSummarizer,
    FastPathParser,
    MODEL_PROFILES,
    SemanticCache,
    SingleResidentModelManager,
)
from paxlet_nl_dsl_llm.server import NLRuntimeEngine


def test_fast_path_parsing():
    parser = FastPathParser()
    res_cpu = parser.parse("pokaż procesor")
    assert res_cpu is not None
    assert res_cpu[0] == "monitor.cpu_status"

    res_mem = parser.parse("status ram")
    assert res_mem is not None
    assert res_mem[0] == "monitor.memory_status"

    # Negation check
    assert parser.parse("nie czyść cache") is None
    assert parser.parse("don't restart") is None


def test_semantic_cache_polarity_and_slots():
    cache = SemanticCache()
    assert cache.check_polarity("uruchom serwer") is False
    assert cache.check_polarity("nie uruchamiaj serwera") is True
    assert cache.check_polarity("don't delete anything") is True

    slots_prod = cache.extract_slots("wyczyść cache na prod")
    assert slots_prod.get("environment") == "prod"

    # Positive query matches cache
    assert cache.lookup("zrestartuj proces bazy danych") is not None
    # Negated query fails closed
    assert cache.lookup("nie restartuj bazy danych") is None


def test_digital_twin_prompt_summarizer():
    summarizer = DigitalTwinPromptSummarizer(max_candidates=6)
    twin = DigitalTwinContext(
        environment="prod",
        active_services=["postgres", "redis-cache", "worker"],
        degraded_services=["redis-cache"]
    )
    summary = summarizer.summarize(twin, query="restart")
    assert summary["has_degraded"] is True
    assert summary["candidate_count"] <= 6
    # Top candidate must be the degraded entity
    assert summary["pruned_candidates"][0]["target"] == "redis-cache"
    assert summary["pruned_candidates"][0]["priority"] == "HIGH_DEGRADED"
    assert "redis-cache" in summary["prompt_snippet"]


def test_single_resident_model_manager_and_profiles():
    manager = SingleResidentModelManager(default_profile="qwen3_1.7b")
    active = manager.get_active_profile()
    assert active.id == "qwen3_1.7b"
    assert active.priority == 1
    assert active.recommended_settings["enable_thinking"] is False

    # Check that all 6 models satisfy the single-resident memory invariant (< 2.5 GB RSS)
    for pid, prof in MODEL_PROFILES.items():
        can_switch, reason = manager.can_switch_to(pid)
        assert can_switch is True, f"Model {pid} failed memory invariant: {reason}"
        assert prof.ram_rss_mb <= 2500, f"Model {pid} exceeds 2.5 GB resident limit"

    # Switch profile
    switched = manager.switch_profile("qwen3.5_2b")
    assert switched.id == "qwen3.5_2b"
    assert switched.official_languages_count == 201


def test_runtime_engine_routing_and_benchmarking():
    engine = NLRuntimeEngine()
    # Client URI resolution
    res_t = engine.interpret("status ram", client_id="taskand")
    assert res_t["resolved_procedure_uri"] == "proc://taskand.dev/monitor/memory/v1"
    assert res_t["matched_layer"] == "SOURCE_LAYER_FAST_PATH_REGEX"

    res_s = engine.interpret("status ram", client_id="nl-dsl-sh")
    assert res_s["resolved_procedure_uri"] == "urn:nl-dsl-sh:sys:ram-check"

    # Disambiguation
    res_c = engine.interpret("zrestartuj bazę")
    assert res_c["status"] == "CLARIFY"
    assert len(res_c["clarification_options"]) > 0

    # Dual-model benchmark (legacy compatibility)
    bench = engine.benchmark_both_models("sprawdź użycie procesora")
    assert "result_qwen_2b" in bench
    assert "result_qwen_08b" in bench
    assert bench["winner"] in ("qwen_2b", "qwen_08b")

    # Multi-model matrix benchmark
    matrix_bench = engine.benchmark_matrix("pokaż zużycie ramu")
    assert "evaluations" in matrix_bench
    assert "qwen3_1.7b" in matrix_bench["evaluations"]
    assert "qwen3.5_2b" in matrix_bench["evaluations"]
    assert matrix_bench["recommended_production_default"] == "qwen3_1.7b"


def test_active_learning_cache_promotion():
    engine = NLRuntimeEngine()
    # 1. Uncached novel query without auto-promote
    novel_q = "zrób kopię zapasową bazy postgres"
    res1 = engine.interpret(novel_q, auto_promote=False)
    assert res1["status"] in ("MATCHED", "CLARIFY")
    assert res1["matched_layer"].startswith("SOURCE_LAYER_LLM")

    # Explicit promotion of verified intent
    ok = engine.promote_intent(novel_q, "database.backup", {"target": "postgres"})
    assert ok is True

    # 2. Subsequent query hits Layer 1.5 Semantic Cache directly
    res2 = engine.interpret(novel_q)
    assert res2["status"] == "MATCHED"
    assert res2["action_id"] == "database.backup"
    assert res2["matched_layer"] == "SOURCE_LAYER_SEMANTIC_CACHE"

    # 3. Test automatic promotion during interpret
    novel_q2 = "wykonaj pełną migrację bazy danych"
    res3 = engine.interpret(novel_q2, auto_promote=True)
    assert res3["status"] in ("MATCHED", "CLARIFY")
    # Query is now automatically in cache
    res4 = engine.interpret(novel_q2)
    assert res4["matched_layer"] == "SOURCE_LAYER_SEMANTIC_CACHE"


def test_system_telemetry_and_digital_twin():
    engine = NLRuntimeEngine()
    twin = engine.observe_telemetry(environment="staging")
    assert twin.environment == "staging"
    assert len(twin.active_services) >= 1
    assert "vps-01-primary" in twin.cluster_nodes
    # Verify reading metrics without error
    metrics = engine.telemetry.read_metrics()
    assert "mem_total_mb" in metrics
    assert metrics["mem_total_mb"] > 0


def test_voice_token_stream_handler():
    engine = NLRuntimeEngine()
    # Streaming speech tokens from Whisper / VAD
    voice_res = engine.stream_voice_token("zre")
    assert "options" in voice_res
    assert voice_res["latency_ms"] < 2.0  # sub-2ms response
    assert len(voice_res["options"]) >= 1
    # Check that next_options DAG is attached
    assert len(voice_res["options"][0]["next_options"]) >= 1


def test_federated_openapi_import():
    engine = NLRuntimeEngine()
    sample_openapi = {
        "openapi": "3.0.0",
        "paths": {
            "/api/v1/clusters/nodes": {
                "get": {
                    "operationId": "list_nodes",
                    "summary": "Pokaż listę węzłów klastra",
                    "description": "Returns active cluster compute nodes"
                }
            },
            "/api/v1/cache/flush": {
                "post": {
                    "operationId": "flush_cache",
                    "summary": "Wyczyść cache klastra",
                    "description": "Purges distributed cluster cache"
                }
            }
        }
    }
    imported_count = engine.registry.import_openapi_schema(sample_openapi, client_id="cluster_api")
    assert imported_count == 2
    act = engine.registry.get("api.list_nodes")
    assert act is not None
    assert act.title == "Pokaż listę węzłów klastra"
    assert act.uri_patterns["cluster_api"] == "openapi:///api/v1/clusters/nodes#GET"


def test_single_word_queries_across_all_commands():
    """Validates deterministic single-word resolution across 17 commands (3+ words each)."""
    engine = NLRuntimeEngine()
    test_cases = [
        ("monitor.cpu_status", ["cpu", "procesor", "obciążenie"]),
        ("monitor.memory_status", ["ram", "pamięć", "memory"]),
        ("cache.cleanup", ["cache", "wyczyść", "purge"]),
        ("service.restart", ["restart", "zrestartuj", "reboot"]),
        ("task.list", ["zadania", "tickety", "tasks"]),
        ("task.schedule", ["harmonogram", "zaplanuj", "schedule"]),
        ("task.get", ["szczegóły", "inspect", "dane"]),
        ("task.create", ["utwórz", "nowy", "create"]),
        ("task.close", ["zakończ", "zamknij", "complete"]),
        ("worktree.allocate", ["worktree", "alokuj", "gałąź"]),
        ("koru.doctor", ["doctor", "napraw", "diagnostyka"]),
        ("koru.queue.next", ["kolejka", "następne", "next"]),
        ("koru.queue.loop", ["pętla", "loop", "automatyzacja"]),
        ("koru.context", ["kontekst", "brief", "context"]),
        ("pr.list", ["pr", "prs", "pulls"]),
        ("token.list", ["token", "tokeny", "zużycie"]),
        ("process.list", ["procesy", "agenty", "processes"]),
    ]

    for expected_action, words in test_cases:
        for word in words:
            res = engine.interpret(word, client_id="taskand")
            assert res["action_id"] == expected_action, f"Failed for single-word '{word}': got {res['action_id']}"
            assert res["status"] in ("MATCHED", "CLARIFY"), f"Unexpected status for '{word}': {res['status']}"
            assert res["matched_layer"] == "SOURCE_LAYER_FAST_PATH_REGEX"
            assert len(res["resolved_procedure_uri"]) > 0

