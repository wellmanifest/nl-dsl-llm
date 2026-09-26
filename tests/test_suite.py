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
