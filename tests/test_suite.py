"""
Complete test suite for paxlet-com/nl-dsl-llm.
"""

from paxlet_nl_dsl_llm.layers import FastPathParser, SemanticCache
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

    # Dual-model benchmark
    bench = engine.benchmark_both_models("sprawdź użycie procesora")
    assert "result_qwen_2b" in bench
    assert "result_qwen_08b" in bench
    assert bench["winner"] in ("qwen_2b", "qwen_08b")
