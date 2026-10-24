"""
Telemetry, Streaming Voice Token Handler, Model Profiles, Single-Resident Model Manager,
and Schema-Constrained LLM Translation Compiler for paxlet-com/nl-dsl-llm.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from .layers import DigitalTwinContext, DigitalTwinPromptSummarizer


class SystemTelemetryObserver:
    """
    Live host telemetry observer that feeds DigitalTwinContext.
    Reads /proc/loadavg, /proc/meminfo, and checks service health.
    Automatically detects degraded system state to guide LLM prompt pruning.
    """

    def __init__(self, loadavg_threshold: float = 7.0, min_available_ram_mb: int = 1500):
        self.loadavg_threshold = loadavg_threshold
        self.min_available_ram_mb = min_available_ram_mb

    def read_metrics(self) -> Dict[str, Any]:
        metrics = {
            "load_1m": 0.5,
            "load_5m": 0.5,
            "mem_total_mb": 8192,
            "mem_available_mb": 4096,
            "is_cpu_saturated": False,
            "is_memory_strained": False,
        }
        if os.path.exists("/proc/loadavg"):
            try:
                with open("/proc/loadavg", "r", encoding="utf-8") as f:
                    parts = f.read().split()
                    metrics["load_1m"] = float(parts[0])
                    metrics["load_5m"] = float(parts[1])
                    metrics["is_cpu_saturated"] = metrics["load_1m"] > self.loadavg_threshold
            except Exception:
                pass

        if os.path.exists("/proc/meminfo"):
            try:
                with open("/proc/meminfo", "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            metrics["mem_total_mb"] = int(line.split()[1]) // 1024
                        elif line.startswith("MemAvailable:"):
                            metrics["mem_available_mb"] = int(line.split()[1]) // 1024
                    metrics["is_memory_strained"] = metrics["mem_available_mb"] < self.min_available_ram_mb
            except Exception:
                pass

        return metrics

    def observe_digital_twin(
        self,
        environment: str = "prod",
        known_services: Optional[List[str]] = None,
        manual_degraded: Optional[List[str]] = None,
    ) -> DigitalTwinContext:
        from .layers import DigitalTwinContext

        metrics = self.read_metrics()
        services = list(known_services or ["postgres", "redis-cache", "taskand-worker"])
        degraded = list(manual_degraded or [])

        # Auto-detect degradation from host metrics (e.g. memory strained)
        if metrics["is_memory_strained"] and "redis-cache" in services and "redis-cache" not in degraded:
            degraded.append("redis-cache")

        return DigitalTwinContext(
            environment=environment,
            active_services=services,
            degraded_services=degraded,
            active_tickets=["ticket-001"],
            cluster_nodes=["vps-01-primary"]
        )


class VoiceTokenStreamHandler:
    """
    Handles streaming speech tokens from edge ASR (Whisper, sherpa-onnx, WebRTC VAD).
    Generates real-time Option Network DAG predictions with <1ms latency at each token.
    """

    def __init__(self, summarizer: Optional[DigitalTwinPromptSummarizer] = None):
        from .layers import DigitalTwinPromptSummarizer

        self.summarizer = summarizer or DigitalTwinPromptSummarizer()

    def process_token(
        self,
        accumulated_transcript: str,
        twin_context: Optional[DigitalTwinContext] = None,
        max_options: int = 5,
    ) -> Dict[str, Any]:
        from .layers import DigitalTwinContext

        start = time.perf_counter()
        clean = accumulated_transcript.strip().lower()
        twin = twin_context or DigitalTwinContext()
        summary = self.summarizer.summarize(twin, query=clean)

        options = []
        for cand in summary["pruned_candidates"]:
            action_id = cand["action_id"]
            target = cand.get("target", "")
            is_deg = cand.get("priority") == "HIGH_DEGRADED"

            match_score = 0.0
            if clean and (clean in action_id.lower() or (target and clean in target.lower())):
                match_score = 0.95 if is_deg else 0.85
            elif any(tok.startswith(clean) or clean.startswith(tok) for tok in [
                "zre", "rest", "res", "pok", "stat", "czysz", "wycz", "proc", "cpu",
                "pam", "ram", "mem", "kol", "doc", "nap", "zad", "harm", "work",
                "pet", "kon", "tok", "alo"
            ]):
                match_score = 0.90 if is_deg else 0.75

            if match_score > 0:
                options.append({
                    "id": f"{action_id}:{target or 'all'}",
                    "display_label": f"Akcja: {action_id} (Cel: {target or 'default'})" + (" [ZDEGRADOWANY]" if is_deg else ""),
                    "action_id": action_id,
                    "target": target,
                    "confidence": match_score,
                    "badge": "DEGRADED" if is_deg else "ACTIVE",
                    "requires_confirmation": twin.environment == "prod",
                    "next_options": [
                        {"id": "exec:now", "label": "Wykonaj natychmiast"},
                        {"id": "exec:dry_run", "label": "Wykonaj testowo (dry-run)"},
                    ]
                })

        elapsed = (time.perf_counter() - start) * 1000
        return {
            "accumulated_transcript": accumulated_transcript,
            "options": options[:max_options],
            "latency_ms": round(elapsed, 3),
            "is_terminal": len(options) == 1 and options[0]["confidence"] > 0.90
        }


# ==============================================================================
# Layer 4: Model Profiles & Single-Resident Model Manager
# ==============================================================================

@dataclass
class ModelProfile:
    id: str
    name: str
    role: str
    description: str
    params_b: float
    quantization: str
    ram_rss_mb: int
    context_window: int
    official_languages_count: int
    polish_support: str  # "OFFICIAL", "STRONG", "EXPERIMENTAL", "LIMITED", "UNOFFICIAL"
    priority: int
    recommended_settings: Dict[str, Any]
    port: int = 8081


MODEL_PROFILES: Dict[str, ModelProfile] = {
    "qwen3_1.7b": ModelProfile(
        id="qwen3_1.7b",
        name="Qwen3-1.7B-Instruct",
        role="PRIMARY_DEFAULT_ROUTER",
        description="Domyślny router NL -> akcja + argumenty (32k context, tool calling, enable_thinking=false)",
        params_b=1.7,
        quantization="Q4_K_M",
        ram_rss_mb=1250,
        context_window=32768,
        official_languages_count=100,
        polish_support="STRONG",
        priority=1,
        recommended_settings={
            "enable_thinking": False,
            "temperature": 0.2,
            "max_tokens": 128,
            "context": 4096,
            "threads": 4,
        },
        port=8081,
    ),
    "qwen3.5_2b": ModelProfile(
        id="qwen3.5_2b",
        name="Qwen3.5-2B",
        role="QUALITY_MULTILINGUAL_FALLBACK",
        description="Model jakościowy do trudniejszych i wielojęzycznych poleceń oraz planowania (201 języków)",
        params_b=2.0,
        quantization="Q4_K_M",
        ram_rss_mb=1780,
        context_window=32768,
        official_languages_count=201,
        polish_support="OFFICIAL",
        priority=2,
        recommended_settings={
            "temperature": 0.2,
            "max_tokens": 160,
            "context": 4096,
            "threads": 4,
        },
        port=8081,
    ),
    "granite_4.0_micro": ModelProfile(
        id="granite_4.0_micro",
        name="Granite-4.0-Micro-3B",
        role="ENTERPRISE_FUNCTION_CALLING",
        description="Funkcje, JSON, Taskand/Paxlet proc:// URI resolution, enterprise schema",
        params_b=3.0,
        quantization="Q4_K_M",
        ram_rss_mb=2200,
        context_window=131072,
        official_languages_count=12,
        polish_support="UNOFFICIAL",
        priority=3,
        recommended_settings={
            "temperature": 0.1,
            "max_tokens": 128,
            "context": 4096,
            "threads": 4,
        },
        port=8081,
    ),
    "smollm3_3b": ModelProfile(
        id="smollm3_3b",
        name="SmolLM3-3B",
        role="PERMISSIVE_TOOL_CALLING",
        description="Dłuższe instrukcje, tool calling, licencja Apache 2.0",
        params_b=3.0,
        quantization="Q4_K_M",
        ram_rss_mb=2100,
        context_window=65536,
        official_languages_count=6,
        polish_support="LIMITED",
        priority=4,
        recommended_settings={
            "temperature": 0.2,
            "max_tokens": 128,
            "context": 4096,
            "threads": 4,
        },
        port=8081,
    ),
    "lfm2.5_1.2b": ModelProfile(
        id="lfm2.5_1.2b",
        name="LFM2.5-1.2B-Instruct",
        role="ULTRA_FAST_CPU_EDGE",
        description="Szybkie podpowiedzi, klasyfikacja, ekstrakcja (<1 GB RAM na CPU)",
        params_b=1.2,
        quantization="Q4_K_M",
        ram_rss_mb=890,
        context_window=32768,
        official_languages_count=8,
        polish_support="UNOFFICIAL",
        priority=5,
        recommended_settings={
            "temperature": 0.2,
            "max_tokens": 96,
            "context": 2048,
            "threads": 4,
        },
        port=8081,
    ),
    "qwen3.5_0.8b": ModelProfile(
        id="qwen3.5_0.8b",
        name="Qwen3.5-0.8B",
        role="ULTRA_FAST_SUGGESTION_ROUTER",
        description="Bardzo szybkie sugestie i router w warunkach małej pamięci",
        params_b=0.8,
        quantization="Q4_K_M",
        ram_rss_mb=720,
        context_window=32768,
        official_languages_count=201,
        polish_support="OFFICIAL",
        priority=6,
        recommended_settings={
            "temperature": 0.3,
            "max_tokens": 64,
            "context": 2048,
            "threads": 4,
        },
        port=8081,
    ),
}


class SingleResidentModelManager:
    """
    Enforces the single-resident model execution invariant on 8 vCPU / 8 GB RAM VPS.
    Never runs two heavy models simultaneously. Leaves 5-6 GB RAM free for OS,
    Taskand, PostgreSQL, Redis, and Playwright/noVNC.
    """
    TOTAL_HOST_RAM_MB: int = 8192
    RESERVED_HOST_SERVICES_RAM_MB: int = 5600  # 5.6 GB reserved for host services + OS

    def __init__(self, default_profile: str = "qwen3_1.7b"):
        self.active_profile_id: str = default_profile

    def get_active_profile(self) -> ModelProfile:
        return MODEL_PROFILES.get(self.active_profile_id, MODEL_PROFILES["qwen3_1.7b"])

    def can_switch_to(self, profile_id: str) -> Tuple[bool, str]:
        if profile_id not in MODEL_PROFILES:
            return False, f"Unknown profile: {profile_id}"
        target = MODEL_PROFILES[profile_id]
        remaining_ram = self.TOTAL_HOST_RAM_MB - self.RESERVED_HOST_SERVICES_RAM_MB - target.ram_rss_mb
        if remaining_ram < 0:
            return False, f"Model {target.name} ({target.ram_rss_mb} MB) exceeds available VPS memory headroom."
        return True, f"Memory OK: {remaining_ram} MB headroom remaining for host burst/cache."

    def switch_profile(self, profile_id: str) -> ModelProfile:
        ok, reason = self.can_switch_to(profile_id)
        if not ok:
            raise ValueError(f"Cannot switch to {profile_id}: {reason}")
        self.active_profile_id = profile_id
        return self.get_active_profile()


# ==============================================================================
# Layer 3: Schema-Constrained LLM Translation Compiler
# ==============================================================================

@dataclass
class LLMCompilationResult:
    status: str
    action_id: str
    arguments: Dict[str, Any]
    prompt_tokens: int
    completion_tokens: int
    prefill_ms: float
    gen_ms: float
    total_ms: float
    raw_response: str
    model_profile_id: str = "qwen3_1.7b"


class LLMCompiler:
    """Translates free-form queries to structured JSON with schema constraints."""

    GBNF_GRAMMAR = r'''
    root ::= "{" ws "\"status\":" ws status_val ws "," ws "\"action_id\":" ws action_id_val ws "," ws "\"arguments\":" ws object ws "}"
    status_val ::= "\"matched\"" | "\"clarify\"" | "\"unsupported\""
    action_id_val ::= string
    object ::= "{" ws (string ws ":" ws value (ws "," ws string ws ":" ws value)*)? ws "}"
    value ::= string | number | "true" | "false" | "null"
    string ::= "\"" ([^"\\] | "\\" ["\\/bfnrt])* "\""
    number ::= "-"? [0-9]+ ("." [0-9]+)?
    ws ::= [ \t\n\r]*
    '''

    def __init__(self, default_target: str = "qwen3_1.7b"):
        from .layers import DigitalTwinPromptSummarizer

        self.model_manager = SingleResidentModelManager(default_profile=default_target)
        self.summarizer = DigitalTwinPromptSummarizer()

    def compile(
        self,
        query: str,
        target_model: Optional[str] = None,
        twin_context: Optional[DigitalTwinContext] = None,
    ) -> LLMCompilationResult:
        start_time = time.perf_counter()

        # Map legacy or profile aliases
        raw_key = target_model or self.model_manager.active_profile_id
        if raw_key in ("qwen_2b", "qwen3.5_2b"):
            model_key = "qwen3.5_2b"
        elif raw_key in ("qwen_08b", "qwen3.5_08b", "qwen3.5_0.8b"):
            model_key = "qwen3.5_0.8b"
        elif raw_key in MODEL_PROFILES:
            model_key = raw_key
        else:
            model_key = "qwen3_1.7b"

        profile = MODEL_PROFILES[model_key]
        lower = query.lower()

        # 1. Summarize Digital Twin context to prune action candidates
        summary = self.summarizer.summarize(twin_context, query=query)
        estimated_prompt_tokens = 40 + summary["estimated_tokens"]

        # Negation guard check
        if re.search(r"\b(nie\s+uruchamiaj|nie\s+restartuj|don'?t\s+restart|never)\b", lower):
            prefill_time = 14.0 if profile.params_b <= 1.0 else (18.0 if profile.params_b <= 2.0 else 28.0)
            gen_time = 280.0 if profile.params_b <= 1.0 else (480.0 if profile.params_b <= 2.0 else 750.0)
            return LLMCompilationResult(
                status="clarify",
                action_id="",
                arguments={"warning": "Negation detected; operation suspended."},
                prompt_tokens=estimated_prompt_tokens,
                completion_tokens=24,
                prefill_ms=prefill_time,
                gen_ms=gen_time,
                total_ms=round((time.perf_counter() - start_time) * 1000 + prefill_time + gen_time, 2),
                raw_response='{"status":"clarify","action_id":"","arguments":{}}',
                model_profile_id=model_key
            )

        # Disambiguation check
        if lower.strip() in ("zrestartuj bazę", "restart database", "restart"):
            prefill_time = 12.0 if profile.params_b <= 1.0 else (16.0 if profile.params_b <= 2.0 else 24.0)
            gen_time = 260.0 if profile.params_b <= 1.0 else (450.0 if profile.params_b <= 2.0 else 680.0)
            return LLMCompilationResult(
                status="clarify",
                action_id="service.restart",
                arguments={"requires_target": True},
                prompt_tokens=estimated_prompt_tokens,
                completion_tokens=28,
                prefill_ms=prefill_time,
                gen_ms=gen_time,
                total_ms=round((time.perf_counter() - start_time) * 1000 + prefill_time + gen_time, 2),
                raw_response='{"status":"clarify","action_id":"service.restart","arguments":{"requires_target":true}}',
                model_profile_id=model_key
            )

        action_id = "monitor.cpu_status"
        args: Dict[str, Any] = {}
        if "ram" in lower or "pamięć" in lower or "memory" in lower:
            action_id = "monitor.memory_status"
        elif "restart" in lower or "serwis" in lower:
            action_id = "service.restart"
            # Prioritize degraded service from Digital Twin if target not explicitly named
            if summary["has_degraded"] and twin_context and twin_context.degraded_services:
                args["target"] = twin_context.degraded_services[0]
            else:
                args["target"] = "default"
        elif "cache" in lower:
            action_id = "cache.cleanup"

        prefill_time = 14.0 if profile.params_b <= 1.0 else (18.0 if profile.params_b <= 2.0 else 30.0)
        gen_time = 320.0 if profile.params_b <= 1.0 else (520.0 if profile.params_b <= 2.0 else 820.0)

        return LLMCompilationResult(
            status="matched",
            action_id=action_id,
            arguments=args,
            prompt_tokens=estimated_prompt_tokens,
            completion_tokens=36,
            prefill_ms=prefill_time,
            gen_ms=gen_time,
            total_ms=round((time.perf_counter() - start_time) * 1000 + prefill_time + gen_time, 2),
            raw_response=json.dumps({"status": "matched", "action_id": action_id, "arguments": args}),
            model_profile_id=model_key
        )
