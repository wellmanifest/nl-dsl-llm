"""Digital Twin context/summarizer, host telemetry, and single-resident model profiles."""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DigitalTwinContext:
    environment: str = "prod"
    active_services: list[str] = field(
        default_factory=lambda: ["postgres", "redis-cache"]
    )
    degraded_services: list[str] = field(default_factory=lambda: ["redis-cache"])
    active_tickets: list[str] = field(
        default_factory=lambda: ["ticket-001", "ticket-002"]
    )
    cluster_nodes: list[str] = field(default_factory=list)


class DigitalTwinPromptSummarizer:
    """
    Prunes the vast environment and action catalogue into a compact top 5-10 candidate
    context before passing to LLM prefill. Ensures CPU prefill stays < 30ms and prompt
    tokens stay < 100 on 8 vCPU VPS.
    Prioritizes degraded entities so recovery actions appear at top of prompt context.
    """

    _BASELINES = (
        ("monitor.cpu_status", "proc://taskand.dev/monitor/cpu/v1"),
        ("monitor.memory_status", "proc://taskand.dev/monitor/memory/v1"),
        ("cache.cleanup", "proc://taskand.dev/cache/cleanup/v1"),
        ("task.list", "proc://taskand.dev/task/list/v1"),
    )

    def __init__(self, max_candidates: int = 8):
        self.max_candidates = max_candidates

    def summarize(
        self, twin: DigitalTwinContext | None = None, query: str = ""
    ) -> dict[str, Any]:
        twin_ctx = twin or DigitalTwinContext()
        candidates: list[dict[str, Any]] = [
            {
                "action_id": "service.restart",
                "target": svc,
                "environment": twin_ctx.environment,
                "priority": "HIGH_DEGRADED",
                "uri": "proc://taskand.dev/service/restart/v1",
            }
            for svc in twin_ctx.degraded_services
        ]
        for svc in twin_ctx.active_services:
            if svc not in twin_ctx.degraded_services:
                candidates.append(
                    {
                        "action_id": "service.restart",
                        "target": svc,
                        "environment": twin_ctx.environment,
                        "priority": "NORMAL",
                        "uri": "proc://taskand.dev/service/restart/v1",
                    }
                )
        for act_id, uri in self._BASELINES:
            candidates.append({"action_id": act_id, "priority": "BASELINE", "uri": uri})

        pruned = candidates[: self.max_candidates]
        degraded_str = ", ".join(twin_ctx.degraded_services) or "None"
        actions_str = "; ".join(
            f"{c['action_id']}(target={c.get('target', 'none')})" for c in pruned
        )
        prompt_str = (
            f"Env: {twin_ctx.environment} | Degraded: {degraded_str} "
            f"| Available Actions: {actions_str}"
        )
        return {
            "query": query,
            "prompt_snippet": prompt_str,
            "pruned_candidates": pruned,
            "candidate_count": len(pruned),
            "estimated_tokens": len(prompt_str.split()) * 2,
            "has_degraded": len(twin_ctx.degraded_services) > 0,
        }


class SystemTelemetryObserver:
    """
    Live host telemetry observer that feeds DigitalTwinContext.
    Reads /proc/loadavg, /proc/meminfo, and checks service health.
    """

    def __init__(
        self, loadavg_threshold: float = 7.0, min_available_ram_mb: int = 1500
    ):
        self.loadavg_threshold = loadavg_threshold
        self.min_available_ram_mb = min_available_ram_mb

    def read_metrics(self) -> dict[str, Any]:
        metrics: dict[str, Any] = {
            "load_1m": 0.5,
            "load_5m": 0.5,
            "mem_total_mb": 8192,
            "mem_available_mb": 4096,
            "is_cpu_saturated": False,
            "is_memory_strained": False,
        }
        if os.path.exists("/proc/loadavg"):
            with (
                contextlib.suppress(OSError, ValueError, IndexError),
                open("/proc/loadavg", encoding="utf-8") as f,
            ):
                parts = f.read().split()
                metrics["load_1m"] = float(parts[0])
                metrics["load_5m"] = float(parts[1])
                metrics["is_cpu_saturated"] = (
                    metrics["load_1m"] > self.loadavg_threshold
                )

        if os.path.exists("/proc/meminfo"):
            with (
                contextlib.suppress(OSError, ValueError, IndexError),
                open("/proc/meminfo", encoding="utf-8") as f,
            ):
                for line in f:
                    if line.startswith("MemTotal:"):
                        metrics["mem_total_mb"] = int(line.split()[1]) // 1024
                    elif line.startswith("MemAvailable:"):
                        metrics["mem_available_mb"] = int(line.split()[1]) // 1024
                metrics["is_memory_strained"] = (
                    metrics["mem_available_mb"] < self.min_available_ram_mb
                )
        return metrics

    def observe_digital_twin(
        self,
        environment: str = "prod",
        known_services: list[str] | None = None,
        manual_degraded: list[str] | None = None,
    ) -> DigitalTwinContext:
        metrics = self.read_metrics()
        services = list(known_services or ["postgres", "redis-cache", "taskand-worker"])
        degraded = list(manual_degraded or [])
        if (
            metrics["is_memory_strained"]
            and "redis-cache" in services
            and "redis-cache" not in degraded
        ):
            degraded.append("redis-cache")
        return DigitalTwinContext(
            environment=environment,
            active_services=services,
            degraded_services=degraded,
            active_tickets=["ticket-001"],
            cluster_nodes=["vps-01-primary"],
        )


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
    polish_support: str
    priority: int
    recommended_settings: dict[str, Any]
    port: int = 8081


MODEL_PROFILES: dict[str, ModelProfile] = {
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
    ),
}


class SingleResidentModelManager:
    """
    Enforces the single-resident model execution invariant on 8 vCPU / 8 GB RAM VPS.
    Never runs two heavy models simultaneously. Leaves 5-6 GB RAM free for OS,
    Taskand, PostgreSQL, Redis, and Playwright/noVNC.
    """

    TOTAL_HOST_RAM_MB: int = 8192
    RESERVED_HOST_SERVICES_RAM_MB: int = 5600

    def __init__(self, default_profile: str = "qwen3_1.7b"):
        self.active_profile_id: str = default_profile

    def get_active_profile(self) -> ModelProfile:
        return MODEL_PROFILES.get(self.active_profile_id, MODEL_PROFILES["qwen3_1.7b"])

    def can_switch_to(self, profile_id: str) -> tuple[bool, str]:
        if profile_id not in MODEL_PROFILES:
            return False, f"Unknown profile: {profile_id}"
        target = MODEL_PROFILES[profile_id]
        remaining_ram = (
            self.TOTAL_HOST_RAM_MB
            - self.RESERVED_HOST_SERVICES_RAM_MB
            - target.ram_rss_mb
        )
        if remaining_ram < 0:
            return (
                False,
                f"Model {target.name} ({target.ram_rss_mb} MB) exceeds available VPS memory headroom.",
            )
        return (
            True,
            f"Memory OK: {remaining_ram} MB headroom remaining for host burst/cache.",
        )

    def switch_profile(self, profile_id: str) -> ModelProfile:
        ok, reason = self.can_switch_to(profile_id)
        if not ok:
            raise ValueError(f"Cannot switch to {profile_id}: {reason}")
        self.active_profile_id = profile_id
        return self.get_active_profile()
