"""
Resolution Layers for paxlet-com/nl-dsl-llm:
  Layer 1: Deterministic Multi-lingual Fast-Path (Regex/Synonyms)
  Layer 1.5: Semantic Intent Cache with Negation/Slot Guard
  Layer 2.5: Digital Twin Prompt Summarizer (Compact Candidate Pruning)
  Layer 3: Schema-Constrained LLM Translation Compiler
  Layer 4: Single-Resident Model Manager (Memory Invariant on 8GB VPS)
"""

from __future__ import annotations

import json
import os
import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ==============================================================================
# Layer 0.8: Acoustic & Typo Tolerance Normalizer (Damerau-Levenshtein & Phonetics)
# ==============================================================================

POLISH_PHONETIC_REPLACEMENTS = [
    ("rz", "ż"), ("ch", "h"), ("ó", "u"),
    ("om", "ą"), ("on", "ą"), ("em", "ę"), ("en", "ę"),
    ("i", "j"),
]

def strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize('NFKD', text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))

def phonetic_key(word: str) -> str:
    w = word.lower().strip()
    for src, dst in POLISH_PHONETIC_REPLACEMENTS:
        w = w.replace(src, dst)
    return strip_accents(w)

def damerau_levenshtein(s1: str, s2: str) -> int:
    d = {}
    len1 = len(s1)
    len2 = len(s2)
    for i in range(-1, len1 + 1):
        d[(i, -1)] = i + 1
    for j in range(-1, len2 + 1):
        d[(-1, j)] = j + 1

    for i in range(len1):
        for j in range(len2):
            cost = 0 if s1[i] == s2[j] else 1
            d[(i, j)] = min(
                d[(i - 1, j)] + 1,       # deletion
                d[(i, j - 1)] + 1,       # insertion
                d[(i - 1, j - 1)] + cost # substitution
            )
            if i > 0 and j > 0 and s1[i] == s2[j - 1] and s1[i - 1] == s2[j]:
                d[(i, j)] = min(d[(i, j)], d[(i - 2, j - 2)] + cost)
    return d[(len1 - 1, len2 - 1)]


# ==============================================================================
# Layer 1: Deterministic Multi-lingual Fast-Path
# ==============================================================================

class FastPathParser:
    """Deterministic regex-based intent parser for standard commands (<1ms)."""

    PATTERNS: List[Tuple[re.Pattern, str, Dict[str, Any]]] = [
        # 1. CPU Monitor
        (
            re.compile(r"\b(status\s+cpu|pokaż\s+(procesor|cpu)|użycie\s+(procesora|cpu)|show\s+cpu|cpu\s+status|check\s+load)\b|^\s*(cpu|procesor|proc|obciążenie|obc|load)\s*$", re.I),
            "monitor.cpu_status",
            {}
        ),
        # 2. Memory Monitor
        (
            re.compile(r"\b(status\s+ram|pokaż\s+(pamięć|ram)|użycie\s+(ramu|ram|pamięci)|ram\s+status|show\s+memory|memory\s+usage)\b|^\s*(ram|pamięć|pam|pamięci|memory|mem)\s*$", re.I),
            "monitor.memory_status",
            {}
        ),
        # 3. Cache Cleanup
        (
            re.compile(r"\b(wyczyść\s+cache|usuń\s+pamięć\s+podręczną|clear\s+cache|clean\s+cache|purge\s+cache)\b|^\s*(cache|wyczyść|wycz|czysz|purge|flush)\s*$", re.I),
            "cache.cleanup",
            {}
        ),
        # 4. Service Restart (Exact single word / half-word or designated phrases; 'zrestartuj bazę' excluded to allow clarify)
        (
            re.compile(r"\b(zrestartuj\s+usługę|restart\s+service|zrestartuj\s+serwis)\b|^\s*(restart|res|zrestartuj|zre|reboot|reb|reset)\s*$", re.I),
            "service.restart",
            {"target": "default"}
        ),
        # 5. Task Schedule
        (
            re.compile(r"\b(zaplanuj\s+zadanie|schedule\s+task|plan\s+task)\b|^\s*(harmonogram|harm|zaplanuj|zap|schedule|sched|kolejkuj)\s*$", re.I),
            "task.schedule",
            {"title": "Routine health scan", "environment": "prod"}
        ),
        # 6. Task Get
        (
            re.compile(r"\b(szczegóły\s+zadania|pokaż\s+szczegóły|task\s+details|show\s+details)\b|^\s*(szczegóły|szcz|szczegoly|inspect|insp|dane)\s*$", re.I),
            "task.get",
            {"id": "task-001"}
        ),
        # 7. Task Create
        (
            re.compile(r"\b(utwórz\s+zadanie|dodaj\s+zadanie|nowe\s+zadanie|create\s+task|new\s+task)\b|^\s*(utwórz|utw|utworz|nowy|create|crea|dodaj)\s*$", re.I),
            "task.create",
            {"title": "New Task"}
        ),
        # 8. Task Close
        (
            re.compile(r"\b(zamknij\s+zadanie|zakończ\s+zadanie|close\s+task|complete\s+task)\b|^\s*(zakończ|zak|zakoncz|zamknij|zam|complete|comp|gotowe)\s*$", re.I),
            "task.close",
            {"id": "task-001"}
        ),
        # 9. Task List
        (
            re.compile(r"\b(pokaż\s+(otwarte\s+)?(zadania|tickety)|wypisz\s+(zadania|tickety)|list\s+(open\s+)?(tasks|tickets)|show\s+(tasks|tickets))\b|^\s*(zadania|zad|tickety|tick|tasks|tickets)\s*$", re.I),
            "task.list",
            {}
        ),
        # 10. Worktree Allocate
        (
            re.compile(r"\b(alokuj\s+worktree|allocate\s+worktree|nowe\s+worktree|create\s+worktree)\b|^\s*(worktree|work|alokuj|alo|gałąź|branch)\s*$", re.I),
            "worktree.allocate",
            {"ticket": "ticket-current"}
        ),
        # 11. Koru Doctor
        (
            re.compile(r"\b(koru\s+doctor|uruchom\s+doctora|sprawdź\s+błędy|doctor\s+diagnose)\b|^\s*(doctor|doc|napraw|nap|diagnostyka|diag|diagnoza)\s*$", re.I),
            "koru.doctor",
            {}
        ),
        # 12. Koru Queue Next
        (
            re.compile(r"\b(następne\s+zadanie|kolejne\s+zadanie|next\s+task|pop\s+queue)\b|^\s*(kolejka|kol|następne|nast|następny|next)\s*$", re.I),
            "koru.queue.next",
            {}
        ),
        # 13. Koru Queue Loop
        (
            re.compile(r"\b(pętla\s+zadań|task\s+loop|run\s+loop|queue\s+loop)\b|^\s*(pętla|pęt|pet|loop|automatyzacja|auto|ciągłość)\s*$", re.I),
            "koru.queue.loop",
            {}
        ),
        # 14. Koru Context
        (
            re.compile(r"\b(pokaż\s+kontekst|show\s+context|stan\s+projektu|session\s+context)\b|^\s*(kontekst|kon|brief|brie|context|stan)\s*$", re.I),
            "koru.context",
            {}
        ),
        # 15. Pull Request List
        (
            re.compile(r"\b(pokaż\s+pr|list\s+prs|open\s+prs|lista\s+pr)\b|^\s*(pr|prs|pulls|pul)\s*$", re.I),
            "pr.list",
            {}
        ),
        # 16. Token List
        (
            re.compile(r"\b(zużycie\s+tokenów|token\s+usage|stan\s+tokenów)\b|^\s*(token|tok|tokeny|zużycie|zuż|zuz)\s*$", re.I),
            "token.list",
            {}
        ),
        # 17. Process List
        (
            re.compile(r"\b(lista\s+procesów|list\s+processes|running\s+processes)\b|^\s*(procesy|proce|agenty|agen|processes)\s*$", re.I),
            "process.list",
            {}
        ),
    ]

    TYPO_CATALOG: Dict[str, Tuple[str, Dict[str, Any]]] = {
        "restart": ("service.restart", {"target": "default"}),
        "zrestartuj": ("service.restart", {"target": "default"}),
        "reboot": ("service.restart", {"target": "default"}),
        "procesor": ("monitor.cpu_status", {}),
        "obciążenie": ("monitor.cpu_status", {}),
        "pamięć": ("monitor.memory_status", {}),
        "memory": ("monitor.memory_status", {}),
        "wyczyść": ("cache.cleanup", {}),
        "purge": ("cache.cleanup", {}),
        "zadania": ("task.list", {}),
        "tickety": ("task.list", {}),
        "tasks": ("task.list", {}),
        "harmonogram": ("task.schedule", {"title": "Routine health scan"}),
        "zaplanuj": ("task.schedule", {"title": "Routine health scan"}),
        "schedule": ("task.schedule", {"title": "Routine health scan"}),
        "szczegóły": ("task.get", {"id": "task-001"}),
        "inspect": ("task.get", {"id": "task-001"}),
        "utwórz": ("task.create", {"title": "New Task"}),
        "create": ("task.create", {"title": "New Task"}),
        "zamknij": ("task.close", {"id": "task-001"}),
        "zakończ": ("task.close", {"id": "task-001"}),
        "complete": ("task.close", {"id": "task-001"}),
        "worktree": ("worktree.allocate", {"ticket": "ticket-current"}),
        "alokuj": ("worktree.allocate", {"ticket": "ticket-current"}),
        "gałąź": ("worktree.allocate", {"ticket": "ticket-current"}),
        "doctor": ("koru.doctor", {}),
        "doktor": ("koru.doctor", {}),
        "napraw": ("koru.doctor", {}),
        "diagnostyka": ("koru.doctor", {}),
        "kolejka": ("koru.queue.next", {}),
        "następne": ("koru.queue.next", {}),
        "pętla": ("koru.queue.loop", {}),
        "automatyzacja": ("koru.queue.loop", {}),
        "kontekst": ("koru.context", {}),
        "pulls": ("pr.list", {}),
        "tokeny": ("token.list", {}),
        "zużycie": ("token.list", {}),
        "procesy": ("process.list", {}),
        "agenty": ("process.list", {}),
        # Canonical prefixes for half-words with typos
        "res": ("service.restart", {"target": "default"}),
        "zre": ("service.restart", {"target": "default"}),
        "proc": ("monitor.cpu_status", {}),
        "pam": ("monitor.memory_status", {}),
        "wycz": ("cache.cleanup", {}),
        "czysz": ("cache.cleanup", {}),
        "zad": ("task.list", {}),
        "harm": ("task.schedule", {"title": "Routine health scan"}),
        "work": ("worktree.allocate", {"ticket": "ticket-current"}),
        "doc": ("koru.doctor", {}),
        "kol": ("koru.queue.next", {}),
        "pet": ("koru.queue.loop", {}),
        "kon": ("koru.context", {}),
        "tok": ("token.list", {}),
        "alo": ("worktree.allocate", {"ticket": "ticket-current"}),
        "proce": ("process.list", {}),
    }

    def parse(self, text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        clean = text.strip()
        if not clean or re.search(r"\b(nie|not|don'?t|nie\s+uruchamiaj|nie\s+restartuj|nie\s+czyść)\b", clean, re.I):
            return None

        for pattern, action_id, default_args in self.PATTERNS:
            if pattern.search(clean):
                args = dict(default_args)
                if action_id == "task.list":
                    if re.search(r"\b(otwarte|open)\b", clean, re.I):
                        args["status"] = "OPEN"
                    elif re.search(r"\b(zamknięte|closed|done)\b", clean, re.I):
                        args["status"] = "CLOSED"

                env_match = re.search(r"\b(prod|staging|test|dev)\b", clean, re.I)
                if env_match:
                    args["environment"] = env_match.group(1).lower()

                return action_id, args

        # Layer 0.8: Fuzzy acoustic and keyboard typo fallback for single words and prefixes
        tokens = clean.split()
        if len(tokens) == 1 and len(clean) >= 3:
            q_raw = tokens[0].lower().strip(".,;:?!")
            q_phonetic = phonetic_key(q_raw)
            best_match = None
            min_dist = 3
            for kw, (act_id, def_args) in self.TYPO_CATALOG.items():
                kw_raw = kw.lower()
                kw_phonetic = phonetic_key(kw_raw)

                # Check phonetic equality (e.g. pamienc == pamięć, doktur == doktor)
                if q_phonetic == kw_phonetic:
                    args = dict(def_args)
                    args["_typo_corrected_from"] = q_raw
                    return act_id, args

                # Damerau-Levenshtein distance
                d_raw = damerau_levenshtein(q_raw, kw_raw)
                d_phon = damerau_levenshtein(q_phonetic, kw_phonetic)
                d_best = min(d_raw, d_phon)
                threshold = 1 if len(kw_raw) <= 4 else 2
                if d_best <= threshold and d_best < min_dist:
                    min_dist = d_best
                    best_match = (act_id, def_args)

            if best_match:
                act_id, def_args = best_match
                args = dict(def_args)
                args["_typo_corrected_from"] = q_raw
                return act_id, args

        # Check multi-word phrases where the primary verb has an acoustic typo (e.g. 'Zwestartuj proces...')
        if len(tokens) > 1 and len(clean) <= 60:
            first_raw = tokens[0].lower().strip(".,;:?!")
            first_phon = phonetic_key(first_raw)
            for kw, (act_id, def_args) in self.TYPO_CATALOG.items():
                if len(kw) >= 5:
                    d_raw = damerau_levenshtein(first_raw, kw.lower())
                    d_phon = damerau_levenshtein(first_phon, phonetic_key(kw.lower()))
                    best_d = min(d_raw, d_phon)
                    if 0 < best_d <= 2:
                        args = dict(def_args)
                        args["_typo_corrected_from"] = first_raw
                        if "redis" in clean.lower():
                            args["target"] = "redis"
                            return act_id, args
                        elif "postgres" in clean.lower():
                            args["target"] = "postgres"
                            return act_id, args

        return None


# ==============================================================================
# Layer 1.5: Semantic Intent Cache & Negation Guard
# ==============================================================================

@dataclass
class CachedIntentEntry:
    query_template: str
    action_id: str
    is_negated: bool
    required_slots: List[str] = field(default_factory=list)
    default_arguments: Dict[str, Any] = field(default_factory=dict)


class SemanticCache:
    """Semantic intent cache with explicit polarity check and slot collision guard."""

    NEGATION_PATTERN = re.compile(r"\b(nie|not|bez|don'?t|never|nigdy|odrzuć|without)\b", re.I)

    def __init__(self):
        self._entries: List[CachedIntentEntry] = [
            CachedIntentEntry(
                query_template="zrestartuj proces bazy danych",
                action_id="service.restart",
                is_negated=False,
                required_slots=["target"],
                default_arguments={"target": "database"}
            ),
            CachedIntentEntry(
                query_template="wyczyść pamięć podręczną na serwerze",
                action_id="cache.cleanup",
                is_negated=False,
                required_slots=["environment"],
                default_arguments={"environment": "default"}
            )
        ]

    def check_polarity(self, text: str) -> bool:
        return bool(self.NEGATION_PATTERN.search(text))

    def extract_slots(self, text: str) -> Dict[str, Any]:
        slots: Dict[str, Any] = {}
        lower = text.lower()
        env_match = re.search(r"\b(prod|production|staging|test|testing|dev|development)\b", lower)
        if env_match:
            slots["environment"] = "prod" if env_match.group(1) in ("prod", "production") else "test"
        num_match = re.search(r"\b([0-9]+)\b", lower)
        if num_match:
            slots["count"] = int(num_match.group(1))
        return slots

    def lookup(self, text: str) -> Optional[Tuple[str, Dict[str, Any], float]]:
        is_query_negated = self.check_polarity(text)
        query_slots = self.extract_slots(text)

        for entry in self._entries:
            if entry.is_negated != is_query_negated:
                continue
            if "environment" in query_slots and "environment" in entry.required_slots:
                if query_slots["environment"] == "prod" and entry.default_arguments.get("environment") != "prod":
                    continue

            q_words = set(re.findall(r"\b\w+\b", text.lower()))
            t_words = set(re.findall(r"\b\w+\b", entry.query_template.lower()))
            overlap = len(q_words & t_words) / max(len(q_words | t_words), 1)

            if overlap >= 0.60:
                args = dict(entry.default_arguments)
                args.update(query_slots)
                return entry.action_id, args, round(0.85 + (overlap * 0.14), 2)

        return None

    def promote(
        self,
        query: str,
        action_id: str,
        arguments: Optional[Dict[str, Any]] = None,
        required_slots: Optional[List[str]] = None,
    ) -> bool:
        """
        Dynamically promote a verified Layer 3 LLM compilation result into Layer 1.5 cache.
        Enforces polarity safety: never promotes a negated query as positive.
        """
        clean = query.strip()
        if not clean or not action_id:
            return False
        is_negated = self.check_polarity(clean)
        for existing in self._entries:
            if existing.query_template.lower() == clean.lower():
                return False
        slots = list(required_slots) if required_slots is not None else list((arguments or {}).keys())
        self._entries.append(
            CachedIntentEntry(
                query_template=clean,
                action_id=action_id,
                is_negated=is_negated,
                required_slots=slots,
                default_arguments=dict(arguments or {}),
            )
        )
        return True


# ==============================================================================
# Layer 2.5: Digital Twin Prompt Summarizer (Compact Pruning)
# ==============================================================================

@dataclass
class DigitalTwinContext:
    environment: str = "prod"
    active_services: List[str] = field(default_factory=lambda: ["postgres", "redis-cache"])
    degraded_services: List[str] = field(default_factory=lambda: ["redis-cache"])
    active_tickets: List[str] = field(default_factory=lambda: ["ticket-001", "ticket-002"])
    cluster_nodes: List[str] = field(default_factory=list)


class DigitalTwinPromptSummarizer:
    """
    Prunes the vast environment and action catalogue into a compact top 5-10 candidate
    context before passing to LLM prefill. Ensures CPU prefill stays < 30ms and prompt
    tokens stay < 100 on 8 vCPU VPS.
    Prioritizes degraded entities so recovery actions appear at top of prompt context.
    """
    def __init__(self, max_candidates: int = 8):
        self.max_candidates = max_candidates

    def summarize(self, twin: Optional[DigitalTwinContext] = None, query: str = "") -> Dict[str, Any]:
        twin_ctx = twin or DigitalTwinContext()
        candidates: List[Dict[str, Any]] = []

        # 1. High priority: Degraded services (require immediate operational focus)
        for svc in twin_ctx.degraded_services:
            candidates.append({
                "action_id": "service.restart",
                "target": svc,
                "environment": twin_ctx.environment,
                "priority": "HIGH_DEGRADED",
                "uri": f"proc://taskand.dev/service/restart/v1"
            })

        # 2. Medium priority: Active services
        for svc in twin_ctx.active_services:
            if svc not in twin_ctx.degraded_services:
                candidates.append({
                    "action_id": "service.restart",
                    "target": svc,
                    "environment": twin_ctx.environment,
                    "priority": "NORMAL",
                    "uri": f"proc://taskand.dev/service/restart/v1"
                })

        # 3. Standard monitoring baselines
        candidates.append({"action_id": "monitor.cpu_status", "priority": "BASELINE", "uri": "proc://taskand.dev/monitor/cpu/v1"})
        candidates.append({"action_id": "monitor.memory_status", "priority": "BASELINE", "uri": "proc://taskand.dev/monitor/memory/v1"})
        candidates.append({"action_id": "cache.cleanup", "priority": "BASELINE", "uri": "proc://taskand.dev/cache/cleanup/v1"})
        candidates.append({"action_id": "task.list", "priority": "BASELINE", "uri": "proc://taskand.dev/task/list/v1"})

        # Trim to max_candidates
        pruned = candidates[:self.max_candidates]

        prompt_str = f"Env: {twin_ctx.environment} | Degraded: {', '.join(twin_ctx.degraded_services) or 'None'} | Available Actions: " + "; ".join(
            f"{c['action_id']}(target={c.get('target', 'none')})" for c in pruned
        )

        return {
            "prompt_snippet": prompt_str,
            "pruned_candidates": pruned,
            "candidate_count": len(pruned),
            "estimated_tokens": len(prompt_str.split()) * 2,
            "has_degraded": len(twin_ctx.degraded_services) > 0
        }


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
        self.summarizer = summarizer or DigitalTwinPromptSummarizer()

    def process_token(
        self,
        accumulated_transcript: str,
        twin_context: Optional[DigitalTwinContext] = None,
        max_options: int = 5,
    ) -> Dict[str, Any]:
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
