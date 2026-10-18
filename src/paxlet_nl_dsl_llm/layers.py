"""
Resolution Layers for paxlet-com/nl-dsl-llm:
  Layer 1: Deterministic Multi-lingual Fast-Path (Regex/Synonyms)
  Layer 1.5: Semantic Intent Cache with Negation/Slot Guard
  Layer 2.5: Digital Twin Prompt Summarizer (Compact Candidate Pruning)
  Layer 3: Schema-Constrained LLM Translation Compiler
  Layer 4: Single-Resident Model Manager (Memory Invariant on 8GB VPS)
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .layers_compiler import (
    MODEL_PROFILES,
    LLMCompilationResult,
    LLMCompiler,
    ModelProfile,
    SingleResidentModelManager,
    SystemTelemetryObserver,
    VoiceTokenStreamHandler,
)


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
            candidates.append({"action_id": "service.restart", "target": svc, "environment": twin_ctx.environment, "priority": "HIGH_DEGRADED", "uri": "proc://taskand.dev/service/restart/v1"})

        # 2. Medium priority: Active services
        for svc in twin_ctx.active_services:
            if svc not in twin_ctx.degraded_services:
                candidates.append({"action_id": "service.restart", "target": svc, "environment": twin_ctx.environment, "priority": "NORMAL", "uri": "proc://taskand.dev/service/restart/v1"})

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
            "query": query,
            "prompt_snippet": prompt_str,
            "pruned_candidates": pruned,
            "candidate_count": len(pruned),
            "estimated_tokens": len(prompt_str.split()) * 2,
            "has_degraded": len(twin_ctx.degraded_services) > 0
        }


__all__ = [
    "POLISH_PHONETIC_REPLACEMENTS", "strip_accents", "phonetic_key", "damerau_levenshtein",
    "FastPathParser", "CachedIntentEntry", "SemanticCache", "DigitalTwinContext",
    "DigitalTwinPromptSummarizer", "SystemTelemetryObserver", "VoiceTokenStreamHandler",
    "ModelProfile", "MODEL_PROFILES", "SingleResidentModelManager", "LLMCompilationResult", "LLMCompiler",
]
