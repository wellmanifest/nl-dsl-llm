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
import stat
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from .layers_compiler import (
    MAX_QUERY_CHARS,
    MODEL_PROFILES,
    NEGATION_PATTERN,
    DigitalTwinContext,
    DigitalTwinPromptSummarizer,
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
    ("rz", "ż"),
    ("ch", "h"),
    ("ó", "u"),
    ("om", "ą"),
    ("on", "ą"),
    ("em", "ę"),
    ("en", "ę"),
    ("i", "j"),
]


def strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def phonetic_key(word: str) -> str:
    w = word.lower().strip()
    for src, dst in POLISH_PHONETIC_REPLACEMENTS:
        w = w.replace(src, dst)
    return strip_accents(w)


def damerau_levenshtein(s1: str, s2: str, max_dist: int | None = None) -> int:
    if s1 == s2:
        return 0
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    len1 = len(s1)
    len2 = len(s2)
    if max_dist is not None and (len1 - len2) > max_dist:
        return max_dist + 1
    if len2 == 0:
        return len1

    prev2: list[int] = []
    prev1: list[int] = list(range(len2 + 1))
    for i in range(len1):
        curr = [i + 1] + [0] * len2
        c1 = s1[i]
        for j in range(len2):
            c2 = s2[j]
            cost = 0 if c1 == c2 else 1
            val = min(prev1[j + 1] + 1, curr[j] + 1, prev1[j] + cost)
            if i > 0 and j > 0 and c1 == s2[j - 1] and s1[i - 1] == c2:
                val = min(val, prev2[j - 1] + cost)
            curr[j + 1] = val
        if max_dist is not None and min(curr) > max_dist:
            return max_dist + 1
        prev2, prev1 = prev1, curr
    return prev1[len2]


# ==============================================================================
# Layer 1: Deterministic Multi-lingual Fast-Path
# ==============================================================================


def _load_fast_path_patterns() -> list[tuple[re.Pattern[str], str, dict[str, Any]]]:
    profile_path = (
        Path(__file__).resolve().parents[2] / "standard" / "semantic-profile.json"
    )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    fd = os.open(os.fspath(profile_path), flags)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_size > 1_048_576:
            raise OSError("semantic-profile.json must be a bounded regular file")
        with os.fdopen(fd, "rb") as handle:
            fd = -1
            raw = handle.read(1_048_577)
    finally:
        if fd >= 0:
            os.close(fd)
    data = json.loads(raw.decode("utf-8"))
    entries = data.get("intent_lexicon", {}).get("fast_path_patterns", [])
    return [
        (
            re.compile(item["pattern"], re.IGNORECASE),
            item["intent"],
            dict(item.get("default_args", {})),
        )
        for item in entries
    ]


class FastPathParser:
    """Deterministic regex-based intent parser for standard commands (<1ms)."""

    PATTERNS: ClassVar[list[tuple[re.Pattern[str], str, dict[str, Any]]]] = (
        _load_fast_path_patterns()
    )

    TYPO_CATALOG: ClassVar[dict[str, tuple[str, dict[str, Any]]]] = {
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

    def parse(self, text: str) -> tuple[str, dict[str, Any]] | None:
        if not isinstance(text, str) or len(text) > MAX_QUERY_CHARS:
            return None
        clean = text.strip()
        if not clean or NEGATION_PATTERN.search(clean):
            return None

        for pattern, action_id, default_args in self.PATTERNS:
            if pattern.search(clean):
                args = dict(default_args)
                if action_id == "task.list":
                    if re.search(r"\b(otwarte|open)\b", clean, re.IGNORECASE):
                        args["status"] = "OPEN"
                    elif re.search(
                        r"\b(zamknięte|closed|done)\b", clean, re.IGNORECASE
                    ):
                        args["status"] = "CLOSED"

                env_match = re.search(
                    r"\b(prod|staging|test|dev)\b", clean, re.IGNORECASE
                )
                if env_match:
                    args["environment"] = env_match.group(1).lower()

                return action_id, args

        # Layer 0.8: Fuzzy acoustic and keyboard typo fallback for single words and prefixes
        tokens = clean.split()
        if len(tokens) == 1 and 3 <= len(clean) <= 32:
            q_raw = tokens[0].lower().strip(".,;:?!")
            if len(q_raw) < 3:
                return None
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
                threshold = 1 if len(kw_raw) <= 4 else 2
                d_raw = damerau_levenshtein(q_raw, kw_raw, max_dist=threshold)
                d_phon = damerau_levenshtein(
                    q_phonetic, kw_phonetic, max_dist=threshold
                )
                d_best = min(d_raw, d_phon)
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
            if 3 <= len(first_raw) <= 32:
                first_phon = phonetic_key(first_raw)
                for kw, (act_id, def_args) in self.TYPO_CATALOG.items():
                    if len(kw) >= 5:
                        d_raw = damerau_levenshtein(first_raw, kw.lower(), max_dist=2)
                        d_phon = damerau_levenshtein(
                            first_phon, phonetic_key(kw.lower()), max_dist=2
                        )
                        best_d = min(d_raw, d_phon)
                        if 0 < best_d <= 2:
                            args = dict(def_args)
                            args["_typo_corrected_from"] = first_raw
                            if "redis" in clean.lower():
                                args["target"] = "redis"
                                return act_id, args
                            if "postgres" in clean.lower():
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
    required_slots: list[str] = field(default_factory=list)
    default_arguments: dict[str, Any] = field(default_factory=dict)


class SemanticCache:
    """Semantic intent cache with explicit polarity check and slot collision guard."""

    NEGATION_PATTERN = NEGATION_PATTERN
    MAX_CACHE_ENTRIES = 512

    def __init__(self, max_entries: int = MAX_CACHE_ENTRIES):
        self.max_entries = max(4, int(max_entries))
        self._seed_count = 2
        self._entries: list[CachedIntentEntry] = [
            CachedIntentEntry(
                query_template="zrestartuj proces bazy danych",
                action_id="service.restart",
                is_negated=False,
                required_slots=["target"],
                default_arguments={"target": "database"},
            ),
            CachedIntentEntry(
                query_template="wyczyść pamięć podręczną na serwerze",
                action_id="cache.cleanup",
                is_negated=False,
                required_slots=["environment"],
                default_arguments={"environment": "default"},
            ),
        ]

    def check_polarity(self, text: str) -> bool:
        if not isinstance(text, str):
            return False
        return bool(self.NEGATION_PATTERN.search(text))

    def extract_slots(self, text: str) -> dict[str, Any]:
        slots: dict[str, Any] = {}
        if not isinstance(text, str) or len(text) > MAX_QUERY_CHARS:
            return slots
        lower = text.lower()
        env_match = re.search(
            r"\b(prod|production|staging|test|testing|dev|development)\b", lower
        )
        if env_match:
            slots["environment"] = (
                "prod" if env_match.group(1) in ("prod", "production") else "test"
            )
        num_match = re.search(r"(?<![0-9])([0-9]{1,18})(?![0-9])", lower)
        if num_match:
            slots["count"] = int(num_match.group(1))
        return slots

    def lookup(self, text: str) -> tuple[str, dict[str, Any], float] | None:
        if not isinstance(text, str) or not text.strip() or len(text) > MAX_QUERY_CHARS:
            return None
        is_query_negated = self.check_polarity(text)
        query_slots = self.extract_slots(text)

        for entry in self._entries:
            if entry.is_negated != is_query_negated:
                continue
            if (
                "environment" in query_slots
                and "environment" in entry.required_slots
                and query_slots["environment"] == "prod"
                and entry.default_arguments.get("environment") != "prod"
            ):
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
        arguments: dict[str, Any] | None = None,
        required_slots: list[str] | None = None,
    ) -> bool:
        """
        Dynamically promote a verified Layer 3 LLM compilation result into Layer 1.5 cache.
        Enforces polarity safety: never promotes a negated query as positive.
        """
        if (
            not isinstance(query, str)
            or not isinstance(action_id, str)
            or len(query) > MAX_QUERY_CHARS
        ):
            return False
        if arguments is not None and not isinstance(arguments, dict):
            return False
        clean = query.strip()
        clean_action = action_id.strip()
        if not clean or not clean_action:
            return False
        is_negated = self.check_polarity(clean)
        for existing in self._entries:
            if existing.query_template.lower() == clean.lower():
                return False
        while (
            len(self._entries) >= self.max_entries
            and len(self._entries) > self._seed_count
        ):
            self._entries.pop(self._seed_count)
        slots = (
            list(required_slots)
            if required_slots is not None
            else list((arguments or {}).keys())
        )
        self._entries.append(
            CachedIntentEntry(
                query_template=clean,
                action_id=clean_action,
                is_negated=is_negated,
                required_slots=slots,
                default_arguments=dict(arguments or {}),
            )
        )
        return True


__all__ = [
    "MODEL_PROFILES",
    "POLISH_PHONETIC_REPLACEMENTS",
    "CachedIntentEntry",
    "DigitalTwinContext",
    "DigitalTwinPromptSummarizer",
    "FastPathParser",
    "LLMCompilationResult",
    "LLMCompiler",
    "ModelProfile",
    "SemanticCache",
    "SingleResidentModelManager",
    "SystemTelemetryObserver",
    "VoiceTokenStreamHandler",
    "damerau_levenshtein",
    "phonetic_key",
    "strip_accents",
]
