"""Core canonical DSL, fast-path NL parser, semantic cache, and Option Network models."""

from __future__ import annotations

import json
import re
import shlex
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, ClassVar

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _format_dsl_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    text = str(value)
    if not text or any(ch.isspace() or ch in ('"', "'") for ch in text):
        return json.dumps(text, ensure_ascii=False)
    return text


@dataclass
class DSLCommand:
    entity: str
    operation: str
    custom_operation: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    filters: dict[str, Any] = field(default_factory=dict)
    idempotency_key: str | None = None
    raw_dsl: str | None = None

    def get_param(self, key: str, default: Any = None) -> Any:
        """Return a parameter from either filters or arguments."""
        if key in self.filters:
            return self.filters[key]
        return self.arguments.get(key, default)

    def to_canonical_string(self) -> str:
        """Serialize to canonical DSL text: entity.operation [key=val ...]."""
        op = self.custom_operation if self.operation == "custom" and self.custom_operation else self.operation
        parts = [f"{self.entity}.{op}"]
        for key, val in sorted(self.filters.items()):
            parts.append(f"filter:{key}={_format_dsl_value(val)}")
        for key, val in sorted(self.arguments.items()):
            parts.append(f"{key}={_format_dsl_value(val)}")
        return " ".join(parts)

    @classmethod
    def from_text(cls, text: str) -> DSLCommand:
        """Parse text DSL in format 'entity.operation [arg=val ...]' or 'operation entity [arg=val ...]'."""
        invalid_type = not isinstance(text, str)
        if invalid_type:
            raise ValueError("DSL command must be a string")
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("Empty DSL command string")
        try:
            tokens = shlex.split(cleaned, posix=True)
        except ValueError as exc:
            raise ValueError(f"Malformed quoted DSL command: {exc}") from exc
        if not tokens:
            raise ValueError("Empty DSL command string")

        head = tokens[0]
        params: dict[str, Any] = {}
        filters: dict[str, Any] = {}

        if "." in head:
            entity, operation = head.split(".", 1)
            arg_tokens = tokens[1:]
        elif len(tokens) >= 2 and not any("=" in tok for tok in tokens[:2]):
            operation, entity = tokens[0], tokens[1]
            arg_tokens = tokens[2:]
        else:
            raise ValueError(
                f"Invalid DSL command format: '{cleaned}'. Expected 'entity.operation' or 'operation entity'"
            )

        if not _IDENTIFIER_RE.fullmatch(entity) or not _IDENTIFIER_RE.fullmatch(operation):
            raise ValueError(f"Invalid DSL entity or operation identifier in '{cleaned}'")

        for token in arg_tokens:
            if "=" not in token:
                raise ValueError(f"Invalid DSL parameter token without '=': '{token}'")
            key, val = token.split("=", 1)
            if key.startswith("filter:"):
                filter_key = key[7:]
                if not filter_key:
                    raise ValueError("Empty filter key in DSL command")
                filters[filter_key] = val
            else:
                if not key:
                    raise ValueError("Empty parameter key in DSL command")
                params[key] = val

        return cls(
            entity=entity.lower(),
            operation=operation.lower(),
            arguments=params,
            filters=filters,
            raw_dsl=cleaned,
        )


@dataclass
class DSLResult:
    success: bool
    status: str
    data: Any
    errors: list[dict[str, str]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DSLExecutor:
    """Canonical DSL execution engine with handler dispatch and validation."""

    def __init__(self) -> None:
        self._handlers: dict[tuple[str, str], Callable[[DSLCommand], Any]] = {}

    def register(self, entity: str, operation: str, handler: Callable[[DSLCommand], Any]) -> None:
        self._handlers[(entity.lower(), operation.lower())] = handler

    def execute(self, cmd: DSLCommand, source_layer: str = "direct_dsl") -> DSLResult:
        start_time = time.perf_counter()
        key = (cmd.entity.lower(), cmd.operation.lower())
        trace_id = str(uuid.uuid4())[:8]
        canonical = cmd.to_canonical_string()

        if key not in self._handlers:
            elapsed = round((time.perf_counter() - start_time) * 1000, 3)
            return DSLResult(
                success=False,
                status="NOT_FOUND",
                data=None,
                errors=[{"code": "DSL_UNKNOWN_HANDLER", "message": f"No handler registered for {cmd.entity}.{cmd.operation}"}],
                meta={"executionTimeMs": elapsed, "sourceLayer": source_layer, "canonicalDsl": canonical, "traceId": trace_id},
            )

        try:
            result_data = self._handlers[key](cmd)
            elapsed = round((time.perf_counter() - start_time) * 1000, 3)
            return DSLResult(
                success=True,
                status="OK",
                data=result_data,
                errors=[],
                meta={"executionTimeMs": elapsed, "sourceLayer": source_layer, "canonicalDsl": canonical, "traceId": trace_id},
            )
        except Exception as exc:  # noqa: BLE001 - handler boundary converts exceptions to DSLResult
            elapsed = round((time.perf_counter() - start_time) * 1000, 3)
            return DSLResult(
                success=False,
                status="EXECUTION_ERROR",
                data=None,
                errors=[{"code": "HANDLER_EXCEPTION", "message": str(exc)}],
                meta={"executionTimeMs": elapsed, "sourceLayer": source_layer, "canonicalDsl": canonical, "traceId": trace_id},
            )


def _load_intent_lexicon() -> dict[str, Any]:
    profile_path = Path(__file__).resolve().with_name("semantic-profile.json")
    data = json.loads(profile_path.read_text(encoding="utf-8"))
    return dict(data.get("intent_lexicon") or {})


_INTENT_LEXICON = _load_intent_lexicon()
_NEGATION_TOKENS = ("nie", "not", "bez", "don'?t", "never", "nigdy", "odrzuć", "odrzuc", "without")


class NLIntentParser:
    """Deterministic rule-based multilingual (PL + EN) natural language parser."""

    VERB_SYNONYMS: ClassVar[dict[str, list[str]]] = dict(_INTENT_LEXICON.get("verbs") or {})
    ENTITY_SYNONYMS: ClassVar[dict[str, list[str]]] = dict(_INTENT_LEXICON.get("entities") or {})
    STATUS_FILTERS: ClassVar[dict[str, list[str]]] = dict(_INTENT_LEXICON.get("status_filters") or {})

    def parse(self, text: str) -> DSLCommand | None:
        if not isinstance(text, str):
            return None
        clean = text.strip()
        if not clean:
            return None

        if re.match(r"^[a-z0-9_-]+\.[a-z0-9_-]+(?:\s|$)", clean, re.IGNORECASE):
            try:
                return DSLCommand.from_text(clean)
            except ValueError:
                pass

        lower = clean.lower()
        detected_op = None
        for op, synonyms in self.VERB_SYNONYMS.items():
            if any(re.search(rf"\b{re.escape(syn)}\b", lower) for syn in synonyms):
                detected_op = op
                break
        if not detected_op:
            detected_op = "list"

        detected_entity = None
        for entity, synonyms in self.ENTITY_SYNONYMS.items():
            if any(re.search(rf"\b{re.escape(syn)}\b", lower) for syn in synonyms):
                detected_entity = entity
                break
        if not detected_entity:
            return None

        arguments: dict[str, Any] = {}
        filters: dict[str, Any] = {}
        for status_code, keywords in self.STATUS_FILTERS.items():
            if any(re.search(rf"\b{re.escape(kw)}\b", lower) for kw in keywords):
                filters["status"] = status_code
                break

        id_match = re.search(r"(?:ticket-|#)?([0-9]{1,4})\b", lower)
        if id_match and detected_op in ("get", "close", "delete"):
            arguments["id"] = id_match.group(1)

        if detected_op == "create":
            title_match = re.search(r'["\']([^"\']+)["\']', clean)
            if title_match:
                arguments["title"] = title_match.group(1)
            else:
                stripped = lower
                for syn in self.VERB_SYNONYMS.get("create", ()):
                    stripped = re.sub(rf"\b{re.escape(syn)}\b", "", stripped)
                for syn in self.ENTITY_SYNONYMS.get(detected_entity, ()):
                    stripped = re.sub(rf"\b{re.escape(syn)}\b", "", stripped)
                stripped = stripped.strip(": -")
                if stripped:
                    arguments["title"] = stripped

        return DSLCommand(entity=detected_entity, operation=detected_op, arguments=arguments, filters=filters, raw_dsl=None)


@dataclass
class CachedIntent:
    pattern: str
    command: DSLCommand
    slots: dict[str, Any] = field(default_factory=dict)
    is_negated: bool = False


class SemanticCache:
    """Vector-inspired intent cache with explicit polarity and slot guards."""

    NEGATION_REGEX = re.compile(r"\b(" + "|".join(_NEGATION_TOKENS) + r")\b", re.IGNORECASE)

    def __init__(self, confidence_threshold: float = 0.85) -> None:
        self.confidence_threshold = confidence_threshold
        self._entries: list[CachedIntent] = []

    def register_template(
        self, pattern: str, command: DSLCommand, slots: dict[str, Any] | None = None
    ) -> None:
        neg = bool(self.NEGATION_REGEX.search(pattern))
        self._entries.append(
            CachedIntent(pattern=pattern.strip().lower(), command=command, slots=dict(slots or {}), is_negated=neg)
        )

    def match(self, text: str) -> tuple[DSLCommand, float] | None:
        if not isinstance(text, str):
            return None
        clean = text.strip().lower()
        if not clean:
            return None

        query_negated = bool(self.NEGATION_REGEX.search(clean))
        tokens = set(re.findall(r"\b\w+\b", clean))
        best_match: tuple[DSLCommand, float] | None = None
        best_score = 0.0

        for entry in self._entries:
            if entry.is_negated != query_negated:
                continue
            entry_tokens = set(re.findall(r"\b\w+\b", entry.pattern))
            if not entry_tokens:
                continue
            union = len(tokens | entry_tokens)
            score = (len(tokens & entry_tokens) / union) if union > 0 else 0.0

            slot_mismatch = False
            for key, expected_val in entry.slots.items():
                expected_str = str(expected_val).lower()
                if expected_str and expected_str in clean:
                    continue
                if key == "env" and ("prod" in clean or "test" in clean or "staging" in clean):
                    slot_mismatch = True
                    break
            if slot_mismatch:
                continue

            if score > best_score and score >= self.confidence_threshold:
                best_score = score
                best_match = (entry.command, score)

        return best_match


@dataclass
class DigitalTwinContext:
    environment: str = "prod"
    active_services: list[str] = field(default_factory=list)
    degraded_services: list[str] = field(default_factory=list)
    active_tickets: list[str] = field(default_factory=list)
    cluster_nodes: list[str] = field(default_factory=list)
    custom_state: dict[str, Any] = field(default_factory=dict)


@dataclass
class OptionNode:
    id: str
    display_label: str
    completion_text: str
    action_id: str
    resolved_uri: str
    bound_arguments: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    requires_confirmation: bool = False
    badge: str = ""
    next_options: list[OptionNode] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OptionNetworkEngine:
    """Sub-5ms Real-time Autocomplete and Contextual Option Network."""

    ACTION_TEMPLATES: ClassVar[list[dict[str, Any]]] = [
        {
            "verb_patterns": ["zrestartuj", "restart", "restartuj", "reboot"],
            "action_id": "service.restart",
            "uri_pattern": "proc://taskand.dev/service/restart/v1",
            "entity_type": "service",
            "destructive": True,
            "next_suboptions": [
                {"flag": "--graceful", "label": "Łagodny restart (z zachowaniem połączeń)"},
                {"flag": "--force", "label": "Wymuszony natychmiastowy restart"},
            ],
        },
        {
            "verb_patterns": ["zatrzymaj", "stop", "wyłącz", "halt"],
            "action_id": "service.stop",
            "uri_pattern": "proc://taskand.dev/service/stop/v1",
            "entity_type": "service",
            "destructive": True,
            "next_suboptions": [
                {"flag": "--graceful", "label": "Łagodne zatrzymanie"},
                {"flag": "--force", "label": "Wymuszone natychmiastowe zatrzymanie"},
            ],
        },
        {
            "verb_patterns": ["status", "stan", "pokaż stan", "sprawdź stan", "check"],
            "action_id": "service.status",
            "uri_pattern": "proc://taskand.dev/service/status/v1",
            "entity_type": "service",
            "destructive": False,
            "next_suboptions": [
                {"flag": "--verbose", "label": "Pełne szczegóły i metryki"},
                {"flag": "--tail", "label": "Ostatnie 50 linii logów"},
            ],
        },
        {
            "verb_patterns": ["zamknij", "close", "zakończ", "finish"],
            "action_id": "ticket.close",
            "uri_pattern": "proc://taskand.dev/ticket/close/v1",
            "entity_type": "ticket",
            "destructive": False,
            "next_suboptions": [
                {"flag": "--resolved", "label": "Jako rozwiązany"},
                {"flag": "--wontfix", "label": "Jako odrzucony / wontfix"},
            ],
        },
        {
            "verb_patterns": ["pokaż", "list", "wypisz", "lista"],
            "action_id": "ticket.list",
            "uri_pattern": "proc://taskand.dev/ticket/list/v1",
            "entity_type": "ticket_list",
            "destructive": False,
            "next_suboptions": [
                {"flag": "status=OPEN", "label": "Tylko otwarte zadania"},
                {"flag": "status=CLOSED", "label": "Tylko zamknięte zadania"},
            ],
        },
    ]

    def suggest(
        self, partial_query: str, twin: DigitalTwinContext | None = None, max_suggestions: int = 5
    ) -> list[OptionNode]:
        if not isinstance(partial_query, str) or max_suggestions <= 0:
            return []
        clean = partial_query.strip().lower()
        if not clean:
            return []

        twin = twin or DigitalTwinContext()
        results: list[OptionNode] = []

        for tpl in self.ACTION_TEMPLATES:
            if not any(v.startswith(clean) or clean.startswith(v[:3]) or v in clean for v in tpl["verb_patterns"]):
                continue
            verb0, act_id, uri_pat = tpl["verb_patterns"][0], tpl["action_id"], tpl["uri_pattern"]
            entity_type = tpl["entity_type"]
            subs = tpl.get("next_suboptions", [])

            if entity_type == "service":
                services = list(twin.active_services) if twin.active_services else ["default-service"]
                for svc in sorted(services, key=lambda s: 0 if s in twin.degraded_services else 1):
                    is_degraded = svc in twin.degraded_services
                    badge = "DEGRADED" if is_degraded else ("PROD" if twin.environment == "prod" else "")
                    suffix = " (Zdegradowany)" if is_degraded else (f" [{twin.environment}]" if twin.environment == "prod" else "")
                    confirm = twin.environment == "prod" and bool(tpl["destructive"])
                    children = [
                        OptionNode(
                            id=f"{act_id}:{svc}:{sub['flag']}", display_label=sub["label"],
                            completion_text=f"{verb0} {svc} {sub['flag']}", action_id=act_id, resolved_uri=uri_pat,
                            bound_arguments={"target": svc, "flag": sub["flag"], "env": twin.environment},
                            confidence=0.95, requires_confirmation=confirm, badge=badge,
                        )
                        for sub in subs
                    ]
                    results.append(OptionNode(
                        id=f"{act_id}:{svc}", display_label=f"{verb0.capitalize()} {svc}{suffix}",
                        completion_text=f"{verb0} {svc}", action_id=act_id, resolved_uri=uri_pat,
                        bound_arguments={"target": svc, "env": twin.environment},
                        confidence=0.98 if is_degraded else 0.90, requires_confirmation=confirm, badge=badge, next_options=children,
                    ))
            elif entity_type == "ticket":
                for ticket_id in (list(twin.active_tickets) if twin.active_tickets else ["ticket-001"]):
                    children = [
                        OptionNode(
                            id=f"{act_id}:{ticket_id}:{sub['flag']}", display_label=sub["label"],
                            completion_text=f"{verb0} {ticket_id} {sub['flag']}", action_id=act_id, resolved_uri=uri_pat,
                            bound_arguments={"id": ticket_id, "resolution": sub["flag"]}, confidence=0.95,
                        )
                        for sub in subs
                    ]
                    results.append(OptionNode(
                        id=f"{act_id}:{ticket_id}", display_label=f"{verb0.capitalize()} {ticket_id}",
                        completion_text=f"{verb0} {ticket_id}", action_id=act_id, resolved_uri=uri_pat,
                        bound_arguments={"id": ticket_id}, confidence=0.92, next_options=children,
                    ))
            elif entity_type == "ticket_list":
                children = [
                    OptionNode(
                        id=f"{act_id}:{sub['flag']}", display_label=sub["label"],
                        completion_text=f"{verb0} zadania {sub['flag']}", action_id=act_id, resolved_uri=uri_pat,
                        bound_arguments={"status": sub["flag"].split("=")[-1]}, confidence=0.95,
                    )
                    for sub in subs
                ]
                results.append(OptionNode(
                    id=f"{act_id}:all", display_label=f"{verb0.capitalize()} zadania",
                    completion_text=f"{verb0} zadania", action_id=act_id, resolved_uri=uri_pat,
                    confidence=0.88, next_options=children,
                ))

        return results[:max_suggestions]
