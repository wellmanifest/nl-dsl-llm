#!/usr/bin/env python3
"""
wellmanifest/nl-dsl-llm reference standard implementation.
Implements the 3-layer architecture:
  Layer 1: Deterministic NL Pattern Parser (Fast Path, PL & EN)
  Layer 2: Canonical DSL Engine (Single Execution Boundary & Validator)
  Layer 3: LLM Translation Fallback (Adaptive Path)
Plus interface parity adapters: CLI, Web REST API, MCP Server.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# Layer 2: Canonical DSL Models & Registry
# ---------------------------------------------------------------------------

@dataclass
class DSLCommand:
    entity: str
    operation: str
    custom_operation: Optional[str] = None
    arguments: Dict[str, Any] = field(default_factory=dict)
    filters: Dict[str, Any] = field(default_factory=dict)
    idempotency_key: Optional[str] = None
    raw_dsl: Optional[str] = None

    def get_param(self, key: str, default: Any = None) -> Any:
        """Helper to get a parameter from either filters or arguments."""
        if key in self.filters:
            return self.filters[key]
        return self.arguments.get(key, default)

    def to_canonical_string(self) -> str:
        """Serialize to canonical DSL text: entity.operation [key=val ...]"""
        op = self.custom_operation if self.operation == "custom" and self.custom_operation else self.operation
        parts = [f"{self.entity}.{op}"]
        for k, v in sorted(self.filters.items()):
            parts.append(f"filter:{k}={json.dumps(v) if isinstance(v, (dict, list)) else v}")
        for k, v in sorted(self.arguments.items()):
            parts.append(f"{k}={json.dumps(v) if isinstance(v, (dict, list)) else v}")
        return " ".join(parts)

    @classmethod
    def from_text(cls, text: str) -> DSLCommand:
        """
        Parse text DSL in format:
          entity.operation [arg=val ...]
          or: operation entity [arg=val ...]
        """
        text = text.strip()
        tokens = text.split()
        if not tokens:
            raise ValueError("Empty DSL command string")

        head = tokens[0]
        params: Dict[str, Any] = {}
        filters: Dict[str, Any] = {}

        if "." in head:
            entity, operation = head.split(".", 1)
        elif len(tokens) >= 2 and not any("=" in tok for tok in tokens[:2]):
            operation = tokens[0]
            entity = tokens[1]
            tokens = tokens[1:]
        else:
            raise ValueError(f"Invalid DSL command format: '{text}'. Expected 'entity.operation' or 'operation entity'")

        for token in tokens[1:]:
            if "=" in token:
                k, v = token.split("=", 1)
                # Strip quotes if present
                if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                    v = v[1:-1]
                if k.startswith("filter:"):
                    filters[k[7:]] = v
                else:
                    params[k] = v

        return cls(
            entity=entity.lower(),
            operation=operation.lower(),
            arguments=params,
            filters=filters,
            raw_dsl=text
        )


@dataclass
class DSLResult:
    success: bool
    status: str
    data: Any
    errors: List[Dict[str, str]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DSLExecutor:
    """Canonical DSL execution engine with handler dispatch and validation."""

    def __init__(self):
        self._handlers: Dict[Tuple[str, str], Callable[[DSLCommand], Any]] = {}

    def register(self, entity: str, operation: str, handler: Callable[[DSLCommand], Any]) -> None:
        self._handlers[(entity.lower(), operation.lower())] = handler

    def execute(self, cmd: DSLCommand, source_layer: str = "direct_dsl") -> DSLResult:
        start_time = time.perf_counter()
        key = (cmd.entity.lower(), cmd.operation.lower())

        if key not in self._handlers:
            elapsed = (time.perf_counter() - start_time) * 1000
            return DSLResult(
                success=False,
                status="NOT_FOUND",
                data=None,
                errors=[{"code": "DSL_UNKNOWN_HANDLER", "message": f"No handler registered for {cmd.entity}.{cmd.operation}"}],
                meta={
                    "executionTimeMs": round(elapsed, 3),
                    "sourceLayer": source_layer,
                    "canonicalDsl": cmd.to_canonical_string(),
                    "traceId": str(uuid.uuid4())[:8]
                }
            )

        try:
            handler = self._handlers[key]
            result_data = handler(cmd)
            elapsed = (time.perf_counter() - start_time) * 1000
            return DSLResult(
                success=True,
                status="OK",
                data=result_data,
                errors=[],
                meta={
                    "executionTimeMs": round(elapsed, 3),
                    "sourceLayer": source_layer,
                    "canonicalDsl": cmd.to_canonical_string(),
                    "traceId": str(uuid.uuid4())[:8]
                }
            )
        except Exception as e:
            elapsed = (time.perf_counter() - start_time) * 1000
            return DSLResult(
                success=False,
                status="EXECUTION_ERROR",
                data=None,
                errors=[{"code": "HANDLER_EXCEPTION", "message": str(e)}],
                meta={
                    "executionTimeMs": round(elapsed, 3),
                    "sourceLayer": source_layer,
                    "canonicalDsl": cmd.to_canonical_string(),
                    "traceId": str(uuid.uuid4())[:8]
                }
            )


# ---------------------------------------------------------------------------
# Layer 1: Deterministic NL Pattern Parser (Fast Path)
# ---------------------------------------------------------------------------

class NLIntentParser:
    """
    Deterministic rule-based multilingual (PL + EN) natural language parser.
    Zero latency (<1ms), zero external API cost.
    """

    VERB_SYNONYMS = {
        # Query/List
        "list": [
            "pokaż", "pokaz", "wypisz", "wyświetl", "wyswietl", "lista", "listuj",
            "pobierz", "sprawdź", "sprawdz", "znajdź", "znajdz", "szukaj", "podsumuj",
            "show", "list", "display", "view", "get", "fetch", "check", "find", "search", "summary"
        ],
        # Get by ID
        "get": [
            "pokaż szczegóły", "szczegóły", "szczegoly", "dane", "pobierz",
            "get", "show details", "details", "inspect"
        ],
        # Create/Add
        "create": [
            "dodaj", "utwórz", "utworz", "stwórz", "stworz", "nowy", "nowa", "nowe", "załóż", "zaloz",
            "add", "create", "new", "register"
        ],
        # Close/Complete
        "close": [
            "zamknij", "zakończ", "zakoncz", "oznacz jako zrobione", "zrobione", "gotowe",
            "close", "complete", "finish", "done"
        ],
        # Delete/Remove
        "delete": [
            "usuń", "usun", "skasuj", "odrzuć", "odrzuc", "wyrzuć", "wyrzuc",
            "delete", "remove", "drop"
        ]
    }

    ENTITY_SYNONYMS = {
        "ticket": ["ticket", "tickety", "ticketów", "ticketow", "zadanie", "zadania", "zadań", "zadan", "zgłoszenie", "task", "tasks", "tickets"],
        "pr": ["pr", "prs", "pull request", "pull requesty", "pull requestów", "pulls", "mr"],
        "account": ["konto", "konta", "kont", "account", "accounts", "provider"],
        "token": ["token", "tokeny", "tokenów", "tokens", "usage", "zużycie", "zuzycie"],
        "process": ["proces", "procesy", "procesów", "agent", "agenty", "agentów", "process", "processes", "agents"]
    }

    def parse(self, text: str) -> Optional[DSLCommand]:
        clean = text.strip()
        if not clean:
            return None

        # Direct DSL check
        if re.match(r"^[a-z0-9_-]+\.[a-z0-9_-]+", clean, re.IGNORECASE):
            try:
                return DSLCommand.from_text(clean)
            except Exception:
                pass

        lower = clean.lower()

        # Detect Operation
        detected_op = None
        for op, synonyms in self.VERB_SYNONYMS.items():
            for syn in synonyms:
                pattern = rf"\b{re.escape(syn)}\b"
                if re.search(pattern, lower):
                    detected_op = op
                    break
            if detected_op:
                break

        # Default to 'list' if query mentions entities without explicit verb
        if not detected_op:
            detected_op = "list"

        # Detect Entity
        detected_entity = None
        for entity, synonyms in self.ENTITY_SYNONYMS.items():
            for syn in synonyms:
                pattern = rf"\b{re.escape(syn)}\b"
                if re.search(pattern, lower):
                    detected_entity = entity
                    break
            if detected_entity:
                break

        if not detected_entity:
            return None

        arguments: Dict[str, Any] = {}
        filters: Dict[str, Any] = {}

        # Status / Condition extraction (PL & EN)
        if re.search(r"\b(niescalone|otwarte|niezamknięte|unmerged|open)\b", lower):
            filters["status"] = "OPEN"
        elif re.search(r"\b(scalone|zamknięte|zrobione|merged|closed|done)\b", lower):
            filters["status"] = "CLOSED"

        # Extract numeric ID if present (e.g. ticket-001 or #123 or 45)
        id_match = re.search(r"(?:ticket-|#)?([0-9]{1,4})\b", lower)
        if id_match and detected_op in ("get", "close", "delete"):
            arguments["id"] = id_match.group(1)

        # Extract title/name if create
        if detected_op == "create":
            # Extract content in quotes or after 'o tytule' / 'title'
            title_match = re.search(r'["\']([^"\']+)["\']', clean)
            if title_match:
                arguments["title"] = title_match.group(1)
            else:
                # Fallback: strip the command trigger words
                stripped = lower
                for syn in self.VERB_SYNONYMS["create"]:
                    stripped = re.sub(rf"\b{re.escape(syn)}\b", "", stripped)
                for syn in self.ENTITY_SYNONYMS.get(detected_entity, []):
                    stripped = re.sub(rf"\b{re.escape(syn)}\b", "", stripped)
                stripped = stripped.strip(": -")
                if stripped:
                    arguments["title"] = stripped

        return DSLCommand(
            entity=detected_entity,
            operation=detected_op,
            arguments=arguments,
            filters=filters,
            raw_dsl=None
        )


# ---------------------------------------------------------------------------
# Layer 1.5: Semantic Intent Cache & Candidate Matcher
# ---------------------------------------------------------------------------

@dataclass
class CachedIntent:
    pattern: str
    command: DSLCommand
    slots: Dict[str, Any] = field(default_factory=dict)
    is_negated: bool = False


class SemanticCache:
    """
    Vector-inspired intent cache with explicit polarity and slot guards.
    Prevents fuzzy matching traps (negation inversion, target mismatch).
    """

    NEGATION_REGEX = re.compile(
        r"\b(nie|not|bez|don'?t|never|nigdy|odrzuć|odrzuc|without)\b",
        re.IGNORECASE,
    )

    def __init__(self, confidence_threshold: float = 0.85):
        self.confidence_threshold = confidence_threshold
        self._entries: List[CachedIntent] = []

    def register_template(
        self, pattern: str, command: DSLCommand, slots: Optional[Dict[str, Any]] = None
    ) -> None:
        neg = bool(self.NEGATION_REGEX.search(pattern))
        self._entries.append(
            CachedIntent(
                pattern=pattern.strip().lower(),
                command=command,
                slots=slots or {},
                is_negated=neg,
            )
        )

    def match(self, text: str) -> Optional[Tuple[DSLCommand, float]]:
        clean = text.strip().lower()
        if not clean:
            return None

        query_negated = bool(self.NEGATION_REGEX.search(clean))
        tokens = set(re.findall(r"\b\w+\b", clean))

        best_match: Optional[Tuple[DSLCommand, float]] = None
        best_score = 0.0

        for entry in self._entries:
            # 1. Explicit Polarity Guard
            if entry.is_negated != query_negated:
                continue

            # 2. Token Jaccard overlap similarity
            entry_tokens = set(re.findall(r"\b\w+\b", entry.pattern))
            if not entry_tokens:
                continue

            intersection = len(tokens & entry_tokens)
            union = len(tokens | entry_tokens)
            score = intersection / union if union > 0 else 0.0

            # 3. Target Slot Guard (e.g. env=prod vs env=test)
            slot_mismatch = False
            for k, expected_v in entry.slots.items():
                if expected_v in clean:
                    pass
                elif k == "env" and ("prod" in clean or "test" in clean):
                    slot_mismatch = True
                    break

            if slot_mismatch:
                continue

            if score > best_score and score >= self.confidence_threshold:
                best_score = score
                best_match = (entry.command, score)

        return best_match


# ---------------------------------------------------------------------------
# Layer 0.5: Real-time Autocomplete & Contextual Option Network
# ---------------------------------------------------------------------------

@dataclass
class DigitalTwinContext:
    environment: str = "prod"
    active_services: List[str] = field(default_factory=list)
    degraded_services: List[str] = field(default_factory=list)
    active_tickets: List[str] = field(default_factory=list)
    cluster_nodes: List[str] = field(default_factory=list)
    custom_state: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptionNode:
    id: str
    display_label: str
    completion_text: str
    action_id: str
    resolved_uri: str
    bound_arguments: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    requires_confirmation: bool = False
    badge: str = ""
    next_options: List[OptionNode] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class OptionNetworkEngine:
    """
    Sub-5ms Real-time Autocomplete and Contextual Option Network.
    Projects Digital Twin environment state into interactive option DAGs.
    """

    ACTION_TEMPLATES = [
        {
            "verb_patterns": ["zrestartuj", "restart", "restartuj", "reboot"],
            "action_id": "service.restart",
            "uri_pattern": "proc://taskand.dev/service/restart/v1",
            "entity_type": "service",
            "destructive": True,
            "next_suboptions": [
                {"flag": "--graceful", "label": "Łagodny restart (z zachowaniem połączeń)"},
                {"flag": "--force", "label": "Wymuszony natychmiastowy restart"}
            ]
        },
        {
            "verb_patterns": ["zatrzymaj", "stop", "wyłącz", "halt"],
            "action_id": "service.stop",
            "uri_pattern": "proc://taskand.dev/service/stop/v1",
            "entity_type": "service",
            "destructive": True,
            "next_suboptions": [
                {"flag": "--graceful", "label": "Łagodne zatrzymanie"},
                {"flag": "--force", "label": "Wymuszone natychmiastowe zatrzymanie"}
            ]
        },
        {
            "verb_patterns": ["status", "stan", "pokaż stan", "sprawdź stan", "check"],
            "action_id": "service.status",
            "uri_pattern": "proc://taskand.dev/service/status/v1",
            "entity_type": "service",
            "destructive": False,
            "next_suboptions": [
                {"flag": "--verbose", "label": "Pełne szczegóły i metryki"},
                {"flag": "--tail", "label": "Ostatnie 50 linii logów"}
            ]
        },
        {
            "verb_patterns": ["zamknij", "close", "zakończ", "finish"],
            "action_id": "ticket.close",
            "uri_pattern": "proc://taskand.dev/ticket/close/v1",
            "entity_type": "ticket",
            "destructive": False,
            "next_suboptions": [
                {"flag": "--resolved", "label": "Jako rozwiązany"},
                {"flag": "--wontfix", "label": "Jako odrzucony / wontfix"}
            ]
        },
        {
            "verb_patterns": ["pokaż", "list", "wypisz", "lista"],
            "action_id": "ticket.list",
            "uri_pattern": "proc://taskand.dev/ticket/list/v1",
            "entity_type": "ticket_list",
            "destructive": False,
            "next_suboptions": [
                {"flag": "status=OPEN", "label": "Tylko otwarte zadania"},
                {"flag": "status=CLOSED", "label": "Tylko zamknięte zadania"}
            ]
        }
    ]

    def suggest(
        self,
        partial_query: str,
        twin: Optional[DigitalTwinContext] = None,
        max_suggestions: int = 5,
    ) -> List[OptionNode]:
        clean = partial_query.strip().lower()
        if not clean:
            return []

        twin = twin or DigitalTwinContext()
        results: List[OptionNode] = []

        for tpl in self.ACTION_TEMPLATES:
            verb_matched = False
            for v in tpl["verb_patterns"]:
                if v.startswith(clean) or clean.startswith(v[:3]) or v in clean:
                    verb_matched = True
                    break

            if not verb_matched:
                continue

            entity_type = tpl["entity_type"]

            if entity_type == "service":
                services = list(twin.active_services) if twin.active_services else ["default-service"]
                for svc in sorted(services, key=lambda s: 0 if s in twin.degraded_services else 1):
                    is_degraded = svc in twin.degraded_services
                    badge = "DEGRADED" if is_degraded else ("PROD" if twin.environment == "prod" else "")
                    label = f"{tpl['verb_patterns'][0].capitalize()} {svc}"
                    if is_degraded:
                        label += " (Zdegradowany)"
                    elif twin.environment == "prod":
                        label += f" [{twin.environment}]"

                    children: List[OptionNode] = []
                    for sub in tpl.get("next_suboptions", []):
                        children.append(
                            OptionNode(
                                id=f"{tpl['action_id']}:{svc}:{sub['flag']}",
                                display_label=sub["label"],
                                completion_text=f"{tpl['verb_patterns'][0]} {svc} {sub['flag']}",
                                action_id=tpl["action_id"],
                                resolved_uri=tpl["uri_pattern"],
                                bound_arguments={"target": svc, "flag": sub["flag"], "env": twin.environment},
                                confidence=0.95,
                                requires_confirmation=(twin.environment == "prod" and tpl["destructive"]),
                                badge=badge,
                                next_options=[]
                            )
                        )

                    results.append(
                        OptionNode(
                            id=f"{tpl['action_id']}:{svc}",
                            display_label=label,
                            completion_text=f"{tpl['verb_patterns'][0]} {svc}",
                            action_id=tpl["action_id"],
                            resolved_uri=tpl["uri_pattern"],
                            bound_arguments={"target": svc, "env": twin.environment},
                            confidence=0.98 if is_degraded else 0.90,
                            requires_confirmation=(twin.environment == "prod" and tpl["destructive"]),
                            badge=badge,
                            next_options=children
                        )
                    )

            elif entity_type == "ticket":
                tickets = list(twin.active_tickets) if twin.active_tickets else ["ticket-001"]
                for t in tickets:
                    children = []
                    for sub in tpl.get("next_suboptions", []):
                        children.append(
                            OptionNode(
                                id=f"{tpl['action_id']}:{t}:{sub['flag']}",
                                display_label=sub["label"],
                                completion_text=f"{tpl['verb_patterns'][0]} {t} {sub['flag']}",
                                action_id=tpl["action_id"],
                                resolved_uri=tpl["uri_pattern"],
                                bound_arguments={"id": t, "resolution": sub["flag"]},
                                confidence=0.95,
                                requires_confirmation=False,
                                badge="",
                                next_options=[]
                            )
                        )
                    results.append(
                        OptionNode(
                            id=f"{tpl['action_id']}:{t}",
                            display_label=f"{tpl['verb_patterns'][0].capitalize()} {t}",
                            completion_text=f"{tpl['verb_patterns'][0]} {t}",
                            action_id=tpl["action_id"],
                            resolved_uri=tpl["uri_pattern"],
                            bound_arguments={"id": t},
                            confidence=0.92,
                            requires_confirmation=False,
                            badge="",
                            next_options=children
                        )
                    )

            elif entity_type == "ticket_list":
                children = []
                for sub in tpl.get("next_suboptions", []):
                    children.append(
                        OptionNode(
                            id=f"{tpl['action_id']}:{sub['flag']}",
                            display_label=sub["label"],
                            completion_text=f"{tpl['verb_patterns'][0]} zadania {sub['flag']}",
                            action_id=tpl["action_id"],
                            resolved_uri=tpl["uri_pattern"],
                            bound_arguments={"status": sub["flag"].split("=")[-1]},
                            confidence=0.95,
                            requires_confirmation=False,
                            badge="",
                            next_options=[]
                        )
                    )
                results.append(
                    OptionNode(
                        id=f"{tpl['action_id']}:all",
                        display_label=f"{tpl['verb_patterns'][0].capitalize()} zadania",
                        completion_text=f"{tpl['verb_patterns'][0]} zadania",
                        action_id=tpl["action_id"],
                        resolved_uri=tpl["uri_pattern"],
                        bound_arguments={},
                        confidence=0.88,
                        requires_confirmation=False,
                        badge="",
                        next_options=children
                    )
                )

        return results[:max_suggestions]


# ---------------------------------------------------------------------------
# Layer 3: LLM Translation Fallback (Adaptive Path)
# ---------------------------------------------------------------------------

class LLMTranslator:
    """
    Adaptive semantic fallback compiler.
    Translates unstructured or complex natural language into canonical DSL statements.
    """

    def __init__(self, model_name: str = "default", api_client: Any = None):
        self.model_name = model_name
        self.api_client = api_client

    def translate(self, prompt: str, schema_hints: Optional[Dict[str, Any]] = None) -> Optional[DSLCommand]:
        """
        Translates complex query to canonical DSLCommand.
        In standalone environments without active API keys, provides a deterministic semantic mock.
        """
        clean = prompt.strip()
        lower = clean.lower()
        if "przetestuj" in lower or "test" in lower:
            return DSLCommand(entity="test", operation="run", arguments={"query": clean})
        if "statystyki" in lower or "metrics" in lower:
            return DSLCommand(entity="metrics", operation="list", arguments={"subject": clean})

        return None


# ---------------------------------------------------------------------------
# Coordinator: NL-DSL-LLM Bridge
# ---------------------------------------------------------------------------

class NLDSLLLMBridge:
    """Coordinates Layer 0.5 (Option Network), Layer 1 (Fast Path), Layer 1.5 (Semantic Cache), Layer 3 (LLM Fallback), and Layer 2 (DSL Executor)."""

    def __init__(
        self,
        executor: DSLExecutor,
        parser: Optional[NLIntentParser] = None,
        translator: Optional[LLMTranslator] = None,
        cache: Optional[SemanticCache] = None,
        option_engine: Optional[OptionNetworkEngine] = None,
    ):
        self.executor = executor
        self.parser = parser or NLIntentParser()
        self.translator = translator or LLMTranslator()
        self.cache = cache or SemanticCache()
        self.option_engine = option_engine or OptionNetworkEngine()

    def suggest_options(
        self,
        partial_query: str,
        twin_context: Optional[DigitalTwinContext] = None,
        max_suggestions: int = 5,
    ) -> Dict[str, Any]:
        """Real-time autocomplete & contextual option network for text/voice input."""
        start = time.perf_counter()
        options = self.option_engine.suggest(
            partial_query, twin=twin_context, max_suggestions=max_suggestions
        )
        elapsed = (time.perf_counter() - start) * 1000
        return {
            "success": True,
            "partial_query": partial_query,
            "suggestions": [opt.to_dict() for opt in options],
            "latency_ms": round(elapsed, 3),
        }

    def handle_request(self, input_text: str, allow_llm_fallback: bool = True) -> DSLResult:
        # Step 1: Direct DSL or Layer 1 Fast Path
        cmd = self.parser.parse(input_text)
        if cmd:
            source = "direct_dsl" if cmd.raw_dsl else "nl_fast_path"
            return self.executor.execute(cmd, source_layer=source)

        # Step 1.5: Layer 1.5 Semantic Cache with Polarity Guard
        cached_result = self.cache.match(input_text)
        if cached_result:
            matched_cmd, score = cached_result
            return self.executor.execute(matched_cmd, source_layer="semantic_cache")

        # Step 2: Layer 3 LLM Fallback if enabled
        if allow_llm_fallback:
            fallback_cmd = self.translator.translate(input_text)
            if fallback_cmd:
                return self.executor.execute(fallback_cmd, source_layer="llm_fallback")

        # Step 3: Failure envelope if unrecognized
        return DSLResult(
            success=False,
            status="VALIDATION_ERROR",
            data=None,
            errors=[{"code": "NL_UNRECOGNIZED_INTENT", "message": f"Could not parse natural language query into valid DSL: '{input_text}'"}],
            meta={
                "executionTimeMs": 0.0,
                "sourceLayer": "nl_fast_path",
                "canonicalDsl": None,
                "traceId": str(uuid.uuid4())[:8]
            }
        )


# ---------------------------------------------------------------------------
# Interface Parity Adapters
# ---------------------------------------------------------------------------

class InterfaceAdapters:
    """Unified adapters for CLI Shell, Web REST API, and Model Context Protocol."""

    @staticmethod
    def cli_format(result: DSLResult, output_format: str = "text") -> str:
        """Format DSLResult for CLI display."""
        if output_format == "json":
            return json.dumps(result.to_dict(), indent=2, ensure_ascii=False)

        if not result.success:
            errs = "; ".join(e.get("message", "") for e in result.errors)
            return f"Error ({result.status}): {errs}"

        if output_format == "markdown":
            lines = [f"### Result ({result.status})"]
            if isinstance(result.data, list):
                if not result.data:
                    lines.append("*Empty list.*")
                else:
                    keys = list(result.data[0].keys()) if isinstance(result.data[0], dict) else ["Value"]
                    lines.append("| " + " | ".join(keys) + " |")
                    lines.append("| " + " | ".join(["---"] * len(keys)) + " |")
                    for row in result.data:
                        if isinstance(row, dict):
                            lines.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
                        else:
                            lines.append(f"| {row} |")
            else:
                lines.append(f"```json\n{json.dumps(result.data, indent=2, ensure_ascii=False)}\n```")
            return "\n".join(lines)

        # Default text format
        if isinstance(result.data, list):
            if not result.data:
                return "No items found."
            return json.dumps(result.data, indent=2, ensure_ascii=False)
        return str(result.data)

    @staticmethod
    def mcp_tools_manifest() -> List[Dict[str, Any]]:
        """Declare standard MCP tools for agents."""
        return [
            {
                "name": "nl_ask",
                "description": "Ask a natural language query (PL or EN); parses into canonical DSL and executes safely.",
                "inputSchema": {
                    "type": "object",
                    "required": ["query"],
                    "properties": {
                        "query": {"type": "string", "description": "Natural language query, e.g. 'pokaż otwarte zadania' or 'list tickets'"},
                        "allow_llm_fallback": {"type": "boolean", "default": True}
                    }
                }
            },
            {
                "name": "execute_dsl",
                "description": "Execute a canonical domain DSL statement directly.",
                "inputSchema": {
                    "type": "object",
                    "required": ["dsl_command"],
                    "properties": {
                        "dsl_command": {"type": "string", "description": "Canonical DSL string, e.g. 'ticket.list status=OPEN'"}
                    }
                }
            }
        ]

    @staticmethod
    def handle_mcp_tool_call(bridge: NLDSLLLMBridge, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch MCP tool execution."""
        if tool_name == "nl_ask":
            query = arguments.get("query", "")
            allow_fallback = arguments.get("allow_llm_fallback", True)
            res = bridge.handle_request(query, allow_llm_fallback=allow_fallback)
            return {"content": [{"type": "text", "text": json.dumps(res.to_dict(), indent=2)}]}
        elif tool_name == "execute_dsl":
            dsl_text = arguments.get("dsl_command", "")
            cmd = DSLCommand.from_text(dsl_text)
            res = bridge.executor.execute(cmd, source_layer="direct_dsl")
            return {"content": [{"type": "text", "text": json.dumps(res.to_dict(), indent=2)}]}
        else:
            raise ValueError(f"Unknown tool: {tool_name}")


# ---------------------------------------------------------------------------
# Self-Test Verification Suite
# ---------------------------------------------------------------------------

def run_self_test() -> int:
    """Executes normative self-tests verifying the standard implementation."""
    print("Running wellmanifest/nl-dsl-llm standard conformance self-test...")

    executor = DSLExecutor()
    # In-memory ticket store
    tickets = [
        {"id": "001", "title": "Setup repository", "status": "CLOSED"},
        {"id": "002", "title": "Add NL-DSL-LLM parser", "status": "OPEN"},
        {"id": "003", "title": "Deploy MCP server", "status": "OPEN"},
    ]

    def handle_ticket_list(cmd: DSLCommand):
        status = cmd.get_param("status")
        if status:
            return [t for t in tickets if t["status"] == status]
        return list(tickets)

    def handle_ticket_create(cmd: DSLCommand):
        new_id = f"{len(tickets) + 1:03d}"
        item = {"id": new_id, "title": cmd.arguments.get("title", "Untitled"), "status": "OPEN"}
        tickets.append(item)
        return item

    executor.register("ticket", "list", handle_ticket_list)
    executor.register("ticket", "create", handle_ticket_create)

    bridge = NLDSLLLMBridge(executor)

    # Test 1: Direct DSL
    res1 = bridge.handle_request("ticket.list status=OPEN")
    assert res1.success is True, f"Direct DSL failed: {res1}"
    assert len(res1.data) == 2
    assert res1.meta["sourceLayer"] == "direct_dsl"
    print("✓ Direct DSL execution verified")

    # Test 2: Layer 1 Polish Query
    res2 = bridge.handle_request("pokaż otwarte zadania")
    assert res2.success is True
    assert len(res2.data) == 2
    assert res2.meta["sourceLayer"] == "nl_fast_path"
    print("✓ Layer 1 Polish query ('pokaż otwarte zadania') verified")

    # Test 3: Layer 1 English Query
    res3 = bridge.handle_request("show open tickets")
    assert res3.success is True
    assert len(res3.data) == 2
    assert res3.meta["sourceLayer"] == "nl_fast_path"
    print("✓ Layer 1 English query ('show open tickets') verified")

    # Test 4: Layer 1 Creation with Title
    res4 = bridge.handle_request("dodaj zadanie 'Implement tests'")
    assert res4.success is True
    assert res4.data["title"] == "Implement tests"
    print("✓ Layer 1 Creation ('dodaj zadanie ...') verified")

    # Test 5: MCP Tool Call Parity
    mcp_call = InterfaceAdapters.handle_mcp_tool_call(bridge, "nl_ask", {"query": "wypisz tickety"})
    mcp_res = json.loads(mcp_call["content"][0]["text"])
    assert mcp_res["success"] is True
    assert len(mcp_res["data"]) == 4  # 3 original + 1 created
    print("✓ Model Context Protocol (MCP) tool call parity verified")

    # Test 6: CLI Markdown formatting
    md_output = InterfaceAdapters.cli_format(res1, output_format="markdown")
    assert "### Result (OK)" in md_output
    assert "| id | title | status |" in md_output
    print("✓ CLI formatting (markdown table) verified")

    # Test 7: Layer 1.5 Semantic Cache & Polarity Guard
    executor.register("cluster", "status", lambda cmd: {"cluster": "ready", "env": cmd.get_param("env", "prod")})
    bridge.cache.register_template(
        "stan klastra produkcyjnego",
        DSLCommand(entity="cluster", operation="status", filters={"env": "prod"}),
        slots={"env": "prod"}
    )
    # Positive cache match (Layer 1 skips, Layer 1.5 matches)
    res_cache = bridge.handle_request("stan klastra produkcyjnego")
    assert res_cache.success is True
    assert res_cache.meta["sourceLayer"] == "semantic_cache"
    assert res_cache.data["cluster"] == "ready"

    # Negated query must be rejected by polarity guard
    res_neg = bridge.handle_request("nie sprawdzaj stanu klastra produkcyjnego", allow_llm_fallback=False)
    assert res_neg.success is False
    assert res_neg.status == "VALIDATION_ERROR"
    print("✓ Layer 1.5 Semantic Cache and Polarity Guard verified")

    # Test 8: Layer 0.5 Real-time Autocomplete & Contextual Option Network
    twin = DigitalTwinContext(
        environment="prod",
        active_services=["postgres", "redis-cache"],
        degraded_services=["redis-cache"],
        active_tickets=["ticket-001", "ticket-002"],
    )
    sugg_resp = bridge.suggest_options("zre", twin_context=twin)
    assert sugg_resp["success"] is True
    assert len(sugg_resp["suggestions"]) >= 2
    # Degraded service (redis-cache) must be first
    assert sugg_resp["suggestions"][0]["bound_arguments"]["target"] == "redis-cache"
    assert sugg_resp["suggestions"][0]["badge"] == "DEGRADED"
    assert sugg_resp["suggestions"][0]["requires_confirmation"] is True
    # Sub-options DAG present
    assert len(sugg_resp["suggestions"][0]["next_options"]) >= 2
    assert "--graceful" in sugg_resp["suggestions"][0]["next_options"][0]["completion_text"]
    assert sugg_resp["latency_ms"] < 10.0
    print("✓ Layer 0.5 Real-time Autocomplete & Option Network verified")

    print("\nALL STANDARD CONFORMANCE CHECKS PASSED (8/8).")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="wellmanifest/nl-dsl-llm Reference Tool")
    parser.add_argument("--self-test", action="store_true", help="Execute built-in conformance self-test suite")
    args = parser.parse_args()

    if args.self_test:
        sys.exit(run_self_test())
    else:
        parser.print_help()
        sys.exit(0)
