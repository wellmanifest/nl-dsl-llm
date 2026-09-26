"""
Procedure and Action Catalog Registry for paxlet-com/nl-dsl-llm.
Maps semantic action_id to executable procedure URIs across clients (Taskand, Clonerd, nl-dsl-sh).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ActionDef:
    action_id: str
    title: str
    description: str
    aliases_pl: List[str] = field(default_factory=list)
    aliases_en: List[str] = field(default_factory=list)
    uri_patterns: Dict[str, str] = field(default_factory=dict)  # client_id -> URI
    handler: Optional[Callable[[Dict[str, Any]], Any]] = None


class ProcedureRegistry:
    """Central catalog of known actions and client-specific URI mappings."""

    def __init__(self):
        self._actions: Dict[str, ActionDef] = {}
        self._register_default_catalog()

    def register(self, action: ActionDef) -> None:
        self._actions[action.action_id] = action

    def get(self, action_id: str) -> Optional[ActionDef]:
        return self._actions.get(action_id)

    def list_actions(self) -> List[ActionDef]:
        return list(self._actions.values())

    def resolve_uri(self, action_id: str, client_id: str, default: str = "") -> str:
        act = self.get(action_id)
        if not act:
            return default
        return act.uri_patterns.get(client_id, act.uri_patterns.get("default", default))

    def import_openapi_schema(self, schema_dict: Dict[str, Any], client_id: str = "openapi") -> int:
        """
        Federated schema discovery: imports OpenAPI v3 path operations into ProcedureRegistry.
        """
        paths = schema_dict.get("paths", {})
        count = 0
        for path, methods in paths.items():
            if not isinstance(methods, dict):
                continue
            for method, op_def in methods.items():
                if method.lower() not in ("get", "post", "put", "delete", "patch") or not isinstance(op_def, dict):
                    continue
                op_id = op_def.get("operationId") or f"{method.lower()}_{path.replace('/', '_').strip('_')}"
                summary = op_def.get("summary") or op_id
                action_id = f"api.{op_id}"
                self.register(ActionDef(
                    action_id=action_id,
                    title=summary,
                    description=op_def.get("description", summary),
                    aliases_pl=[summary.lower()],
                    aliases_en=[op_id.replace("_", " ")],
                    uri_patterns={client_id: f"openapi://{path}#{method.upper()}", "default": f"proc://api/{op_id}/v1"}
                ))
                count += 1
        return count

    def _register_default_catalog(self) -> None:
        # 1. CPU Monitor
        self.register(ActionDef(
            action_id="monitor.cpu_status",
            title="CPU Usage Status",
            description="Check CPU core usage and load average",
            aliases_pl=["pokaż procesor", "status cpu", "użycie procesora", "obciążenie cpu"],
            aliases_en=["cpu status", "show cpu", "cpu usage", "check load"],
            uri_patterns={
                "taskand": "proc://taskand.dev/monitor/cpu/v1",
                "nl-dsl-sh": "urn:nl-dsl-sh:sys:cpu-check",
                "clonerd": "cluster.node.cpu",
                "default": "proc://paxlet.com/monitor/cpu/v1"
            }
        ))

        # 2. Memory / RAM Monitor
        self.register(ActionDef(
            action_id="monitor.memory_status",
            title="Memory Status",
            description="Check RAM allocation, free memory and swap usage",
            aliases_pl=["status ram", "pokaż pamięć", "użycie ram", "ile wolnego ramu"],
            aliases_en=["ram status", "memory usage", "free memory", "check swap"],
            uri_patterns={
                "taskand": "proc://taskand.dev/monitor/memory/v1",
                "nl-dsl-sh": "urn:nl-dsl-sh:sys:ram-check",
                "clonerd": "cluster.node.memory",
                "default": "proc://paxlet.com/monitor/memory/v1"
            }
        ))

        # 3. Service Restart
        self.register(ActionDef(
            action_id="service.restart",
            title="Restart Service",
            description="Restart a designated daemon or container service",
            aliases_pl=["zrestartuj usługę", "restartuj serwis", "zresetuj kontener"],
            aliases_en=["restart service", "reboot daemon", "restart container"],
            uri_patterns={
                "taskand": "proc://taskand.dev/service/restart/v1",
                "nl-dsl-sh": "urn:nl-dsl-sh:service:restart",
                "clonerd": "cluster.service.restart",
                "default": "proc://paxlet.com/service/restart/v1"
            }
        ))

        # 4. Task / Ticket List
        self.register(ActionDef(
            action_id="task.list",
            title="List Tasks",
            description="Query and filter tasks by status or assigned workstream",
            aliases_pl=["pokaż zadania", "wypisz zadania", "lista ticketów", "pokaż tickety"],
            aliases_en=["list tasks", "show tickets", "get tasks", "query tickets"],
            uri_patterns={
                "taskand": "proc://taskand.dev/task/list/v1",
                "nl-dsl-sh": "urn:nl-dsl-sh:task:list",
                "clonerd": "cluster.task.list",
                "default": "proc://paxlet.com/task/list/v1"
            }
        ))

        # 5. Clean Cache
        self.register(ActionDef(
            action_id="cache.cleanup",
            title="Clean Cache",
            description="Purge temporary files and cache stores",
            aliases_pl=["wyczyść cache", "usuń pamięć podręczną", "wyczyść pamięć podręczną"],
            aliases_en=["clear cache", "clean cache", "purge cache", "flush cache"],
            uri_patterns={
                "taskand": "proc://taskand.dev/cache/clean/v1",
                "nl-dsl-sh": "urn:nl-dsl-sh:cache:purge",
                "clonerd": "cluster.cache.clean",
                "default": "proc://paxlet.com/cache/clean/v1"
            }
        ))
