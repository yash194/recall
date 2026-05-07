"""Append-only audit log with helpers."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from recall.core.storage import Storage
from recall.types import AuditEntry


class AuditLog:
    """Thin wrapper around storage's audit_log table."""

    def __init__(self, storage: Storage):
        self._storage = storage

    def append(
        self,
        operation: str,
        target_type: str,
        target_id: str,
        actor: str = "system",
        reason: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        entry = AuditEntry(
            tenant=self._storage.tenant,
            timestamp=datetime.now(),
            operation=operation,
            actor=actor,
            target_type=target_type,
            target_id=target_id,
            payload=payload or {},
            reason=reason,
        )
        self._storage.append_audit(entry)

    def for_target(self, target_id: str) -> list[AuditEntry]:
        return self._storage.query_audit(target_id=target_id)

    def since(self, t: datetime) -> list[AuditEntry]:
        return self._storage.query_audit(since=t)

    def export_jsonl(self) -> str:
        import json

        entries = self._storage.query_audit()
        out_lines = []
        for e in entries:
            out_lines.append(
                json.dumps(
                    {
                        "seq": e.seq,
                        "tenant": e.tenant,
                        "timestamp": e.timestamp.isoformat() if e.timestamp else None,
                        "operation": e.operation,
                        "actor": e.actor,
                        "target_type": e.target_type,
                        "target_id": e.target_id,
                        "payload": e.payload,
                        "reason": e.reason,
                    },
                    default=str,
                )
            )
        return "\n".join(out_lines)
