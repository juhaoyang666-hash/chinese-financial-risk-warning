from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS risk_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text_id TEXT,
    entity TEXT NOT NULL,
    risk_type TEXT,
    severity TEXT,
    risk_score REAL,
    risk_level TEXT,
    evidence TEXT,
    source_type TEXT,
    action TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS entity_profiles (
    entity TEXT PRIMARY KEY,
    event_count INTEGER,
    max_risk_level TEXT,
    avg_risk_score REAL,
    risk_type_distribution TEXT,
    latest_evidence TEXT,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS review_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text_id TEXT,
    entity TEXT,
    risk_score REAL,
    risk_level TEXT,
    reason TEXT,
    status TEXT DEFAULT 'pending',
    payload TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS prediction_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT,
    text TEXT,
    stage TEXT,
    latency_sec REAL,
    result_json TEXT,
    invalid_json INTEGER DEFAULT 0,
    hallucinated_entity INTEGER DEFAULT 0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS drift_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    positive_rate REAL,
    avg_text_length REAL,
    avg_entity_count REAL,
    hallucination_rate REAL,
    invalid_json_rate REAL,
    risk_level_distribution TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

LEVEL_RANK = {"low": 1, "medium": 2, "high": 3, "critical": 4}


class RiskDatabase:
    def __init__(self, path: str | Path = "outputs/risk_system/risk_system.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.init_schema()

    def close(self) -> None:
        self.conn.close()

    def init_schema(self) -> None:
        self.conn.executescript(SCHEMA_SQL)
        self.conn.commit()

    def risk_event_exists(self, text_id: str, event: dict[str, Any]) -> bool:
        row = self.conn.execute(
            """
            SELECT 1 FROM risk_events
            WHERE COALESCE(text_id, '') = COALESCE(?, '')
              AND COALESCE(entity, '') = COALESCE(?, '')
              AND COALESCE(risk_type, '') = COALESCE(?, '')
              AND COALESCE(evidence, '') = COALESCE(?, '')
            LIMIT 1
            """,
            (text_id, event.get("entity"), event.get("risk_type"), event.get("evidence")),
        ).fetchone()
        return row is not None

    def insert_risk_event(self, text_id: str, event: dict[str, Any]) -> bool:
        if self.risk_event_exists(text_id, event):
            return False
        self.conn.execute(
            """
            INSERT INTO risk_events
            (text_id, entity, risk_type, severity, risk_score, risk_level, evidence, source_type, action)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                text_id,
                event.get("entity"),
                event.get("risk_type"),
                event.get("severity"),
                event.get("risk_score"),
                event.get("risk_level"),
                event.get("evidence"),
                event.get("source_type"),
                event.get("action"),
            ),
        )
        if event.get("risk_level") in {"high", "critical"} or event.get("action") in {"人工复核", "低置信复核"}:
            self.enqueue_review(text_id, event, reason=event.get("action", "人工复核"))
        self.conn.commit()
        return True

    def enqueue_review(self, text_id: str, event: dict[str, Any], reason: str) -> None:
        self.enqueue_review_case(
            text_id=text_id,
            entity=event.get("entity"),
            risk_score=event.get("risk_score"),
            risk_level=event.get("risk_level"),
            reason=reason,
            payload=event,
            commit=False,
        )

    def enqueue_review_case(
        self,
        *,
        text_id: str,
        reason: str,
        payload: dict[str, Any],
        entity: str | None = None,
        risk_score: float | None = None,
        risk_level: str | None = None,
        commit: bool = True,
    ) -> bool:
        existing = self.conn.execute(
            """
            SELECT 1 FROM review_queue
            WHERE COALESCE(text_id, '') = COALESCE(?, '')
              AND COALESCE(entity, '') = COALESCE(?, '')
              AND COALESCE(reason, '') = COALESCE(?, '')
            LIMIT 1
            """,
            (text_id, entity, reason),
        ).fetchone()
        if existing:
            return False
        self.conn.execute(
            """
            INSERT INTO review_queue (text_id, entity, risk_score, risk_level, reason, payload)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                text_id,
                entity,
                risk_score,
                risk_level,
                reason,
                json.dumps(payload, ensure_ascii=False),
            ),
        )
        if commit:
            self.conn.commit()
        return True

    def log_prediction(
        self,
        request_id: str,
        text: str,
        stage: str,
        latency_sec: float,
        result: dict[str, Any],
        invalid_json: bool = False,
        hallucinated_entity: bool = False,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO prediction_logs
            (request_id, text, stage, latency_sec, result_json, invalid_json, hallucinated_entity)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request_id,
                text,
                stage,
                latency_sec,
                json.dumps(result, ensure_ascii=False),
                int(invalid_json),
                int(hallucinated_entity),
            ),
        )
        self.conn.commit()

    def refresh_entity_profiles(self) -> None:
        rows = self.conn.execute("SELECT * FROM risk_events ORDER BY created_at ASC, id ASC").fetchall()
        by_entity: dict[str, list[sqlite3.Row]] = {}
        for row in rows:
            by_entity.setdefault(row["entity"], []).append(row)
        for entity, entity_rows in by_entity.items():
            scores = [float(row["risk_score"] or 0.0) for row in entity_rows]
            max_level = max((row["risk_level"] for row in entity_rows), key=lambda item: LEVEL_RANK.get(item, 0))
            dist: dict[str, int] = {}
            for row in entity_rows:
                risk_type = row["risk_type"] or "unknown"
                dist[risk_type] = dist.get(risk_type, 0) + 1
            latest = entity_rows[-1]["evidence"] or ""
            self.conn.execute(
                """
                INSERT INTO entity_profiles
                (entity, event_count, max_risk_level, avg_risk_score, risk_type_distribution, latest_evidence, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(entity) DO UPDATE SET
                    event_count=excluded.event_count,
                    max_risk_level=excluded.max_risk_level,
                    avg_risk_score=excluded.avg_risk_score,
                    risk_type_distribution=excluded.risk_type_distribution,
                    latest_evidence=excluded.latest_evidence,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    entity,
                    len(entity_rows),
                    max_level,
                    sum(scores) / len(scores) if scores else 0.0,
                    json.dumps(dist, ensure_ascii=False),
                    latest,
                ),
            )
        self.conn.commit()

    def get_entity_profile(self, entity: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM entity_profiles WHERE entity = ?", (entity,)).fetchone()
        if not row:
            return None
        data = dict(row)
        data["risk_type_distribution"] = json.loads(data.get("risk_type_distribution") or "{}")
        return data

    def review_queue(self, status: str = "pending", limit: int = 50) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM review_queue WHERE status = ? ORDER BY risk_score DESC, id DESC LIMIT ?",
            (status, limit),
        ).fetchall()
        return [dict(row) for row in rows]

    def metrics(self) -> dict[str, Any]:
        event_count = self.conn.execute("SELECT COUNT(*) AS n FROM risk_events").fetchone()["n"]
        profile_count = self.conn.execute("SELECT COUNT(*) AS n FROM entity_profiles").fetchone()["n"]
        pending_count = self.conn.execute(
            "SELECT COUNT(*) AS n FROM review_queue WHERE status = 'pending'"
        ).fetchone()["n"]
        avg_score = self.conn.execute("SELECT AVG(risk_score) AS v FROM risk_events").fetchone()["v"] or 0.0
        return {
            "risk_event_count": event_count,
            "entity_profile_count": profile_count,
            "pending_review_count": pending_count,
            "avg_risk_score": float(avg_score),
        }

    def insert_drift_snapshot(self, name: str, snapshot: dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT INTO drift_snapshots
            (name, positive_rate, avg_text_length, avg_entity_count, hallucination_rate, invalid_json_rate, risk_level_distribution)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                name,
                snapshot.get("positive_rate"),
                snapshot.get("avg_text_length"),
                snapshot.get("avg_entity_count"),
                snapshot.get("hallucination_rate"),
                snapshot.get("invalid_json_rate"),
                json.dumps(snapshot.get("risk_level_distribution", {}), ensure_ascii=False),
            ),
        )
        self.conn.commit()
