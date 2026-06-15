#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from risk_nlp.database import RiskDatabase
from risk_nlp.schema import load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SQLite risk event store and entity profiles.")
    parser.add_argument("--risk_events_file", default="outputs/risk_events/finnsp_eval_risk_events.jsonl")
    parser.add_argument("--db_path", default="outputs/risk_system/risk_system.db")
    parser.add_argument("--output_json", default="outputs/risk_profiles/entity_profiles.json")
    parser.add_argument("--reset_db", action="store_true", help="Remove the existing SQLite demo DB before rebuilding.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_jsonl(ROOT / args.risk_events_file)
    db_path = ROOT / args.db_path
    if args.reset_db and db_path.exists():
        db_path.unlink()
    db = RiskDatabase(db_path)
    for record in records:
        for event in record.get("risk_events", []):
            db.insert_risk_event(record.get("id", ""), event)
    db.refresh_entity_profiles()
    rows = db.conn.execute("SELECT * FROM entity_profiles ORDER BY avg_risk_score DESC, event_count DESC").fetchall()
    profiles = []
    for row in rows:
        item = dict(row)
        item["risk_type_distribution"] = json.loads(item.get("risk_type_distribution") or "{}")
        item["profile_source"] = "finnsp_eval_gold_weak_events"
        profiles.append(item)
    output_path = ROOT / args.output_json
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(profiles, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(profiles)} profiles to {output_path}")
    db.close()


if __name__ == "__main__":
    main()
