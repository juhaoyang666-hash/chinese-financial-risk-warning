from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from risk_nlp.candidates import extract_candidate_entities, filter_entities_by_candidates
from risk_nlp.database import RiskDatabase
from risk_nlp.risk_events import build_risk_events
from risk_nlp.risk_types import infer_risk_type
from risk_nlp.scoring import RiskScoreInput, compute_risk_score, risk_level


class RiskSystemTest(unittest.TestCase):
    def test_risk_type_keywords(self):
        match = infer_risk_type("钱宝网涉嫌非法吸收公众存款被立案")
        self.assertEqual(match.risk_type, "诈骗/非法集资")
        self.assertEqual(match.severity, "critical")

    def test_score_range_and_level(self):
        score = compute_risk_score(RiskScoreInput(model_confidence=0.9, severity="critical", source_type="regulator"))
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)
        self.assertIn(risk_level(score), {"high", "critical"})

    def test_candidate_filter(self):
        text = "小资钱包逾期，恒丰银行表示关注。 实体：小资钱包;恒丰银行"
        candidates = extract_candidate_entities(text)
        filtered = filter_entities_by_candidates(["小资钱包", "不存在主体"], text, candidates)
        self.assertEqual(filtered, ["小资钱包"])

    def test_candidate_filter_without_explicit_hint_allows_in_text_entity(self):
        text = "深圳市公安局通报绿化贷涉嫌非法吸收公众存款案件进展"
        filtered = filter_entities_by_candidates(["绿化贷"], text, ["深圳市公安局通报绿化贷"])
        self.assertEqual(filtered, ["绿化贷"])

    def test_candidate_filter_with_explicit_hint_is_strict(self):
        text = "绿化贷涉嫌非法吸收公众存款。 实体：壹佰金融"
        candidates = extract_candidate_entities(text)
        filtered = filter_entities_by_candidates(["绿化贷"], text, candidates)
        self.assertEqual(filtered, [])

    def test_build_risk_events(self):
        events = build_risk_events("小资钱包涉嫌诈骗且无法兑付", ["小资钱包"], confidence=0.93)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["entity"], "小资钱包")
        self.assertIn(events[0]["risk_level"], {"high", "critical"})

    def test_sqlite_write_and_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = RiskDatabase(Path(tmp) / "risk.db")
            event = build_risk_events("小资钱包涉嫌诈骗且无法兑付", ["小资钱包"], confidence=0.93)[0]
            db.insert_risk_event("case-1", event)
            db.insert_risk_event("case-1", event)
            db.refresh_entity_profiles()
            profile = db.get_entity_profile("小资钱包")
            self.assertIsNotNone(profile)
            self.assertEqual(profile["event_count"], 1)
            self.assertEqual(len(db.review_queue()), 1)
            self.assertEqual(db.conn.execute("SELECT COUNT(*) FROM risk_events").fetchone()[0], 1)
            db.close()


if __name__ == "__main__":
    unittest.main()
