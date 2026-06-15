from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.service import RiskWarningService


class ServiceLogicTest(unittest.TestCase):
    def test_model_positive_without_entity_goes_to_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = RiskWarningService(db_path=str(Path(tmp) / "risk.db"))
            service.encoder_predict = lambda text: (False, 0.12)  # type: ignore[method-assign]
            service.llm_predict = lambda text, candidates: (  # type: ignore[method-assign]
                {"has_negative": True, "entities": []},
                False,
                '{"has_negative": true, "entities": []}',
            )

            result = service.score("某平台被投诉存在逾期兑付风险", text_id="case-missing-entity", force_llm=True)

            self.assertTrue(result["has_negative"])
            self.assertTrue(result["model_has_negative"])
            self.assertTrue(result["entity_missing_review"])
            self.assertEqual(result["risk_events"], [])
            queue = service.db.review_queue()
            self.assertEqual(len(queue), 1)
            self.assertEqual(queue[0]["reason"], "entity_missing_or_filtered")
            service.db.close()


if __name__ == "__main__":
    unittest.main()
