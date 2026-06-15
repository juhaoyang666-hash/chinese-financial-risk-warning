from __future__ import annotations

import unittest


try:
    from app.main import metrics
except Exception:  # pragma: no cover
    metrics = None


class ApiSmokeTest(unittest.TestCase):
    @unittest.skipIf(metrics is None, "fastapi app is not importable")
    def test_metrics_endpoint(self):
        response = metrics()
        self.assertIn("risk_event_count", response)


if __name__ == "__main__":
    unittest.main()
