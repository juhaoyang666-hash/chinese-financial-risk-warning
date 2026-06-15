from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from risk_nlp.schema import normalize_finnsp_output, parse_model_output


class SchemaNormalizationTest(unittest.TestCase):
    def test_no_negative_variants(self):
        examples = [
            "该文本中不包含负面金融实体。",
            "以上文本不存在负面主体。",
            "分析上述文本，发现没有负面金融实体。",
        ]
        for text in examples:
            self.assertEqual(normalize_finnsp_output(text), {"has_negative": False, "entities": []})

    def test_single_entity(self):
        parsed = normalize_finnsp_output("文中包含的负面主体：北银创投")
        self.assertTrue(parsed["has_negative"])
        self.assertEqual(parsed["entities"], ["北银创投"])

    def test_multi_entity(self):
        parsed = normalize_finnsp_output("负面金融主体包含以下几个：小资钱包;资易贷;恒丰银行。")
        self.assertTrue(parsed["has_negative"])
        self.assertEqual(parsed["entities"], ["小资钱包", "资易贷", "恒丰银行"])

    def test_parse_valid_json(self):
        parsed, invalid = parse_model_output('{"has_negative": true, "entities": ["小资钱包", "资易贷"]}')
        self.assertFalse(invalid)
        self.assertTrue(parsed["has_negative"])
        self.assertEqual(parsed["entities"], ["小资钱包", "资易贷"])

    def test_parse_fallback_text(self):
        parsed, invalid = parse_model_output("负面金融主体：钱宝网")
        self.assertTrue(invalid)
        self.assertTrue(parsed["has_negative"])
        self.assertEqual(parsed["entities"], ["钱宝网"])


if __name__ == "__main__":
    unittest.main()

