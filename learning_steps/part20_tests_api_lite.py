from __future__ import annotations

import unittest

import pandas as pd

from part17_api_lite import ARTIFACT_DIR, PhishingEngine


DATASET = r"C:\Users\PRIYAMVADA NAMBIAR\Downloads\output.csv"


class TestPhishingApiLite(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = PhishingEngine(ARTIFACT_DIR)
        df = pd.read_csv(DATASET)
        cls.sample = df.drop(columns=["Result"]).iloc[0].to_dict()

    def test_predict_ensemble_success(self) -> None:
        out = self.engine.predict_one(self.sample, mode="ensemble")
        self.assertIn("prediction_label", out)
        self.assertIn("prob_positive_class", out)
        self.assertIn("confidence", out)
        self.assertIn("decision_type", out)
        self.assertEqual(out["mode"], "ensemble")

    def test_predict_hybrid_success(self) -> None:
        out = self.engine.predict_one(self.sample, mode="hybrid")
        self.assertEqual(out["mode"], "hybrid")
        self.assertIn(out["prediction_text"], {"Phishing", "Legitimate"})

    def test_missing_feature_rejected(self) -> None:
        bad = dict(self.sample)
        bad.pop("URL_Length")
        with self.assertRaises(ValueError):
            self.engine.predict_one(bad, mode="ensemble")

    def test_invalid_mode_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.engine.predict_one(self.sample, mode="bad_mode")

    def test_non_numeric_rejected(self) -> None:
        bad = dict(self.sample)
        bad["URL_Length"] = "abc"
        with self.assertRaises(ValueError):
            self.engine.predict_one(bad, mode="ensemble")


if __name__ == "__main__":
    unittest.main()
