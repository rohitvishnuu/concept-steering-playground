"""
Unit tests that don't require downloading/loading a real model.
Run with: python -m pytest test_steering.py -v
"""
import json
import unittest
from pathlib import Path

DATA_PATH = Path(__file__).parent / "data" / "concept_pairs.json"


class TestConceptData(unittest.TestCase):
    def setUp(self):
        self.data = json.loads(DATA_PATH.read_text())

    def test_all_concepts_have_positive_and_negative(self):
        for concept, entry in self.data.items():
            self.assertIn("positive", entry, f"{concept} missing 'positive'")
            self.assertIn("negative", entry, f"{concept} missing 'negative'")

    def test_balanced_pair_counts(self):
        # Not strictly required for the method to work, but a big imbalance
        # would bias the mean-difference vector, so we guard against it.
        for concept, entry in self.data.items():
            pos, neg = len(entry["positive"]), len(entry["negative"])
            ratio = max(pos, neg) / min(pos, neg)
            self.assertLess(ratio, 2.0, f"{concept} has an unbalanced pair count: {pos} vs {neg}")

    def test_no_empty_strings(self):
        for concept, entry in self.data.items():
            for group in ("positive", "negative"):
                for s in entry[group]:
                    self.assertTrue(s.strip(), f"Empty example in {concept}/{group}")

    def test_no_duplicate_examples_within_concept(self):
        for concept, entry in self.data.items():
            all_examples = entry["positive"] + entry["negative"]
            self.assertEqual(
                len(all_examples), len(set(all_examples)),
                f"Duplicate example found in concept '{concept}'",
            )


if __name__ == "__main__":
    unittest.main()
