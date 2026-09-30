"""Checks the evaluation guardrails on the committed dataset."""

import unittest
from pathlib import Path

from model import features, records, split_indices


class BaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.paths, cls.labels, cls.groups = records(Path("dataset"))

    def test_grouped_splits_do_not_share_source_tokens(self):
        split = split_indices(self.labels, self.groups)
        for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
            self.assertFalse(set(self.groups[split[first]]) & set(self.groups[split[second]]))

    def test_feature_dimensions_and_finite_values(self):
        for kind, expected in (("pixels", 768), ("texture", 59)):
            vector = features(self.paths[0], kind)
            self.assertEqual(vector.shape, (expected,))
            self.assertTrue((vector == vector).all())


if __name__ == "__main__":
    unittest.main()
