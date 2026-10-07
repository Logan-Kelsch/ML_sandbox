"""Offline checks: python -m unittest discover -s tests -v."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from pipelines.datasets import load_dataset


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.magic = Path(self.directory.name) / "magic04.data"
        self.features = np.arange(1000, dtype=float).reshape(100, 10)
        with self.magic.open("w") as file:
            for index, row in enumerate(self.features):
                label = "g" if index % 2 else "h"
                file.write(",".join(map(str, row)) + f",{label}\n")

    def load_magic(self, **kwargs):
        return load_dataset("magic", data_path=self.magic, **kwargs)

    def test_train_only_scaling_and_raw_test_option(self):
        raw_train, raw_test, y_train, y_test, _ = self.load_magic(preprocessing=None)
        train, test, yt, yv, scaler = self.load_magic()
        np.testing.assert_allclose(scaler.mean_, raw_train.mean(axis=0))
        np.testing.assert_allclose(train.mean(axis=0), 0, atol=1e-12)
        np.testing.assert_allclose(train.std(axis=0), 1)
        np.testing.assert_allclose(test, scaler.transform(raw_test))
        np.testing.assert_array_equal(yt, y_train)
        np.testing.assert_array_equal(yv, y_test)
        _, unscaled_test, _, _, returned_scaler = self.load_magic(preprocess_test=False)
        np.testing.assert_array_equal(unscaled_test, raw_test)
        np.testing.assert_allclose(returned_scaler.transform(unscaled_test), test)

    def test_fraction_subsets_and_complements(self):
        train, test, yt, yv, scaler = self.load_magic(
            train_size=0.4, test_size=0.2, preprocessing="none"
        )
        self.assertEqual((len(train), len(test)), (40, 20))
        self.assertEqual(set(train[:, 0]) & set(test[:, 0]), set())
        self.assertEqual((yt.sum(), yv.sum()), (20, 10))
        self.assertIsNone(scaler)
        train, test, *_ = self.load_magic(train_size=0.7, test_size=None)
        self.assertEqual((len(train), len(test)), (70, 30))
        train, test, *_ = self.load_magic(train_size=None, test_size=None)
        self.assertEqual((len(train), len(test)), (80, 20))

    def test_preprocessing_choices_and_custom_clone(self):
        for style in ["standard", "minmax", "robust", "maxabs", None, "none"]:
            with self.subTest(style=style):
                train, test, *_ = self.load_magic(preprocessing=style)
                self.assertTrue(np.isfinite(train).all() and np.isfinite(test).all())
        original = StandardScaler(with_mean=False)
        *_, fitted = self.load_magic(preprocessing=original)
        self.assertIsNot(original, fitted)
        self.assertFalse(hasattr(original, "mean_"))
        self.assertFalse(fitted.with_mean)

    def test_miniboone_header_labels_and_ordered_split(self):
        path = Path(self.directory.name) / "MiniBooNE_PID.txt"
        features = np.arange(1000, dtype=float).reshape(20, 50)
        with path.open("w") as file:
            file.write("8 12\n")
            np.savetxt(file, features)
        train, test, yt, yv, _ = load_dataset(
            "MiniBooNE", data_path=path, preprocessing=None,
            shuffle=False, stratify=False,
        )
        np.testing.assert_array_equal(train, features[:16])
        np.testing.assert_array_equal(test, features[16:])
        np.testing.assert_array_equal(yt, [1] * 8 + [0] * 8)
        np.testing.assert_array_equal(yv, [0] * 4)
        path.write_text("9 12\n" + path.read_text().split("\n", 1)[1])
        with self.assertRaises(ValueError):
            load_dataset("miniboone", data_path=path)

    def test_invalid_options(self):
        for options in [
            {"train_size": 0}, {"test_size": 1}, {"test_size": float("nan")},
            {"train_size": True}, {"test_size": 20},
            {"train_size": 0.9, "test_size": 0.2},
            {"preprocessing": "invalid"}, {"shuffle": False},
        ]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.load_magic(**options)
        with self.assertRaises(ValueError):
            load_dataset("invalid")


if __name__ == "__main__":
    unittest.main()
