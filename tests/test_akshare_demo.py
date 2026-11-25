import os
import sys
import unittest
from datetime import date
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gp import AkShareDemo as demo


MARKET_PREFIXES = ("上证", "深证", "相对")


def make_frame(n=240):
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = 100 + np.sin(np.arange(n) / 5) * 3 + np.arange(n) * 0.05
    return pd.DataFrame({
        "日期": dates,
        "开盘": close + 0.2,
        "最高": close + 0.8,
        "最低": close - 0.8,
        "收盘": close,
        "成交量": np.arange(n, dtype=float) + 1000,
        "成交额": (np.arange(n, dtype=float) + 1000) * close,
        "换手率": np.linspace(0.1, 0.5, n),
    }).set_index("日期")


def make_market_frame(index, n):
    return pd.DataFrame({
        "上证收益_1日": np.sin(np.arange(n) / 7) * 0.01 + 0.0002,
        "深证收益_1日": np.cos(np.arange(n) / 9) * 0.012 - 0.0001,
    }, index=index)


class TestFeaturesAndLabels(unittest.TestCase):
    def test_build_features_columns_and_last_row(self):
        df = make_frame()
        features = demo.build_features(df)
        base_columns = [
            c for c in demo.FEATURE_COLUMNS
            if not c.startswith(MARKET_PREFIXES)
        ]
        self.assertTrue(set(base_columns).issubset(features.columns))
        self.assertFalse(features.isna().any().any())
        self.assertEqual(features.index[-1], df.index[-1])

    def test_build_features_with_market(self):
        df = make_frame()
        market = make_market_frame(df.index, len(df))
        features = demo.build_features(df, market)
        self.assertTrue({"上证收益_1日", "深证收益_1日", "相对上证_1日", "相对深证_1日"}.issubset(features.columns))
        self.assertFalse(features.isna().any().any())

    def test_make_labels_horizons_and_threshold(self):
        df = make_frame()
        labels_1 = demo.make_labels(df, horizon=1)
        labels_3 = demo.make_labels(df, horizon=3)
        labels_5 = demo.make_labels(df, horizon=5)
        self.assertEqual(len(labels_1), len(df) - 1)
        self.assertEqual(len(labels_3), len(df) - 3)
        self.assertEqual(len(labels_5), len(df) - 5)
        self.assertNotIn(df.index[-1], labels_5.index)

        filtered = demo.make_labels(df, threshold=0.01)
        self.assertLessEqual(len(filtered), len(labels_1))

    def test_multi_horizon_labels_display(self):
        df = make_frame()
        labels = demo.make_multi_horizon_labels(df)
        self.assertEqual(list(labels.columns), ["标签_1日", "标签_3日", "标签_5日"])
        self.assertTrue(pd.isna(labels["标签_1日"].iloc[-1]))
        self.assertTrue(labels["标签_3日"].iloc[-3:].isna().all())
        self.assertTrue(labels["标签_5日"].iloc[-5:].isna().all())
        values = pd.Series(labels.to_numpy().ravel()).dropna()
        self.assertTrue(set(values.unique()).issubset({0, 1, 2}))

    def test_multi_horizon_labels_flat_is_sideways(self):
        df = make_frame()
        df["收盘"] = 100.0
        labels = demo.make_multi_horizon_labels(df, threshold=0.0)
        self.assertTrue((labels["标签_1日"].dropna() == 2).all())

    def test_prepare_dataset_alignment(self):
        df = make_frame()
        X, y = demo.prepare_dataset(df)
        expected = [c for c in demo.FEATURE_COLUMNS if not c.startswith(MARKET_PREFIXES)]
        self.assertEqual(list(X.columns), expected)
        self.assertTrue(X.index.equals(y.index))
        self.assertNotIn(df.index[-1], X.index)

    def test_prepare_dataset_with_market(self):
        df = make_frame()
        market = make_market_frame(df.index, len(df))
        X, y = demo.prepare_dataset(df, market_df=market)
        self.assertTrue("上证收益_1日" in X.columns)
        self.assertTrue(X.index.equals(y.index))

    def test_market_data_degrades_gracefully(self):
        demo.load_market_data.clear()
        with patch.object(demo.ak, "index_zh_daily_tx", side_effect=RuntimeError("offline")):
            result = demo.load_market_data(date(2026, 1, 1), date(2026, 1, 2))
        self.assertTrue(result.empty)
        self.assertEqual(list(result.columns), ["上证收益_1日", "深证收益_1日"])


class TestEvaluation(unittest.TestCase):
    def test_evaluate_models_returns_metrics(self):
        df = make_frame()
        X, y = demo.prepare_dataset(df)
        self.assertGreaterEqual(len(X), demo.MIN_SAMPLES)
        result = demo.evaluate_models(X, y, model_option="随机森林")
        self.assertIsNotNone(result)
        self.assertEqual(result["model_name"], "随机森林")
        self.assertEqual(result["n_train"] + result["n_test"], len(X))
        self.assertBetween(result["accuracy"], 0.0, 1.0)
        self.assertEqual(result["confusion"].shape, (2, 2))

    def test_evaluate_models_auto_selects(self):
        df = make_frame()
        X, y = demo.prepare_dataset(df)
        result = demo.evaluate_models(X, y)
        self.assertIsNotNone(result)
        self.assertIn(
            result["model_name"],
            ["随机森林", "逻辑回归", "直方图梯度提升"],
        )

    def test_evaluate_models_insufficient_data(self):
        df = make_frame(n=100)
        X, y = demo.prepare_dataset(df)
        self.assertIsNone(demo.evaluate_models(X, y))

    def assertBetween(self, value, lower, upper):
        self.assertGreaterEqual(value, lower)
        self.assertLessEqual(value, upper)


if __name__ == "__main__":
    unittest.main()
