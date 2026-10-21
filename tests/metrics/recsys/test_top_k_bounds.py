"""Regression tests for invalid TopK recommendation counts."""

import pandas as pd
import pytest

from evidently.legacy.metrics.recsys.base_top_k import TopKMetricResult


@pytest.mark.parametrize("start_index", [0, 1])
@pytest.mark.parametrize("k", [1, 2, 3])
def test_top_k_metric_result_preserves_valid_rank_values(start_index, k):
    scores = pd.Series([0.1, 0.2, 0.3], index=range(start_index, start_index + 3))
    result = TopKMetricResult(k=k, current=scores, reference=scores.copy())

    assert result.current_value == pytest.approx(scores.iloc[k - 1])
    assert result.reference_value == pytest.approx(scores.iloc[k - 1])


@pytest.mark.parametrize("available", [0, 2])
def test_top_k_metric_result_rejects_insufficient_current_recommendations(available):
    scores = pd.Series([0.1, 0.2][:available], dtype=float)
    with pytest.raises(
        ValueError, match=rf"k=3 recommendations, but current has only {available}"
    ):
        TopKMetricResult(k=3, current=scores)


def test_top_k_metric_result_rejects_insufficient_reference_recommendations():
    current = pd.Series([0.1, 0.2, 0.3])
    reference = pd.Series([0.1, 0.2])

    with pytest.raises(ValueError, match=r"k=3 recommendations, but reference has only 2"):
        TopKMetricResult(k=3, current=current, reference=reference)


def test_top_k_metric_result_rejects_non_positive_k():
    with pytest.raises(ValueError, match="positive k"):
        TopKMetricResult(k=0, current=pd.Series([0.1, 0.2]))


def test_top_k_metric_result_rejects_missing_rank_index():
    scores = pd.Series([0.1, 0.2, 0.3], index=[0, 2, 3])
    with pytest.raises(ValueError, match="cannot find rank 1"):
        TopKMetricResult(k=2, current=scores)
