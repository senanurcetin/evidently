import numpy as np
import pandas as pd
import pytest

from evidently.legacy.metrics import HitRateKMetric
from evidently.legacy.metrics.recsys.hit_rate_k import HitRateKMetricResult
from evidently.legacy.pipeline.column_mapping import ColumnMapping
from evidently.legacy.pipeline.column_mapping import RecomType
from evidently.legacy.report import Report


def test_hit_rate_value():
    current = pd.DataFrame(
        data=dict(
            user_id=["a", "a", "a", "b", "b", "b", "c", "c", "c"],
            prediction=[1, 2, 3, 1, 2, 3, 1, 2, 3],
            target=[1, 0, 0, 0, 0, 0, 0, 0, 1],
        ),
    )

    metric = HitRateKMetric(k=2)
    report = Report(metrics=[metric])
    column_mapping = ColumnMapping(recommendations_type=RecomType.RANK)
    report.run(reference_data=None, current_data=current, column_mapping=column_mapping)

    results = metric.get_result()
    assert len(results.current) == 3
    assert np.isclose(results.current[0], 0.5)
    assert np.isclose(results.current[1], 0.5)
    assert np.isclose(results.current[2], 1)


def test_hit_rate_value_judged_only():
    current = pd.DataFrame(
        data=dict(
            user_id=["a", "a", "a", "b", "b", "b", "c", "c", "c"],
            prediction=[1, 2, 3, 1, 2, 3, 1, 2, 3],
            target=[1, 0, 0, 0, 0, 0, 0, 0, 1],
        ),
    )

    metric = HitRateKMetric(k=3, no_feedback_users=True)
    report = Report(metrics=[metric])
    column_mapping = ColumnMapping(recommendations_type=RecomType.RANK)
    report.run(reference_data=None, current_data=current, column_mapping=column_mapping)

    results = metric.get_result()
    assert len(results.current) == 3
    assert np.isclose(results.current[0], 0.3333333)
    assert np.isclose(results.current[1], 0.3333333)
    assert np.isclose(results.current[2], 0.6666666)


def test_hit_rate_value_judged_only_scores():
    current = pd.DataFrame(
        data=dict(
            user_id=["a", "a", "a", "b", "b", "b", "c", "c", "c"],
            prediction=[1.25, 1.0, 0.3, 0.9, 0.8, 0.7, 1.0, 0.5, 0.3],
            target=[1, 0, 0, 0, 0, 0, 0, 0, 1],
        ),
    )

    metric = HitRateKMetric(k=3, no_feedback_users=True)
    report = Report(metrics=[metric])
    column_mapping = ColumnMapping()
    report.run(reference_data=None, current_data=current, column_mapping=column_mapping)

    results = metric.get_result()
    assert len(results.current) == 3
    assert np.isclose(results.current[0], 0.3333333)
    assert np.isclose(results.current[1], 0.3333333)
    assert np.isclose(results.current[2], 0.6666666)


@pytest.mark.parametrize("k", [4, 10])
def test_hit_rate_rejects_rank_above_available_recommendations(k):
    current = pd.DataFrame(
        {
            "user_id": ["a", "a", "a", "b", "b", "b"],
            "prediction": [1, 2, 3, 1, 2, 3],
            "target": [1, 0, 0, 0, 1, 0],
        }
    )
    metric = HitRateKMetric(k=k)
    report = Report(metrics=[metric])
    mapping = ColumnMapping(recommendations_type=RecomType.RANK)
    report.run(reference_data=None, current_data=current, column_mapping=mapping)

    with pytest.raises(ValueError, match=rf"top-{k}.*current.*top-3"):
        metric.get_result()


def test_hit_rate_rejects_reference_rank_above_available_recommendations():
    current = pd.DataFrame(
        {
            "user_id": ["a", "a", "a", "b", "b", "b"],
            "prediction": [1, 2, 3, 1, 2, 3],
            "target": [1, 0, 0, 0, 1, 0],
        }
    )
    reference = pd.DataFrame(
        {
            "user_id": ["a", "a", "b", "b"],
            "prediction": [1, 2, 1, 2],
            "target": [1, 0, 0, 1],
        }
    )
    metric = HitRateKMetric(k=3)
    report = Report(metrics=[metric])
    mapping = ColumnMapping(recommendations_type=RecomType.RANK)
    report.run(reference_data=reference, current_data=current, column_mapping=mapping)

    with pytest.raises(ValueError, match=r"top-3.*reference.*top-2"):
        metric.get_result()


def test_hit_rate_scalar_values_use_position_after_index_round_trip():
    # Index labels may be strings after JSON serialization of a pandas Series.
    result = HitRateKMetricResult(
        k=3,
        current=pd.Series([0.25, 0.5, 0.75], index=["0", "1", "2"]),
        reference=pd.Series([0.2, 0.4, 0.6], index=["0", "1", "2"]),
    )
    assert result.current_value == pytest.approx(0.75)
    assert result.reference_value == pytest.approx(0.6)


def test_hit_rate_scalar_value_without_reference():
    result = HitRateKMetricResult(k=2, current=pd.Series([0.1, 0.9]))
    assert result.current_value == pytest.approx(0.9)
    assert result.reference_value is None


@pytest.mark.parametrize("k", [0, -1])
def test_hit_rate_rejects_nonpositive_k(k):
    current = pd.DataFrame(
        {
            "user_id": ["a", "a", "a"],
            "prediction": [1, 2, 3],
            "target": [1, 0, 0],
        }
    )
    metric = HitRateKMetric(k=k)
    report = Report(metrics=[metric])
    mapping = ColumnMapping(recommendations_type=RecomType.RANK)
    report.run(reference_data=None, current_data=current, column_mapping=mapping)

    with pytest.raises(ValueError, match="positive integer"):
        metric.get_result()
