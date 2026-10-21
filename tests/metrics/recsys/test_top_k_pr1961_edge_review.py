"""Additional edge-case review of upstream TopK PR #1961.

The PR's new helper is used unchanged in this fork-only integration branch.
Both expected failures are currently known and intentionally marked xfail.
"""

import pandas as pd
import pytest

from evidently.legacy.metrics.recsys.base_top_k import _value_at_k


@pytest.mark.xfail(strict=True, reason="PR #1961 evaluates min(empty_series.index) before validation")
def test_empty_top_k_scores_produce_actionable_error():
    with pytest.raises(ValueError, match=r"only recommendations through"):
        _value_at_k(pd.Series(dtype=float), k=3, data_name="current data")


@pytest.mark.xfail(strict=True, reason="PR #1961 accepts k=0 for list-valued score results")
def test_top_k_rank_must_be_positive_for_list_values():
    with pytest.raises(ValueError, match=r"positive|greater than zero"):
        _value_at_k([0.25, 0.5], k=0, data_name="current data")
