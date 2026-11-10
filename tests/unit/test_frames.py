import numpy as np
import pandas as pd
import pytest

import plotters as pl
from plotters import ConventionError


def test_series_dataframe_dict_all_become_series_dicts():
    s = pd.Series([1.0, 2.0], name="a")
    assert list(pl.as_series_dict(s)) == ["a"]
    assert list(pl.as_series_dict(pd.DataFrame({"a": [1], "b": [2]}))) == ["a", "b"]
    assert list(pl.as_series_dict({"x": s})) == ["x"]


def test_unnamed_series_uses_value_label():
    assert list(pl.as_series_dict(pd.Series([1, 2]), value="ret")) == ["ret"]


def test_wrong_shape_raises_convention_error():
    with pytest.raises(ConventionError, match="convention A"):
        pl.as_series_dict([1, 2, 3])
    with pytest.raises(ConventionError):
        pl.as_series_dict({"x": [1, 2]})


def test_duplicate_columns_are_refused():
    df = pd.DataFrame([[1, 2]], columns=["a", "a"])
    with pytest.raises(ValueError, match="duplicated"):
        pl.as_series_dict(df)


def test_pick_keeps_requested_order_and_reports_missing():
    d = {k: pd.Series([1]) for k in "abc"}
    assert list(pl.pick(d, ["c", "a"])) == ["c", "a"]
    assert list(pl.pick(d)) == ["a", "b", "c"]
    with pytest.raises(KeyError, match=r"\['zz'\] not in \['a', 'b', 'c'\]"):
        pl.pick(d, ["zz"])


def test_long_to_wide_orders_numeric_columns_numerically():
    frame = pd.DataFrame({"t": [0] * 4, "k": ["126", "18", "21", "5"], "v": [1.0, 2.0, 3.0, 4.0]})
    wide = pl.long_to_wide(frame, "t", "k", "v")
    assert list(wide.columns) == ["5", "18", "21", "126"]


def test_long_to_wide_missing_column_lists_available():
    with pytest.raises(KeyError, match="not in columns"):
        pl.long_to_wide(pd.DataFrame({"a": [1]}), "a", "b", "c")


def test_group_values_orders_by_median_unless_keep():
    frame = pd.DataFrame({"g": ["hi", "hi", "lo", "lo"], "v": [10, 12, 1, 2]})
    assert list(pl.group_values(frame, "g", "v")) == ["lo", "hi"]
    assert list(pl.group_values(frame, "g", "v", keep=["hi", "lo"])) == ["hi", "lo"]


def test_normalize_within_divides_by_group_reference():
    frame = pd.DataFrame({"g": ["a", "a", "b", "b"], "v": [2.0, 4.0, 10.0, 20.0]})
    assert pl.normalize_within(frame, "v", "g").tolist() == [1.0, 2.0, 1.0, 2.0]
    assert pl.normalise_within is pl.normalize_within


def test_group_by_prefix_uses_longest_prefix():
    groups = {"ewrms": ("ewrms",), "range": ("ewrms_gk", "parkinson")}
    out = pl.group_by_prefix(["ewrms_har", "ewrms_gk", "parkinson_1"], groups)
    assert out.tolist() == ["ewrms", "range", "range"]


def test_group_by_prefix_raises_on_unmatched_unless_other():
    with pytest.raises(ValueError, match="no prefix"):
        pl.group_by_prefix(["mystery"], {"a": ("a",)})
    assert pl.group_by_prefix(["mystery"], {"a": ("a",)}, other="rest").tolist() == ["rest"]


def test_label_from_skips_nan_parts():
    frame = pd.DataFrame({"model": ["m", "n"], "param": [np.nan, "x"]})
    assert pl.label_from(frame, "model", "param").tolist() == ["m", "n x"]
