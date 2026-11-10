import numpy as np
import pandas as pd
import pytest

import plotters as pl


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def _grid(fn):
    g = np.linspace(-2, 2, 41)
    a, b = np.meshgrid(g, g)
    return pd.DataFrame(fn(a, b), index=g, columns=g)


def test_decision_boundary_is_the_contour_at_the_level_with_tinted_sides_and_points(rng):
    grid = _grid(lambda a, b: a + b)
    pts = pd.DataFrame(
        {"x1": rng.normal(size=30), "x2": rng.normal(size=30), "cls": ["a", "b"] * 15}
    )
    fig = pl.plot_decision_boundary(grid, pts, x="x1", y="x2", group="cls", level=0.0)
    kinds = [t.type for t in fig.data]
    assert kinds[:2] == ["heatmap", "contour"] and kinds.count("scatter") == 2
    contour = fig.data[1]
    assert (contour.contours.start, contour.contours.end) == (0.0, 0.0)
    assert set(np.asarray(fig.data[0].z).ravel()) == {0.0, 1.0}
    assert [t.name for t in fig.data if t.type == "scatter"] == ["a", "b"]


def test_decision_boundary_reference_untinted_and_validation(rng):
    grid = _grid(lambda a, b: a * b)
    fig = pl.plot_decision_boundary(
        grid, fill=False, reference=grid * 2, reference_label="Bayes", level=0.0
    )
    assert [t.type for t in fig.data] == ["contour", "contour"] and fig.data[1].name == "Bayes"
    pts = pd.DataFrame({"a": [0.0], "b": [0.0], "c": ["x"]})
    with pytest.raises(ValueError, match="x=, y= and group="):
        pl.plot_decision_boundary(grid, pts)
    many = pd.DataFrame({"a": [0.0] * 4, "b": [0.0] * 4, "c": list("wxyz")})
    with pytest.raises(ValueError, match="at most 3"):
        pl.plot_decision_boundary(grid, many, x="a", y="b", group="c")
    with pytest.raises(pl.ConventionError):
        pl.plot_decision_boundary(np.zeros((3, 3)))  # type: ignore[arg-type]


def test_coefficient_path_labels_ends_nudged_apart_and_marks_selection(rng):
    lam = np.logspace(-2, 2, 30)
    paths = pd.DataFrame(
        {f"b{i}": np.full(30, 1.0) for i in range(5)}, index=lam
    )  # all end at the same height
    fig = pl.plot_coefficient_path(paths, selected=1.0, logx=True)
    labels = [a for a in fig.layout.annotations if a.text.startswith("b")]
    ys = sorted(a.y for a in labels)
    assert len(labels) == 5 and np.min(np.diff(ys)) > 0, (
        "labels at the same height are pushed apart"
    )
    assert (
        any(sh.x0 == 1.0 and sh.xref == "x" for sh in fig.layout.shapes)
        and fig.layout.xaxis.type == "log"
    )
    legend = pl.plot_coefficient_path(paths, label_ends=False)
    assert legend.layout.showlegend is True
    with pytest.raises(ValueError, match="strictly positive"):
        pl.plot_coefficient_path(paths.set_axis(paths.index - 1000), logx=True)
    with pytest.raises(ValueError, match="numeric index"):
        pl.plot_coefficient_path(paths.set_axis(list("abcdefghijklmnopqrstuvwxyzabcd")))


def test_one_standard_error_rule_picks_the_simplest_within_one_se():
    mean = pd.Series([0.9, 0.5, 0.42, 0.40, 0.41, 0.45], index=range(1, 7))
    se = pd.Series(0.04, index=range(1, 7))
    assert pl.one_standard_error_choice(mean, se) == 3
    assert pl.one_standard_error_choice(mean, se, simpler="high") == 5
    with pytest.raises(ValueError):
        pl.one_standard_error_choice(mean, se.iloc[:3])


def test_error_curves_mark_the_minimum_or_the_one_se_choice():
    k = list(range(1, 9))
    test = pd.Series([0.9, 0.5, 0.42, 0.40, 0.41, 0.45, 0.6, 0.8], index=k)
    curves = pd.DataFrame({"train": np.linspace(0.8, 0.05, 8), "test": test}, index=k)
    se = curves * 0 + 0.05
    low = pl.plot_error_curves(curves, select="test")
    assert low.layout.shapes[0].x0 == 4 and "minimum at 4" in low.layout.title.text
    one = pl.plot_error_curves(curves, se=se, select="test", rule="1se")
    assert one.layout.shapes[0].x0 == 3 and "one-standard-error" in one.layout.title.text
    assert sum(t.fill == "toself" for t in one.data) == 2, "a ribbon per curve"
    ref = pl.plot_error_curves(curves, reference={"Bayes": 0.21})
    assert ref.layout.shapes[0].y0 == 0.21
    with pytest.raises(ValueError, match="needs se"):
        pl.plot_error_curves(curves, select="test", rule="1se")
    with pytest.raises(ValueError, match="not one of"):
        pl.plot_error_curves(curves, select="nope")
    with pytest.raises(ValueError, match="at most"):
        pl.plot_error_curves(pd.DataFrame(np.ones((3, 9))))


def test_roc_auc_known_values_and_ties():
    y = np.array([0, 0, 1, 1])
    assert pl.roc_auc(y, [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)
    assert pl.roc_auc(y, [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    assert pl.roc_auc(y, [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)
    fpr, tpr, thr = pl.roc_curve(y, [0.1, 0.4, 0.35, 0.8])
    assert (fpr[0], tpr[0], fpr[-1], tpr[-1]) == (0, 0, 1, 1) and np.isinf(thr[0])


def test_roc_auc_equals_the_rank_probability(rng):
    y = rng.integers(0, 2, 400)
    s = y * 0.8 + rng.normal(size=400)
    pos, neg = s[y == 1], s[y == 0]
    wins = (pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()
    assert pl.roc_auc(y, s) == pytest.approx(wins)


def test_average_precision_and_pr_curve_endpoints():
    y = np.array([1, 0, 1, 0])
    s = [0.9, 0.8, 0.7, 0.1]
    assert pl.average_precision(y, s) == pytest.approx(0.5 * 1.0 + 0.5 * (2 / 3))
    precision, recall, _ = pl.precision_recall_curve(y, s)
    assert recall[-1] == 1.0 and precision[0] == 1.0


def test_plot_roc_models_auc_in_legend_and_validation(rng):
    y = rng.integers(0, 2, 300)
    s = pd.DataFrame({"good": y + rng.normal(0, 0.5, 300), "bad": rng.normal(size=300)})
    fig = pl.plot_roc(y, s)
    names = [t.name for t in fig.data if t.name]
    assert names[0].startswith("good (AUC ") and float(names[0].split()[-1][:-1]) > 0.8
    assert fig.layout.yaxis.scaleanchor == "x"
    pr = pl.plot_roc(y, s["good"].rename("m"), curve="pr")
    assert "AP" in pr.data[0].name and pr.layout.shapes[0].y0 == pytest.approx(y.mean())
    with pytest.raises(ValueError, match="0 or 1"):
        pl.plot_roc(np.full(300, 2), s)
    with pytest.raises(ValueError, match="both classes"):
        pl.plot_roc(np.zeros(300), s)
    with pytest.raises(ValueError, match="curve"):
        pl.plot_roc(y, s, curve="x")  # type: ignore[arg-type]


def test_partial_dependence_recovers_the_marginal_effect_and_validates(rng):
    X = pd.DataFrame(rng.uniform(-1, 1, (300, 3)), columns=list("xyz"))
    pdp = pl.partial_dependence(
        lambda f: 3 * f["x"] + f["y"] ** 2, X, ["x", "y", "z"], grid_size=11
    )
    x = pdp["x"]
    assert (np.diff(x.to_numpy()) > 0).all() and (x.iloc[-1] - x.iloc[0]) / (
        x.index[-1] - x.index[0]
    ) == pytest.approx(3.0)
    assert np.ptp(pdp["z"].to_numpy()) < 1e-12, "an ignored feature has a flat partial dependence"
    with pytest.raises(KeyError):
        pl.partial_dependence(lambda f: f["x"], X, ["nope"])
    fig = pl.plot_partial_dependence(pdp, X, ncols=2)
    assert len(fig.layout.annotations) == 3
    assert sum(t.mode == "markers" for t in fig.data) == 3, "a rug of deciles per panel"
    with pytest.raises(ValueError):
        pl.plot_partial_dependence({})


def test_scree_shares_cumulative_and_validation():
    fig = pl.plot_scree([5.0, 3.0, 1.0, 1.0], threshold=0.9)
    assert list(fig.data[0].y) == pytest.approx([0.5, 0.3, 0.1, 0.1])
    assert fig.data[1].y[-1] == pytest.approx(1.0) and fig.layout.shapes[0].y0 == 0.9
    assert len(pl.plot_scree([5, 3, 1, 1], top=2).data[0].x) == 2
    with pytest.raises(ValueError, match="negative"):
        pl.plot_scree([1, -1])
    with pytest.raises(ValueError, match="normalize"):
        pl.plot_scree([1, 2], normalize=False)


def test_confusion_normalises_rows_by_default_and_keeps_counts_for_hover():
    truth = list("aaaabbbbcc")
    pred = list("aaabbbbbcb")
    fig = pl.plot_confusion(truth, pred)
    z = np.asarray(fig.data[0].z)
    assert z.sum(axis=1) == pytest.approx([1.0, 1.0, 1.0]) and z[0, 0] == pytest.approx(0.75)
    counts = np.asarray(fig.data[0].customdata)
    assert counts[0, 0] == 3 and counts.sum() == 10
    raw = pl.plot_confusion(truth, pred, normalize=None)
    assert np.asarray(raw.data[0].z)[0, 0] == 3
    ordered = pl.plot_confusion(truth, pred, labels=list("cba"))
    assert list(ordered.data[0].x) == list("cba")
    with pytest.raises(ValueError):
        pl.plot_confusion(truth, pred[:5])
    with pytest.raises(ValueError):
        pl.plot_confusion(truth, pred, normalize="x")  # type: ignore[arg-type]


def test_dendrogram_leaves_cut_line_and_missing_scipy_message(rng):
    pytest.importorskip("scipy")
    data = pd.DataFrame(
        np.vstack([rng.normal(0, 0.1, (5, 2)), rng.normal(5, 0.1, (5, 2))]),
        index=[f"r{i}" for i in range(10)],
    )
    fig = pl.plot_dendrogram(data, n_clusters=2)
    assert sorted(fig.layout.yaxis.ticktext) == sorted(data.index)
    cut = fig.layout.shapes[0].x0
    top = max(v for v in fig.data[0].x if v is not None)
    assert 0 < cut < top
    with pytest.raises(ValueError, match="at least two"):
        pl.plot_dendrogram(data.head(1))
    with pytest.raises(ValueError, match="between 1"):
        pl.plot_dendrogram(data, n_clusters=0)


def test_residuals_against_fitted_replaces_the_time_panel(rng):
    idx = pd.date_range("2022", periods=200)
    fitted = pd.Series(rng.normal(size=200), index=idx)
    resid = pd.Series(rng.normal(size=200), index=idx)
    fig = pl.plot_residuals(resid, fitted=fitted)
    assert fig.layout.annotations[0].text == "Against fitted" and fig.data[0].mode == "markers"
    assert pl.plot_residuals(resid).layout.annotations[0].text == "Over time"
