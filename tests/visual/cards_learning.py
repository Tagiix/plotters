"""Gallery cards for the statistical-learning plots (decision boundaries, paths, error curves ...)."""

import numpy as np
import pandas as pd

import plotters as pl

from .registry import card


def _mixture(n=100, seed=60):
    """Two classes, each a mixture of 10 Gaussian blobs (the structure of ESL's simulated example)."""
    rng = np.random.default_rng(seed)
    centres = {0: rng.normal([1.0, 0.0], 1.0, (10, 2)), 1: rng.normal([0.0, 1.0], 1.0, (10, 2))}
    rows = []
    for cls, c in centres.items():
        pick = rng.integers(0, 10, n)
        pts = c[pick] + rng.normal(0, 0.45, (n, 2))
        rows += [{"x1": a, "x2": b, "class": f"class {cls}"} for a, b in pts]
    return pd.DataFrame(rows), centres


def _bayes_prob(grid_pts, centres, sd=0.45):
    def density(c):
        d2 = ((grid_pts[:, None, :] - c[None, :, :]) ** 2).sum(-1)
        return np.exp(-d2 / (2 * sd**2)).mean(1)

    p0, p1 = density(centres[0]), density(centres[1])
    return p1 / (p0 + p1)


def _knn_grid(train, k, lim=(-3, 4), size=90):
    g = np.linspace(*lim, size)
    a, b = np.meshgrid(g, g)
    pts = np.c_[a.ravel(), b.ravel()]
    xy = train[["x1", "x2"]].to_numpy()
    y = (train["class"] == "class 1").to_numpy().astype(float)
    d = ((pts[:, None, :] - xy[None, :, :]) ** 2).sum(-1)
    nearest = np.argpartition(d, k, axis=1)[:, :k]
    return pd.DataFrame(y[nearest].mean(1).reshape(a.shape), index=g, columns=g), pts, g


@card(
    "Learning",
    "plot_decision_boundary",
    [
        "Solid line: the 15-nearest-neighbour boundary at 0.5; dashed: the Bayes-optimal boundary. The two sides are tinted lightly.",
        "Training points are open circles in their class colour, so the tint never hides them.",
    ],
)
def ln_boundary():
    train, centres = _mixture()
    grid, pts, g = _knn_grid(train, 15)
    bayes = pd.DataFrame(_bayes_prob(pts, centres).reshape(grid.shape), index=g, columns=g)
    return pl.plot_decision_boundary(
        grid,
        train,
        "15-nearest neighbours",
        "dashed = Bayes optimal",
        x="x1",
        y="x2",
        group="class",
        reference=bayes,
        reference_label="Bayes",
    )


@card(
    "Learning",
    "plot_decision_boundary: 1-NN",
    [
        "1-nearest-neighbour: a ragged boundary that follows every training point (zero training error, high variance).",
    ],
)
def ln_boundary_1nn():
    train, _ = _mixture()
    grid, _, _ = _knn_grid(train, 1)
    return pl.plot_decision_boundary(
        grid, train, "1-nearest neighbour", x="x1", y="x2", group="class"
    )


@card(
    "Learning",
    "plot_coefficient_path",
    [
        "Ridge profiles against effective degrees of freedom; names at the right ends, nudged apart so none overlap.",
        "Dotted vertical line is the chosen penalty.",
    ],
)
def ln_path():
    rng = np.random.default_rng(61)
    n, p = 97, 8
    base = rng.normal(size=(n, 1))
    x = 0.6 * base + rng.normal(size=(n, p))
    x = (x - x.mean(0)) / x.std(0, ddof=1)
    beta = np.array([0.7, 0.3, -0.15, 0.15, -0.1, 0.0, 0.1, -0.05])
    y = x @ beta + rng.normal(0, 0.6, n)
    u, d, vt = np.linalg.svd(x, full_matrices=False)
    lam = np.logspace(-2, 4, 60)
    rows, dfs = [], []
    for lv in lam:
        coef = vt.T @ ((d / (d**2 + lv)) * (u.T @ (y - y.mean())))
        rows.append(coef)
        dfs.append(np.sum(d**2 / (d**2 + lv)))
    names = ["lcavol", "lweight", "age", "lbph", "svi", "lcp", "gleason", "pgg45"]
    paths = pd.DataFrame(rows, index=dfs, columns=names).sort_index()
    return pl.plot_coefficient_path(
        paths, "Ridge coefficient profiles", xlabel="effective degrees of freedom", selected=5.0
    )


@card(
    "Learning",
    "plot_error_curves",
    [
        "Training error keeps falling while test error is U-shaped; the dotted line is the Bayes error.",
        "Ring marks the minimum test error.",
    ],
)
def ln_errors():
    k = np.arange(1, 16)
    test = 0.21 + 0.012 * (k - 6) ** 2 / 4 + 0.03
    curves = pd.DataFrame({"train": 0.30 - 0.019 * k, "test": test}, index=k)
    return pl.plot_error_curves(
        curves,
        "Error against complexity",
        reference={"Bayes error": 0.21},
        select="test",
        xlabel="model complexity (degrees of freedom)",
    )


@card(
    "Learning",
    "plot_error_curves: one-standard-error rule",
    [
        "Cross-validated error with standard-error ribbon: the hollow ring is the minimum, the filled ring the simplest model within one standard error of it.",
    ],
)
def ln_errors_1se():
    k = np.arange(0, 9)
    cv = np.array([1.00, 0.72, 0.60, 0.56, 0.545, 0.54, 0.535, 0.537, 0.54])
    curves = pd.DataFrame({"CV error": cv}, index=k)
    se = pd.DataFrame({"CV error": np.full(9, 0.04)}, index=k)
    return pl.plot_error_curves(
        curves, "Subset size", se=se, select="CV error", rule="1se", xlabel="subset size"
    )


@card(
    "Learning",
    "plot_roc",
    [
        "Two models on the same data with the area under each curve in the legend; dotted diagonal is chance.",
    ],
)
def ln_roc():
    rng = np.random.default_rng(62)
    y = rng.integers(0, 2, 1200)
    return pl.plot_roc(
        y,
        pd.DataFrame(
            {"model A": y * 1.6 + rng.normal(size=1200), "model B": y * 0.9 + rng.normal(size=1200)}
        ),
        "ROC curves",
    )


@card(
    "Learning",
    "plot_roc: precision-recall",
    [
        "Precision-recall on imbalanced classes (8% positive); dotted line is the positive rate, the no-skill baseline.",
    ],
)
def ln_pr():
    rng = np.random.default_rng(63)
    y = (rng.uniform(size=3000) < 0.08).astype(int)
    return pl.plot_roc(
        y,
        pd.Series(y * 1.5 + rng.normal(size=3000), name="model"),
        "Precision and recall",
        curve="pr",
    )


@card(
    "Learning",
    "plot_partial_dependence",
    [
        "One panel per feature on a shared y axis; ticks along the bottom are the deciles of the data, so the sparse ends are not over-read.",
        "z is ignored by the model: a flat line.",
    ],
)
def ln_pdp():
    rng = np.random.default_rng(64)
    X = pd.DataFrame(
        {
            "x": rng.normal(size=500),
            "y": rng.uniform(-2, 2, 500),
            "z": rng.normal(size=500),
            "w": rng.uniform(0, 3, 500),
        }
    )
    curves = pl.partial_dependence(
        lambda f: 2 * np.sin(f["x"]) + 0.5 * f["y"] ** 2 + np.log1p(f["w"]), X, ["x", "y", "z", "w"]
    )
    return pl.plot_partial_dependence(curves, X, "Partial dependence", ncols=2)


@card(
    "Learning",
    "plot_scree",
    [
        "Bars are each component's share of the variance; the line is the running total, with a dotted 90% mark.",
    ],
)
def ln_scree():
    return pl.plot_scree(
        [4.1, 2.2, 1.1, 0.7, 0.45, 0.3, 0.2, 0.15, 0.1], "Principal components", threshold=0.9
    )


@card(
    "Learning",
    "plot_confusion",
    [
        "Rows are the actual class shown as shares (the diagonal is each class's recall); hover shows the counts.",
    ],
)
def ln_confusion():
    rng = np.random.default_rng(65)
    classes = np.array(["setosa", "versicolor", "virginica"])
    truth = rng.choice(classes, 300, p=[0.3, 0.4, 0.3])
    pred = truth.copy()
    flip = rng.uniform(size=300) < 0.18
    pred[flip] = rng.choice(classes, flip.sum())
    return pl.plot_confusion(truth, pred, "Confusion matrix")


@card(
    "Learning",
    "plot_dendrogram",
    [
        "Average-linkage tree; the dashed line is the cut that gives 3 clusters, branches below it belong together.",
    ],
)
def ln_dendrogram():
    rng = np.random.default_rng(66)
    pts = np.vstack([rng.normal(c, 0.5, (8, 2)) for c in ([0, 0], [4, 1], [1, 5])])
    return pl.plot_dendrogram(
        pd.DataFrame(pts, index=[f"obs {i}" for i in range(24)]),
        "Hierarchical clustering",
        n_clusters=3,
    )
