import numpy as np
import pandas as pd
import pytest

from segmentation import analysis
from segmentation.data import FEATURES_2D, INCOME, SPEND, load_customers
from segmentation.predict import assign
from segmentation.profiles import level, profile_segments


@pytest.fixture(scope="module")
def df():
    return load_customers()


def test_data_loads_clean(df):
    assert len(df) == 200
    assert set(df["gender"]) == {"Male", "Female"}
    assert df[INCOME].between(1, 200).all() and df[SPEND].between(1, 100).all()


def test_k5_is_best_by_silhouette_and_davies_bouldin(df):
    scores = analysis.k_selection(df[FEATURES_2D])
    best = analysis.best_k(scores)
    assert best["silhouette"] == 5 and best["davies_bouldin"] == 5
    assert scores["inertia"].is_monotonic_decreasing


def test_kmeans_is_deterministic_with_seed(df):
    a = analysis.fit_kmeans(df[FEATURES_2D], 5).predict(df[FEATURES_2D])
    b = analysis.fit_kmeans(df[FEATURES_2D], 5).predict(df[FEATURES_2D])
    np.testing.assert_array_equal(a, b)


def test_profiles_cover_all_customers_with_distinct_names(df):
    labels = analysis.fit_kmeans(df[FEATURES_2D], 5).predict(df[FEATURES_2D])
    prof = profile_segments(df, labels)
    assert prof["customers"].sum() == 200
    assert prof["segment"].nunique() == 5
    assert {"Premium spenders", "Careful affluent", "Budget conscious",
            "Impulsive budget shoppers", "Average customers"} == set(prof["segment"])


def test_level_thresholds():
    assert level(1, 10, 20) == "low" and level(15, 10, 20) == "mid" and level(25, 10, 20) == "high"


def test_stability_detects_clear_vs_random_structure():
    rng = np.random.default_rng(0)
    blobs = np.vstack([rng.normal(c, 0.3, (60, 2)) for c in [(0, 0), (5, 5), (0, 5)]])
    noise = rng.uniform(0, 1, (180, 2))
    assert np.median(analysis.bootstrap_stability(blobs, 3, n_boot=15)) > 0.95
    assert np.median(analysis.bootstrap_stability(noise, 3, n_boot=15)) < np.median(
        analysis.bootstrap_stability(blobs, 3, n_boot=15))


def test_algorithms_agree_on_this_data(df):
    table, labels = analysis.compare_algorithms(df[FEATURES_2D], 5)
    assert set(labels) == {"K-Means", "Hierarchical (Ward)", "Gaussian mixture"}
    assert (table["ARI vs K-Means"] > 0.85).all()


def test_predict_assigns_obvious_customers(df):
    model = analysis.fit_kmeans(df[FEATURES_2D], 5)
    labels = model.predict(df[FEATURES_2D])
    names = dict(zip(*profile_segments(df, labels)[["cluster", "segment"]].T.values))
    out = assign({"pipeline": model, "features": FEATURES_2D, "segment_names": names},
                 [95, 20, 95], [90, 10, 10])
    assert out["segment"].tolist() == ["Premium spenders", "Budget conscious", "Careful affluent"]
    assert isinstance(out, pd.DataFrame)
