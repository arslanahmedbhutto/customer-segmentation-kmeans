"""Run the whole analysis: choose K, segment, profile, validate, save model and reports.

    python -m segmentation.run
"""

import json

import joblib
import numpy as np
import pandas as pd

from segmentation import FIGURES_DIR, MODELS_DIR, REPORTS_DIR, SEED
from segmentation import analysis, plots
from segmentation.data import FEATURES_2D, FEATURES_3D, load_customers
from segmentation.profiles import profile_segments

K = 5


def results() -> dict:
    """Everything the report and notebook need, computed in one place."""
    df = load_customers()
    X = df[FEATURES_2D]

    scores = analysis.k_selection(X)
    model = analysis.fit_kmeans(X, K)
    labels = model.predict(X)
    profiles = profile_segments(df, labels)
    km = model.named_steps["kmeans"]
    centers = model.named_steps["standardscaler"].inverse_transform(km.cluster_centers_)
    aris = analysis.bootstrap_stability(X, K)
    algos, algo_labels = analysis.compare_algorithms(X, K)

    scores_3d = analysis.k_selection(df[FEATURES_3D])
    return dict(df=df, scores=scores, best_k=analysis.best_k(scores), model=model, labels=labels,
                profiles=profiles, centers=centers, aris=aris, algos=algos, algo_labels=algo_labels,
                scores_3d=scores_3d, best_k_3d=analysis.best_k(scores_3d))


def main():
    r = results()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plots.distributions(r["df"], FIGURES_DIR / "distributions.png")
    plots.k_selection(r["scores"], K, FIGURES_DIR / "k_selection.png")
    plots.segments_scatter(r["df"], r["labels"], r["profiles"], r["centers"], FIGURES_DIR / "segments.png")
    plots.profile_bars(r["profiles"], FIGURES_DIR / "segment_profiles.png")
    plots.stability(r["aris"], FIGURES_DIR / "stability.png")
    plots.algorithm_comparison(r["df"], r["algo_labels"], FIGURES_DIR / "algorithm_comparison.png")

    MODELS_DIR.mkdir(exist_ok=True)
    names = dict(zip(r["profiles"]["cluster"], r["profiles"]["segment"]))
    joblib.dump({"pipeline": r["model"], "features": FEATURES_2D, "segment_names": names},
                MODELS_DIR / "kmeans_segments.joblib")

    seg = r["df"].assign(cluster=r["labels"], segment=[names[c] for c in r["labels"]])
    seg.to_csv(REPORTS_DIR / "customer_segments.csv", index=False)

    profiles = r["profiles"].round(1)
    (REPORTS_DIR / "segment_profiles.md").write_text(
        "# Segment profiles (K-Means, K=5, income + spending score)\n\n" + profiles.to_markdown(index=False) + "\n",
        encoding="utf-8")
    summary = {
        "k": K,
        "best_k_by_index_2d": r["best_k"],
        "best_k_by_index_3d": r["best_k_3d"],
        "silhouette_k5": float(r["scores"].loc[r["scores"]["k"] == K, "silhouette"].iloc[0]),
        "bootstrap_ari": {"median": float(np.median(r["aris"])), "p5": float(np.percentile(r["aris"], 5)),
                          "mean": float(r["aris"].mean()), "n": len(r["aris"])},
        "k_scores_2d": r["scores"].round(4).to_dict("records"),
        "k_scores_3d": r["scores_3d"].round(4).to_dict("records"),
        "algorithms": r["algos"].round(4).to_dict("records"),
        "profiles": profiles.to_dict("records"),
    }
    (REPORTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.set_option("display.width", 160)
    print(r["scores"].round(3).to_string(index=False))
    print("best K by index:", r["best_k"], "| 3-feature:", r["best_k_3d"])
    print(profiles.drop(columns="suggested_action").to_string(index=False))
    print(f"bootstrap ARI median {np.median(r['aris']):.3f}, 5th pct {np.percentile(r['aris'], 5):.3f}")
    print(r["algos"].round(3).to_string(index=False))


if __name__ == "__main__":
    np.random.seed(SEED)
    main()
