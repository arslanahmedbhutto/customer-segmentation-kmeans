"""Assign new customers to a segment with the saved model.

    python -m segmentation.predict --income 90 --spending 85
    python -m segmentation.predict --income 25 --spending 15 --income 60 --spending 50
"""

import argparse

import joblib
import pandas as pd

from segmentation import MODELS_DIR


def load(path=MODELS_DIR / "kmeans_segments.joblib") -> dict:
    return joblib.load(path)


def assign(bundle: dict, income_k, spending_score) -> pd.DataFrame:
    X = pd.DataFrame({"income_k": income_k, "spending_score": spending_score})[bundle["features"]]
    clusters = bundle["pipeline"].predict(X)
    return X.assign(cluster=clusters, segment=[bundle["segment_names"][c] for c in clusters])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--income", type=float, action="append", required=True, help="annual income in k$")
    ap.add_argument("--spending", type=float, action="append", required=True, help="spending score 1-100")
    args = ap.parse_args()
    if len(args.income) != len(args.spending):
        ap.error("give one --spending per --income")
    print(assign(load(), args.income, args.spending).to_string(index=False))


if __name__ == "__main__":
    main()
