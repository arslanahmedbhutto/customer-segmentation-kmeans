"""Turn cluster numbers into named, actionable customer segments."""

import pandas as pd

from segmentation.data import AGE, INCOME, SPEND

# Persona per (income level, spending level), each relative to the customer base
PERSONAS = {
    ("high", "high"): ("Premium spenders", "Loyalty / VIP programme, early access, premium lines"),
    ("high", "low"): ("Careful affluent", "Find out why they hold back: quality, service, personalised offers"),
    ("low", "high"): ("Impulsive budget shoppers", "Promotions and bundles; watch credit/returns risk"),
    ("low", "low"): ("Budget conscious", "Value ranges, discount days, low-cost retention"),
    ("mid", "mid"): ("Average customers", "Standard campaigns; upsell toward the premium segment"),
}


def level(value: float, low: float, high: float) -> str:
    return "low" if value < low else "high" if value > high else "mid"


def profile_segments(df: pd.DataFrame, labels) -> pd.DataFrame:
    """Size, averages, gender mix and a persona for every cluster."""
    d = df.assign(cluster=labels)
    inc_lo, inc_hi = df[INCOME].quantile([1 / 3, 2 / 3])
    sp_lo, sp_hi = df[SPEND].quantile([1 / 3, 2 / 3])

    rows = []
    for c, g in d.groupby("cluster"):
        key = (level(g[INCOME].mean(), inc_lo, inc_hi), level(g[SPEND].mean(), sp_lo, sp_hi))
        name, action = PERSONAS.get(key, (f"{key[0]} income / {key[1]} spending".capitalize(), "—"))
        rows.append({
            "cluster": int(c), "segment": name, "customers": len(g), "share_%": 100 * len(g) / len(d),
            "avg_income_k": g[INCOME].mean(), "avg_spending_score": g[SPEND].mean(),
            "avg_age": g[AGE].mean(), "female_%": 100 * (g["gender"] == "Female").mean(),
            "suggested_action": action,
        })
    return pd.DataFrame(rows).sort_values("avg_income_k").reset_index(drop=True)
