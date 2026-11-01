"""Estimate a price elasticity for every product with a log-log regression on its weekly panel.

    log(units) = a + e * log(price) + month effects + trend

e is the elasticity: a 1% price change moves units by e%. Standard errors are heteroskedasticity-robust.
Products whose elasticity is not significantly negative are kept in the table but excluded from the
markdown optimization.
"""
import json
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.prepare import SKUS, WEEKLY

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
ELASTICITIES = os.path.join(RESULTS, "elasticities.csv")
METRICS = os.path.join(RESULTS, "metrics.json")
CLIP = (-6.0, -0.2)   # range used by the optimizer; anything outside is treated as not usable


def fit_one(g: pd.DataFrame) -> dict:
    g = g.sort_values("week")
    X = pd.DataFrame({"log_price": np.log(g["median_price"]),
                      "trend": np.arange(len(g)) / 52.0})
    X = pd.concat([X, pd.get_dummies(g["week"].dt.month, prefix="m", drop_first=True, dtype=float)], axis=1)
    X = sm.add_constant(X)
    model = sm.OLS(np.log(g["units"]), X).fit(cov_type="HC1")
    return {"elasticity": float(model.params["log_price"]), "se": float(model.bse["log_price"]),
            "p_value": float(model.pvalues["log_price"]), "r2": float(model.rsquared), "weeks": int(len(g))}


def load_metrics() -> dict:
    return json.load(open(METRICS)) if os.path.exists(METRICS) else {}


def save_metrics(m: dict):
    json.dump(m, open(METRICS, "w"), indent=2)


def plot(table: pd.DataFrame, panel: pd.DataFrame, path: str):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    usable = table[table["usable"]]
    axes[0].hist(usable["elasticity"], bins=15, edgecolor="white")
    axes[0].set_title(f"Price elasticity, {len(usable)} products")
    axes[0].set_xlabel("elasticity (log-log slope)")
    spread = panel.groupby("sku")["median_price"].agg(lambda s: np.log(s).std())
    examples = usable.assign(spread=usable["sku"].map(spread)).sort_values("spread", ascending=False)["sku"].head(2)
    for ax, sku in zip(axes[1:], examples):
        g = panel[panel["sku"] == sku]
        ax.scatter(g["median_price"], g["units"], s=12, alpha=0.6)
        e = float(table.loc[table["sku"] == sku, "elasticity"].iloc[0])
        p = np.linspace(g["median_price"].min(), g["median_price"].max(), 50)
        k = np.exp(np.log(g["units"]).mean() - e * np.log(g["median_price"]).mean())
        ax.plot(p, k * p ** e, color="black", linewidth=1)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"{sku}: elasticity {e:.2f} (log axes)")
        ax.set_xlabel("weekly median price (GBP)")
        ax.set_ylabel("weekly units")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    panel = pd.read_parquet(WEEKLY)
    products = pd.read_csv(SKUS, dtype={"sku": str})
    rows = []
    for sku, g in panel.groupby("sku"):
        r = fit_one(g)
        r["sku"] = sku
        rows.append(r)
    table = pd.DataFrame(rows).merge(products[["sku", "description", "revenue"]], on="sku")
    table["usable"] = (table["p_value"] < 0.05) & (table["elasticity"] < CLIP[1]) & (table["elasticity"] > CLIP[0])
    table = table.sort_values("revenue", ascending=False)
    table.to_csv(ELASTICITIES, index=False)
    plot(table, panel, os.path.join(RESULTS, "elasticities.png"))

    usable = table[table["usable"]]
    m = load_metrics()
    m["elasticity"] = {
        "products_modeled": int(len(table)),
        "products_usable": int(len(usable)),
        "share_significant_negative": round(float(((table["p_value"] < 0.05) & (table["elasticity"] < 0)).mean()), 3),
        "median_elasticity": round(float(usable["elasticity"].median()), 2),
        "iqr_elasticity": [round(float(usable["elasticity"].quantile(0.25)), 2), round(float(usable["elasticity"].quantile(0.75)), 2)],
        "median_r2": round(float(usable["r2"].median()), 2),
        "examples": [{"sku": r.sku, "description": r.description, "elasticity": round(r.elasticity, 2), "r2": round(r.r2, 2)}
                     for r in usable.sort_values("r2", ascending=False).head(5).itertuples()],
    }
    save_metrics(m)
    print(f"{len(usable)} of {len(table)} products have a usable elasticity; median {m['elasticity']['median_elasticity']}")
    print(f"saved {ELASTICITIES}")
