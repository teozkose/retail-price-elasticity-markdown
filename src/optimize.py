"""Choose a markdown depth for every product with a mixed-integer linear program.

Each product gets exactly one depth from DEPTHS. For depth d, predicted units over the promotion window
follow the fitted elasticity: units(d) = baseline_units * (1 - d) ** elasticity. Revenue, margin and the
discount cost of every (product, depth) pair are computed up front, so the program is linear in the
binary choice variables.

Scenarios:
  no_markdown      everything at list price (the reference)
  max_revenue      maximize revenue; inventory only
  revenue_margin   maximize revenue, but total margin must not fall below the no-markdown margin
  max_margin       maximize margin; inventory only
  budget_capped    maximize revenue with the discount budget capped at BUDGET_SHARE of baseline revenue

Assumptions, because the dataset has no cost or stock data: unit cost = COST_SHARE of list price;
inventory on hand = INVENTORY_WEEKS weeks of baseline demand. Both are parameters at the top of the file.
"""
import json
import os
import numpy as np
import pandas as pd
import pulp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.elasticity import ELASTICITIES, RESULTS, load_metrics, save_metrics
from src.prepare import WEEKLY

DEPTHS = [0.0, 0.10, 0.20, 0.30, 0.40]
WINDOW_WEEKS = 4           # length of the markdown window
BASELINE_WEEKS = 12        # recent weeks used for baseline demand and list price
COST_SHARE = 0.55          # unit cost as a share of list price
INVENTORY_WEEKS = 6        # stock on hand, in weeks of baseline demand
BUDGET_SHARE = 0.10        # discount budget as a share of baseline revenue
PLAN = os.path.join(RESULTS, "markdown_plan.csv")
SCENARIOS = os.path.join(RESULTS, "scenarios.csv")


def baselines(panel: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    recent = panel.sort_values("week").groupby("sku").tail(BASELINE_WEEKS)
    base = recent.groupby("sku").agg(base_units=("units", "mean"), list_price=("median_price", "median"))
    base = base.join(table.set_index("sku")[["elasticity", "description"]], how="inner")
    base["base_units"] = base["base_units"] * WINDOW_WEEKS
    base["cost"] = base["list_price"] * COST_SHARE
    base["inventory"] = base["base_units"] / WINDOW_WEEKS * INVENTORY_WEEKS
    return base.reset_index()


def candidates(base: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in base.itertuples():
        for d in DEPTHS:
            price = r.list_price * (1 - d)
            units = r.base_units * (1 - d) ** r.elasticity
            rows.append({"sku": r.sku, "depth": d, "price": price, "units": units,
                         "revenue": price * units, "margin": (price - r.cost) * units,
                         "discount_cost": r.list_price * d * units, "inventory": r.inventory})
    return pd.DataFrame(rows)


def solve(cand: pd.DataFrame, objective: str, margin_floor=None, budget=None) -> pd.DataFrame:
    prob = pulp.LpProblem("markdown", pulp.LpMaximize)
    x = {(r.sku, r.depth): pulp.LpVariable(f"x_{r.sku}_{int(r.depth * 100)}", cat="Binary") for r in cand.itertuples()}
    prob += pulp.lpSum(x[(r.sku, r.depth)] * getattr(r, objective) for r in cand.itertuples())
    for sku, g in cand.groupby("sku"):
        prob += pulp.lpSum(x[(sku, d)] for d in g["depth"]) == 1
        prob += pulp.lpSum(x[(sku, r.depth)] * r.units for r in g.itertuples()) <= g["inventory"].iloc[0]
    if margin_floor is not None:
        prob += pulp.lpSum(x[(r.sku, r.depth)] * r.margin for r in cand.itertuples()) >= margin_floor
    if budget is not None:
        prob += pulp.lpSum(x[(r.sku, r.depth)] * r.discount_cost for r in cand.itertuples()) <= budget
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    assert pulp.LpStatus[prob.status] == "Optimal", pulp.LpStatus[prob.status]
    chosen = cand[[x[(r.sku, r.depth)].value() > 0.5 for r in cand.itertuples()]]
    return chosen


def capped_by_inventory(cand: pd.DataFrame, chosen: pd.DataFrame) -> int:
    """Products whose chosen depth is the deepest one that still fits in inventory, while a deeper depth exists."""
    n = 0
    for r in chosen.itertuples():
        deeper = cand[(cand["sku"] == r.sku) & (cand["depth"] > r.depth)]
        if len(deeper) and (deeper["units"] > deeper["inventory"]).all():
            n += 1
    return n


def summarize(name: str, chosen: pd.DataFrame) -> dict:
    return {"scenario": name, "revenue": round(chosen["revenue"].sum(), 0), "margin": round(chosen["margin"].sum(), 0),
            "units": round(chosen["units"].sum(), 0), "discount_cost": round(chosen["discount_cost"].sum(), 0),
            "products_marked_down": int((chosen["depth"] > 0).sum()), "avg_depth": round(float(chosen["depth"].mean()), 3)}


def plot(scen: pd.DataFrame, path: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    idx = np.arange(len(scen)); w = 0.38
    ax.bar(idx - w / 2, scen["revenue"] / 1000, w, label="revenue")
    ax.bar(idx + w / 2, scen["margin"] / 1000, w, label="margin")
    ax.set_xticks(idx); ax.set_xticklabels(scen["scenario"], rotation=15)
    ax.set_ylabel(f"GBP thousands over {WINDOW_WEEKS} weeks")
    ax.set_title("Markdown scenarios")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    panel = pd.read_parquet(WEEKLY)
    table = pd.read_csv(ELASTICITIES, dtype={"sku": str})
    table = table[table["usable"]]
    base = baselines(panel, table)
    cand = candidates(base)

    no_md = cand[cand["depth"] == 0]
    base_rev, base_margin = no_md["revenue"].sum(), no_md["margin"].sum()
    runs = {
        "no_markdown": no_md,
        "max_revenue": solve(cand, "revenue"),
        "revenue_margin": solve(cand, "revenue", margin_floor=base_margin),
        "max_margin": solve(cand, "margin"),
        "budget_capped": solve(cand, "revenue", budget=BUDGET_SHARE * base_rev),
    }
    scen = pd.DataFrame([summarize(k, v) for k, v in runs.items()])
    scen.to_csv(SCENARIOS, index=False)
    plan = runs["revenue_margin"].merge(base[["sku", "description", "list_price", "elasticity"]], on="sku")
    plan = plan[["sku", "description", "list_price", "elasticity", "depth", "price", "units", "revenue", "margin"]]
    plan.sort_values("revenue", ascending=False).round(2).to_csv(PLAN, index=False)
    plot(scen, os.path.join(RESULTS, "scenarios.png"))

    m = load_metrics()
    m["optimization"] = {
        "products": int(len(base)), "depths": DEPTHS, "window_weeks": WINDOW_WEEKS,
        "assumptions": {"cost_share_of_list_price": COST_SHARE, "inventory_weeks": INVENTORY_WEEKS, "budget_share": BUDGET_SHARE},
        "scenarios": scen.to_dict(orient="records"),
        "revenue_margin_vs_no_markdown": {
            "revenue_lift_pct": round(100 * (scen.loc[2, "revenue"] / base_rev - 1), 1),
            "margin_change_pct": round(100 * (scen.loc[2, "margin"] / base_margin - 1), 1),
            "inventory_binding_products": capped_by_inventory(cand, runs["revenue_margin"]),
        },
        "max_revenue_inventory_binding_products": capped_by_inventory(cand, runs["max_revenue"]),
    }
    save_metrics(m)
    print(scen.to_string(index=False))
    print(f"saved {SCENARIOS} and {PLAN}")
