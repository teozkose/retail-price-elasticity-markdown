"""Render README.md from README.template.md and results/metrics.json so every number in the
write-up comes from the last run."""
import json
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
m = json.load(open(os.path.join(ROOT, "results", "metrics.json")))
e, o = m["elasticity"], m["optimization"]

scen = pd.DataFrame(o["scenarios"])
scen_rows = "\n".join(
    f"| {r.scenario.replace('_', ' ')} | {r.revenue:,.0f} | {r.margin:,.0f} | {r.units:,.0f} | {r.products_marked_down} | {100 * r.avg_depth:.0f}% |"
    for r in scen.itertuples()
)
examples = "\n".join(f"| {x['sku']} | {x['description'].strip().title()} | {x['elasticity']:.2f} | {x['r2']:.2f} |" for x in e["examples"])
plan = pd.read_csv(os.path.join(ROOT, "results", "markdown_plan.csv"), dtype={"sku": str}).head(8)
plan_rows = "\n".join(f"| {r.sku} | {str(r.description).strip().title()} | {r.list_price:.2f} | {r.elasticity:.2f} | {100 * r.depth:.0f}% | {r.units:,.0f} | {r.revenue:,.0f} |"
                      for r in plan.itertuples())
rm = o["revenue_margin_vs_no_markdown"]
fill = {
    "N_MODELED": str(e["products_modeled"]), "N_USABLE": str(e["products_usable"]),
    "SHARE_SIG": f"{100 * e['share_significant_negative']:.0f}%",
    "MEDIAN_E": f"{e['median_elasticity']:.2f}", "IQR_LO": f"{e['iqr_elasticity'][0]:.2f}", "IQR_HI": f"{e['iqr_elasticity'][1]:.2f}",
    "MEDIAN_R2": f"{e['median_r2']:.2f}", "EXAMPLES": examples,
    "N_OPT": str(o["products"]), "WINDOW": str(o["window_weeks"]),
    "DEPTHS": ", ".join(f"{100 * d:.0f}%" for d in o["depths"]),
    "COST_SHARE": f"{100 * o['assumptions']['cost_share_of_list_price']:.0f}%",
    "INV_WEEKS": str(o["assumptions"]["inventory_weeks"]), "BUDGET_SHARE": f"{100 * o['assumptions']['budget_share']:.0f}%",
    "SCENARIOS": scen_rows, "PLAN": plan_rows,
    "LIFT": f"{rm['revenue_lift_pct']:+.1f}%",
    "MARGIN_CHANGE": "margin unchanged" if abs(rm["margin_change_pct"]) < 0.05 else f"margin {rm['margin_change_pct']:+.1f}%",
    "BINDING": str(rm["inventory_binding_products"]),
    "BINDING_MAXREV": str(o["max_revenue_inventory_binding_products"]),
}
text = open(os.path.join(ROOT, "README.template.md")).read()
for k, v in fill.items():
    text = text.replace("{{" + k + "}}", v)
open(os.path.join(ROOT, "README.md"), "w").write(text)
print("README.md rendered")
