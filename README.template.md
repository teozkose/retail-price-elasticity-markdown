# Price Elasticity and Markdown Optimization

How much does demand for each product respond to its price, and given that, which markdowns should a retailer run?

**Data:** two years of invoice lines from a UK online gift retailer (UCI Online Retail II, about 1 million lines). Aggregated to a weekly price/quantity panel for the {{N_MODELED}} highest-revenue products that were sold in at least 40 weeks at 5 or more distinct prices.

## 1. Elasticity per product

For each product, a log-log regression of weekly units on weekly median price, with month effects and a trend, gives the price elasticity (the % change in units for a 1% change in price). Robust standard errors.

- {{SHARE_SIG}} of the {{N_MODELED}} products have a significantly negative elasticity (p < 0.05).
- {{N_USABLE}} products pass the filter used downstream (significant, between -6 and -0.2).
- Median elasticity {{MEDIAN_E}} (interquartile range {{IQR_LO}} to {{IQR_HI}}); median R² {{MEDIAN_R2}}.

Best-fitting examples:

| SKU | Product | Elasticity | R² |
|---|---|---|---|
{{EXAMPLES}}

![Elasticities](results/elasticities.png)

These are large elasticities. Part of that is real (cheap gift items are very price-sensitive); part is the data: the retailer sells to both consumers and trade buyers, and weekly median price moves partly because the customer mix changes, not only because the list price changes. Treat the numbers as the observed response of weekly units to weekly price, not as clean causal estimates. The optimization below only needs the ranking and rough magnitude.

## 2. Markdown plan as a mixed-integer program

Each of the {{N_OPT}} usable products gets exactly one markdown depth from {{DEPTHS}} for a {{WINDOW}}-week window. Predicted units at depth d follow the fitted curve, units × (1 − d)^elasticity. Revenue, margin and discount cost per (product, depth) are precomputed, so the choice is a binary linear program solved with PuLP/CBC in under a second.

Assumptions, stated because the data has neither costs nor stock levels: unit cost = {{COST_SHARE}} of list price; inventory on hand = {{INV_WEEKS}} weeks of baseline demand; markdown budget = {{BUDGET_SHARE}} of baseline revenue. All three are parameters at the top of `src/optimize.py`.

| Scenario | Revenue (GBP) | Margin (GBP) | Units | Products marked down | Avg depth |
|---|---|---|---|---|---|
{{SCENARIOS}}

Reading the table: maximizing revenue alone discounts almost everything, gives up margin, and runs {{BINDING_MAXREV}} products into their inventory cap. Adding a margin floor ("revenue margin") keeps margin at least at the no-markdown level and still lifts revenue {{LIFT}} ({{MARGIN_CHANGE}}; {{BINDING}} products at their inventory cap). The budget-capped run shows what a fixed promotion budget buys.

Top rows of the recommended plan (revenue-max with the margin floor):

| SKU | Product | List price | Elasticity | Depth | Units | Revenue |
|---|---|---|---|---|---|---|
{{PLAN}}

![Scenarios](results/scenarios.png)

## How it works

```
data/download.py     fetch the workbook, cache as parquet (not committed)
src/prepare.py       clean, build the weekly panel, pick the products
src/elasticity.py    log-log OLS per product with month and trend controls; results/elasticities.csv
src/optimize.py      candidate table, five MILP scenarios; results/scenarios.csv, results/markdown_plan.csv
src/report.py        render this README from results/metrics.json
notebooks/01_analysis.ipynb   the same steps with the intermediate tables shown
```

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./run_all.sh
```

About three minutes, most of it reading the Excel workbook once. Every number above is read from `results/metrics.json`, which the run regenerates.

Data: [Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii), UCI Machine Learning Repository (CC BY 4.0).
