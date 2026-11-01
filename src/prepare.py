"""Turn invoice lines into a weekly price/quantity panel for the products worth modeling.

Output: data/weekly.parquet with one row per (sku, week): units, revenue, median unit price, orders.
Products kept: UK sales, real stock codes, at least MIN_WEEKS weeks of sales and MIN_PRICE_LEVELS
distinct weekly prices, then the TOP_N by revenue.
"""
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "transactions.parquet")
WEEKLY = os.path.join(ROOT, "data", "weekly.parquet")
SKUS = os.path.join(ROOT, "results", "products.csv")

TOP_N = 60
MIN_WEEKS = 40
MIN_PRICE_LEVELS = 5


def load_clean(path: str = RAW) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df[(df["Country"] == "United Kingdom") & (df["Quantity"] > 0) & (df["Price"] > 0)]
    df = df[~df["Invoice"].str.startswith("C")]                      # cancellations
    df = df[df["StockCode"].str.match(r"^\d{5}[A-Za-z]?$")]          # drops postage, manual, fees
    df = df[df["Price"] < 100]                                        # a few data-entry outliers
    return df.rename(columns={"StockCode": "sku", "Quantity": "units", "Price": "price",
                              "InvoiceDate": "ts", "Description": "description", "Invoice": "invoice"})


def weekly_panel(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(week=df["ts"].dt.to_period("W").dt.start_time, revenue=df["units"] * df["price"])
    g = df.groupby(["sku", "week"]).agg(units=("units", "sum"), revenue=("revenue", "sum"),
                                        median_price=("price", "median"), orders=("invoice", "nunique")).reset_index()
    g["avg_price"] = g["revenue"] / g["units"]
    return g


def select_products(panel: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    stats = panel.groupby("sku").agg(weeks=("week", "nunique"), revenue=("revenue", "sum"),
                                     price_levels=("median_price", lambda s: s.round(2).nunique()))
    keep = stats[(stats["weeks"] >= MIN_WEEKS) & (stats["price_levels"] >= MIN_PRICE_LEVELS)]
    keep = keep.sort_values("revenue", ascending=False).head(TOP_N)
    names = df.groupby("sku")["description"].agg(lambda s: s.mode().iloc[0])
    keep["description"] = names.reindex(keep.index)
    return keep.reset_index()


if __name__ == "__main__":
    df = load_clean()
    panel = weekly_panel(df)
    products = select_products(panel, df)
    panel = panel[panel["sku"].isin(products["sku"])].sort_values(["sku", "week"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(SKUS), exist_ok=True)
    panel.to_parquet(WEEKLY, index=False)
    products.to_csv(SKUS, index=False)
    print(f"invoice lines: {len(df):,} -> weekly rows: {len(panel):,} for {len(products)} products")
    print(f"saved {WEEKLY} and {SKUS}")
