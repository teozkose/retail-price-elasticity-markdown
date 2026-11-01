"""Download the UCI Online Retail II dataset (CC BY 4.0) and cache it as parquet.

The source is a 45 MB Excel workbook with two yearly sheets (Dec 2009 to Dec 2011, about 1.07 million
invoice lines from a UK online gift retailer). Reading it takes about a minute; the parquet cache makes
every later step fast.
"""
import io
import os
import sys
import urllib.request
import zipfile

import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(HERE, "transactions.parquet")

if os.path.exists(DEST) and os.path.getsize(DEST) > 1_000_000:
    print(f"already present: {DEST}")
    sys.exit(0)

print(f"downloading {URL}")
blob = urllib.request.urlopen(URL, timeout=120).read()
with zipfile.ZipFile(io.BytesIO(blob)) as z:
    name = [n for n in z.namelist() if n.endswith(".xlsx")][0]
    xlsx = z.read(name)
print("reading workbook (about a minute)")
book = pd.ExcelFile(io.BytesIO(xlsx))
df = pd.concat([pd.read_excel(book, sheet) for sheet in book.sheet_names], ignore_index=True)
for c in ["Invoice", "StockCode", "Description", "Country"]:
    df[c] = df[c].astype(str)
df.to_parquet(DEST, index=False)
print(f"saved {DEST}: {len(df):,} rows")
