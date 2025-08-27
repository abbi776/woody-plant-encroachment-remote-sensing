#!/usr/bin/env python3
"""
Scan a folder for classified rasters named like:
  P1ANAE_wetlands_classified_map_2016_RF_normaldata.tif
  P3ANAE-wetlands-classified-map-2018.tif
Compute per-class area (ha) per ROI and year, plus consecutive & named-pair deltas.
Write an Excel workbook with multiple sheets.
"""

import os, re, argparse
import numpy as np
import pandas as pd
import rasterio

LABELS = {0: "River Red Gum", 1: "Non-Forest Floodplain", 2: "Water"}

PATTERN = re.compile(
    r"^(?P<roi>P0*\d+)[\s_-]*ANAE[\s_-]*wetlands[\s_-]*classified[\s_-]*map[\s_-]*"
    r"(?P<year>\d{4})(?:[A-Za-z0-9._-]*)?(?:\.(?:tif|tiff))?$",
    re.IGNORECASE
)

def normalize_roi(roi_raw: str) -> str:
    return re.sub(r"^P0*", "P", roi_raw.upper().replace(" ", ""))

def pairwise_consecutive(sorted_years):
    if not sorted_years: return
    prev = sorted_years[0]
    for y in sorted_years[1:]:
        yield prev, y
        prev = y

def class_areas_ha(path):
    with rasterio.open(path) as src:
        data = src.read(1)
        pix_area_m2 = src.res[0] * src.res[1]
        nodata = src.nodata
    unique, counts = np.unique(data, return_counts=True)
    out = {}
    for cls, cnt in zip(unique, counts):
        if nodata is not None and cls == nodata:  # skip nodata
            continue
        if nodata is None and cls == -1:          # common manual nodata
            continue
        out[int(cls)] = (int(cnt) * pix_area_m2) / 10_000.0
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, help="Directory containing classified rasters")
    ap.add_argument("--out-xlsx", required=True, help="Excel output path")
    ap.add_argument("--named-pairs", nargs="*", default=["2016:2018","2018:2025","2016:2025"],
                    help="Pairs like 2016:2018 2018:2025")
    args = ap.parse_args()

    files = os.listdir(args.in_dir)
    matches = []
    for f in files:
        m = PATTERN.match(f)
        if m:
            matches.append((normalize_roi(m.group("roi")), int(m.group("year")), f))

    print(f"Found {len(matches)} matching rasters.")
    if not matches:
        os.makedirs(os.path.dirname(args.out_xlsx) or ".", exist_ok=True)
        with pd.ExcelWriter(args.out_xlsx, engine="openpyxl") as xw:
            pd.DataFrame().to_excel(xw, index=False, sheet_name="areas_long")
        print(f"✅ Wrote empty workbook to: {args.out_xlsx}")
        return

    # ROI → year → filename
    roi_year = {}
    for roi, year, fname in matches:
        roi_year.setdefault(roi, {})[year] = fname
    for roi, ymap in sorted(roi_year.items()):
        print(f"  {roi}: years -> {sorted(ymap.keys())}")

    # Long table
    rows = []
    for roi, ymap in sorted(roi_year.items()):
        for year, fname in sorted(ymap.items()):
            path = os.path.join(args.in_dir, fname)
            try:
                areas = class_areas_ha(path)
            except Exception as e:
                print(f"⚠️ Failed reading {path}: {e}")
                continue
            for cls, area in areas.items():
                rows.append({
                    "ROI": roi,
                    "year": year,
                    "class": cls,
                    "class_label": LABELS.get(cls, f"Class {cls}"),
                    "area_ha": round(float(area), 4)
                })
    if not rows:
        print("No class areas computed. Writing empty workbook.")
        with pd.ExcelWriter(args.out_xlsx, engine="openpyxl") as xw:
            pd.DataFrame().to_excel(xw, index=False, sheet_name="areas_long")
        return

    df_long = (pd.DataFrame(rows)
               .sort_values(["ROI","class","year"])
               .reset_index(drop=True))

    # Wide table (years as columns)
    df_wide = (df_long
               .pivot_table(index=["ROI","class","class_label"],
                            columns="year", values="area_ha", fill_value=0.0)
               .reset_index().rename_axis(None, axis=1))

    # Consecutive deltas
    pair_rows = []
    for roi, g in df_long.groupby("ROI"):
        years = sorted(g["year"].unique())
        for y1, y2 in pairwise_consecutive(years):
            gg = g[g["year"].isin([y1,y2])]
            for cls, gcls in gg.groupby("class"):
                lab = gcls["class_label"].iloc[0]
                a1 = float(gcls.loc[gcls["year"]==y1, "area_ha"].sum())
                a2 = float(gcls.loc[gcls["year"]==y2, "area_ha"].sum())
                change = a2 - a1
                pct = (change / a1 * 100.0) if a1 != 0 else np.nan
                pair_rows.append({
                    "ROI": roi, "class": cls, "class_label": lab,
                    "year_from": y1, "year_to": y2,
                    "area_from_ha": round(a1,4), "area_to_ha": round(a2,4),
                    "change_ha": round(change,4),
                    "percent_change": round(pct,2) if np.isfinite(pct) else "NA"
                })
    df_pair_consec = pd.DataFrame(pair_rows)

    # Named pairs
    named_pairs = []
    for token in args.named_pairs:
        try:
            y1, y2 = map(int, token.split(":"))
            named_pairs.append((y1, y2))
        except Exception:
            print(f"⚠️ Skipping malformed named pair: {token}")

    named_rows = []
    for roi, g in df_long.groupby("ROI"):
        avail = set(g["year"].unique())
        for y1, y2 in named_pairs:
            if {y1,y2}.issubset(avail):
                gg = g[g["year"].isin([y1,y2])]
                for cls, gcls in gg.groupby("class"):
                    lab = gcls["class_label"].iloc[0]
                    a1 = float(gcls.loc[gcls["year"]==y1,"area_ha"].sum())
                    a2 = float(gcls.loc[gcls["year"]==y2,"area_ha"].sum())
                    change = a2 - a1
                    pct = (change / a1 * 100.0) if a1 != 0 else np.nan
                    named_rows.append({
                        "ROI": roi, "class": cls, "class_label": lab,
                        "year_from": y1, "year_to": y2,
                        "area_from_ha": round(a1,4), "area_to_ha": round(a2,4),
                        "change_ha": round(change,4),
                        "percent_change": round(pct,2) if np.isfinite(pct) else "NA"
                    })
    df_pair_named = pd.DataFrame(named_rows)

    # Write Excel
    os.makedirs(os.path.dirname(args.out_xlsx) or ".", exist_ok=True)
    with pd.ExcelWriter(args.out_xlsx, engine="openpyxl") as xw:
        df_long.to_excel(xw, index=False, sheet_name="areas_long")
        df_wide.to_excel(xw, index=False, sheet_name="areas_wide")
        df_pair_consec.to_excel(xw, index=False, sheet_name="pairwise_consecutive")
        df_pair_named.to_excel(xw, index=False, sheet_name="pairwise_named")
    print(f"✅ Saved: {args.out_xlsx}")

if __name__ == "__main__":
    main()
