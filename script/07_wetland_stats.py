#!/usr/bin/env python3
"""
Step 7 — Wetland-level stats from classified maps (multi-year).

Examples:
  # Explicit rasters
  python scripts/07_wetland_stats.py \
    --wetlands data/raw/wetlands/P5ANAEWetlands.shp \
    --rasters 2016 data/processed/P5_wetlands_classified_2016.tif \
    --rasters 2018 data/processed/P5_wetlands_classified_2018.tif \
    --rasters 2025 data/processed/P5_wetlands_classified_2025.tif \
    --out-xlsx docs/tables/wetland_class_areas_timeseries.xlsx

  # Auto-detect in a directory (filenames like P5ANAE_wetlands_classified_map_2016_*.tif)
  python scripts/07_wetland_stats.py \
    --wetlands data/raw/wetlands/P5ANAEWetlands.shp \
    --in-dir data/processed/rf_maps \
    --fname-pattern '(?P<year>\\d{4}).*\\.tif$' \
    --out-xlsx docs/tables/wetland_class_areas_timeseries.xlsx
"""
import os, re, argparse, itertools
import numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterstats import zonal_stats

DEFAULT_CLASS_MAP = {0: "River Red Gum", 1: "Non-Forest Floodplain", 2: "Water"}

def pick_excel_engine():
    for eng in ("xlsxwriter", "openpyxl"):
        try:
            __import__(eng)
            return eng
        except ImportError:
            continue
    return None

def excel_safe(name: str, used=None) -> str:
    safe = re.sub(r"[\\[\\]:*?/\\\\]", "_", name).strip("'")
    safe = safe[:31] if len(safe) > 31 else safe
    if used is not None:
        base, i = safe, 1
        while safe in used:
            suffix = f"_{i}"
            safe = (base[:31 - len(suffix)] + suffix) if len(base) + len(suffix) > 31 else base + suffix
            i += 1
        used.add(safe)
    return safe

def consecutive_pairs(yrs): return list(zip(yrs[:-1], yrs[1:]))

def flatten_pair_cols(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.columns, pd.MultiIndex):
        df = df.copy(); df.columns = [str(c) for c in df.columns]; return df
    def _fmt(col):
        if isinstance(col, tuple) and len(col) == 2:
            a, b = col; return f"{a}_to_{b}" if b not in ("", None) else str(a)
        return "_".join(str(x) for x in col if x != "")
    df = df.copy(); df.columns = [_fmt(c) for c in df.columns]; return df

def pairwise_changes(pivot_df, years_list):
    rows = []
    for y0, y1 in itertools.combinations(years_list, 2):
        tmp = pivot_df.copy()
        tmp["t0"], tmp["t1"] = y0, y1
        tmp["change_ha"] = tmp[y1] - tmp[y0]
        tmp["percent_change"] = np.where(tmp[y0] > 0, np.round((tmp["change_ha"] / tmp[y0]) * 100.0, 2), np.nan)
        rows.append(tmp[["WetlandID","class","class_label","t0","t1",y0,y1,"change_ha","percent_change"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def detect_rasters(in_dir, fname_pattern):
    pat = re.compile(fname_pattern, re.IGNORECASE)
    files = [f for f in os.listdir(in_dir) if f.lower().endswith((".tif",".tiff"))]
    out = {}
    for f in files:
        m = pat.search(f)
        if m and "year" in m.groupdict():
            yr = int(m.group("year"))
            out[yr] = os.path.join(in_dir, f)
    return out

def compute_per_wetland(stats_gdf, year, pixel_area_m2, label_map, nodata=-1, id_col="OBJECTID"):
    zs = zonal_stats(stats_gdf, stats_gdf._current_raster, categorical=True, nodata=nodata)
    recs = []
    for i, cat_counts in enumerate(zs):
        wetland_id = stats_gdf.iloc[i].get(id_col, f"Wetland_{i+1}")
        wetland_area_ha = float(stats_gdf.iloc[i]["_wetland_area_ha"])
        total_pix = int(sum((cat_counts or {}).values())) if cat_counts else 0
        present_classes = set((cat_counts or {}).keys())
        all_classes = sorted(present_classes.union(set(label_map.keys())))
        for cls in all_classes:
            cnt = int(cat_counts.get(cls, 0)) if cat_counts else 0
            area_ha = (cnt * pixel_area_m2) / 10_000.0
            pct_of_polygon_area = (area_ha / wetland_area_ha * 100.0) if wetland_area_ha > 0 else 0.0
            recs.append({
                "WetlandID": wetland_id,
                "Year": int(year),
                "class": int(cls),
                "class_label": label_map.get(int(cls), f"Class {cls}"),
                "pixel_count": cnt,
                "area_ha": round(area_ha, 4),
                "percent_of_polygon_area": round(pct_of_polygon_area, 3),
                "wetland_area_ha": round(wetland_area_ha, 4),
                "total_valid_pixels": total_pix,
            })
    return recs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wetlands", required=True, help="Wetland polygons (SHP/GeoJSON)")
    ap.add_argument("--wetland-id-col", default="OBJECTID")
    ap.add_argument("--rasters", nargs=2, action="append", metavar=("YEAR","PATH"),
                    help="Repeat for each year, e.g., --rasters 2016 path.tif")
    ap.add_argument("--in-dir", default=None, help="Directory to auto-detect rasters")
    ap.add_argument("--fname-pattern", default=r"(?P<year>\d{4}).*\.tif(f)?$",
                    help="Regex with (?P<year>YYYY) to pick year from filename")
    ap.add_argument("--class-map", default=None, help="Path to CSV with columns class,label")
    ap.add_argument("--nodata", type=int, default=-1)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--min-change-ha", type=float, default=None, help="Filter small |Δ| in RRG change tables")
    ap.add_argument("--out-xlsx", required=True)
    args = ap.parse_args()

    # Class labels
    label_map = DEFAULT_CLASS_MAP
    if args.class_map and os.path.exists(args.class_map):
        df = pd.read_csv(args.class_map)
        label_map = {int(r["class"]): str(r["label"]) for _, r in df.iterrows()}

    # Rasters (explicit or auto-detect)
    rasters = {}
    if args.rasters:
        for y, p in args.rasters:
            rasters[int(y)] = p
    elif args.in_dir:
        rasters = detect_rasters(args.in_dir, args.fname_pattern)
    else:
        raise SystemExit("Provide either --rasters YEAR PATH ... or --in-dir with --fname-pattern.")

    years_sorted = sorted(rasters.keys())
    if not years_sorted: raise SystemExit("No rasters found.")
    first = years_sorted[0]
    if not os.path.exists(rasters[first]):
        raise FileNotFoundError(f"Raster not found: {rasters[first]}")

    # Load wetlands and align CRS
    gdf = gpd.read_file(args.wetlands)
    with rasterio.open(rasters[first]) as src0:
        raster_crs = src0.crs
        pixel_area_m2 = src0.res[0] * src0.res[1]
    if gdf.crs != raster_crs:
        gdf = gdf.to_crs(raster_crs)
    gdf["_wetland_area_ha"] = gdf.geometry.area / 10_000.0

    # Iterate years
    all_records = []
    for yr in years_sorted:
        rpath = rasters[yr]
        if not os.path.exists(rpath): raise FileNotFoundError(f"{yr} raster not found: {rpath}")
        gdf._current_raster = rpath  # pass path into compute func
        all_records.extend(compute_per_wetland(gdf, yr, pixel_area_m2, label_map, nodata=args.nodata, id_col=args.wetland_id_col))

    df_all = pd.DataFrame(all_records)

    # Pivots
    pivot_area = (df_all.pivot_table(index=["WetlandID","class","class_label"], columns="Year",
                                     values="area_ha", aggfunc="sum", fill_value=0.0)
                  .reset_index())[["WetlandID","class","class_label",*years_sorted]]

    pivot_pixels = (df_all.pivot_table(index=["WetlandID","class","class_label"], columns="Year",
                                       values="pixel_count", aggfunc="sum", fill_value=0)
                    .reset_index())[["WetlandID","class","class_label",*years_sorted]]

    pivot_percent_polygon = (df_all.pivot_table(index=["WetlandID","class","class_label"], columns="Year",
                                                values="percent_of_polygon_area", aggfunc="mean", fill_value=0.0)
                             .reset_index())[["WetlandID","class","class_label",*years_sorted]]

    # Pairwise changes (area)
    changes_area = pairwise_changes(pivot_area, years_sorted)

    # RRG encroachment/retreat tables (class 0 by default)
    rrg = 0 if 0 in label_map else list(label_map.keys())[0]
    rrg_changes = changes_area[changes_area["class"] == rrg].copy()
    rrg_changes["direction"] = np.where(
        rrg_changes["change_ha"] > 0, "Encroachment",
        np.where(rrg_changes["change_ha"] < 0, "Retreat", "No change")
    )
    if args.min_change_ha is not None:
        rrg_changes = rrg_changes[rrg_changes["change_ha"].abs() >= float(args.min_change_ha)]

    # Quick counts per pair
    summary_rows = []
    for (t0, t1), sub in rrg_changes.groupby(["t0","t1"]):
        enc = int((sub["direction"] == "Encroachment").sum())
        ret = int((sub["direction"] == "Retreat").sum())
        same = int((sub["direction"] == "No change").sum())
        summary_rows.append({"t0": t0, "t1": t1, "encroachment_count": enc, "retreat_count": ret, "no_change_count": same})
    rrg_quick_counts = pd.DataFrame(summary_rows).sort_values(["t0","t1"]).reset_index(drop=True)

    # Global tops
    rrg_top_enc = (rrg_changes.sort_values("change_ha", ascending=False)
                   .head(args.top_k)[["WetlandID","t0","t1","change_ha","percent_change"]]
                   .reset_index(drop=True))
    rrg_top_ret = (rrg_changes.sort_values("change_ha", ascending=True)
                   .head(args.top_k)[["WetlandID","t0","t1","change_ha","percent_change"]]
                   .reset_index(drop=True))

    # Consecutive-only tops + continuous/flip patterns
    cons_pairs = consecutive_pairs(years_sorted)
    pair_top_sheets = {}
    for (t0, t1) in cons_pairs:
        sub = rrg_changes[(rrg_changes["t0"] == t0) & (rrg_changes["t1"] == t1)].copy()
        pair_top_sheets[f"RRG_top_encroachment_{t0}_{t1}"] = (sub.sort_values("change_ha", ascending=False)
                                                               .head(args.top_k)[["WetlandID","t0","t1","change_ha","percent_change"]])
        pair_top_sheets[f"RRG_top_retreat_{t0}_{t1}"] = (sub.sort_values("change_ha", ascending=True)
                                                         .head(args.top_k)[["WetlandID","t0","t1","change_ha","percent_change"]])

    rrg_continuous_enc = pd.DataFrame(); rrg_continuous_ret = pd.DataFrame()
    rrg_flip_enc_to_ret = pd.DataFrame(); rrg_flip_ret_to_enc = pd.DataFrame()

    if cons_pairs:
        rrg_cons = rrg_changes[rrg_changes.apply(lambda r: (r["t0"], r["t1"]) in cons_pairs, axis=1)].copy()
        rrg_wide = rrg_cons.pivot_table(index="WetlandID", columns=["t0","t1"], values="change_ha", aggfunc="sum", fill_value=np.nan)
        have_all = rrg_wide.dropna()

        # All deltas > 0 → continuous encroachers
        enc_mask = (have_all > 0).all(axis=1)
        if enc_mask.any():
            rrg_continuous_enc = have_all[enc_mask].copy()
            rrg_continuous_enc["sum_change_ha"] = rrg_continuous_enc.sum(axis=1)
            rrg_continuous_enc["min_change_ha"] = rrg_continuous_enc.min(axis=1)
            rrg_continuous_enc = rrg_continuous_enc.sort_values(["min_change_ha","sum_change_ha"], ascending=[False,False]).reset_index()

        # All deltas < 0 → continuous retreaters
        ret_mask = (have_all < 0).all(axis=1)
        if ret_mask.any():
            rrg_continuous_ret = have_all[ret_mask].copy()
            rrg_continuous_ret["sum_change_ha"] = rrg_continuous_ret.sum(axis=1)
            rrg_continuous_ret["max_change_ha"] = rrg_continuous_ret.max(axis=1)
            rrg_continuous_ret = rrg_continuous_ret.sort_values(["max_change_ha","sum_change_ha"], ascending=[True,True]).reset_index()

        # Flip patterns
        if have_all.shape[1] >= 2:
            signs = np.sign(have_all.values)
            df_signs = pd.DataFrame(signs, index=have_all.index, columns=have_all.columns)
            enc_to_ret_ids = df_signs[(df_signs.iloc[:,0] > 0) & (df_signs.iloc[:,-1] < 0)].index
            ret_to_enc_ids = df_signs[(df_signs.iloc[:,0] < 0) & (df_signs.iloc[:,-1] > 0)].index
            if len(enc_to_ret_ids): rrg_flip_enc_to_ret = have_all.loc[enc_to_ret_ids].copy().reset_index()
            if len(ret_to_enc_ids): rrg_flip_ret_to_enc = have_all.loc[ret_to_enc_ids].copy().reset_index()

    # Flatten MultiIndex cols for Excel
    for df in (rrg_continuous_enc, rrg_continuous_ret, rrg_flip_enc_to_ret, rrg_flip_ret_to_enc):
        if not df.empty:
            df.sort_index(axis=1, inplace=True)
    if not rrg_continuous_enc.empty: rrg_continuous_enc = flatten_pair_cols(rrg_continuous_enc)
    if not rrg_continuous_ret.empty: rrg_continuous_ret = flatten_pair_cols(rrg_continuous_ret)
    if not rrg_flip_enc_to_ret.empty: rrg_flip_enc_to_ret = flatten_pair_cols(rrg_flip_enc_to_ret)
    if not rrg_flip_ret_to_enc.empty: rrg_flip_ret_to_enc = flatten_pair_cols(rrg_flip_ret_to_enc)

    # Write Excel
    os.makedirs(os.path.dirname(args.out_xlsx) or ".", exist_ok=True)
    engine = pick_excel_engine(); used = set()
    with pd.ExcelWriter(args.out_xlsx, engine=engine) as xw:
        df_all.to_excel(xw, index=False, sheet_name=excel_safe("long_all_years", used))
        pivot_area.to_excel(xw, index=False, sheet_name=excel_safe("wide_area_by_year", used))
        pivot_pixels.to_excel(xw, index=False, sheet_name=excel_safe("wide_pixels_by_year", used))
        pivot_percent_polygon.to_excel(xw, index=False, sheet_name=excel_safe("wide_percent_polygon", used))
        pair = pairwise_changes(pivot_area, years_sorted)
        pair.to_excel(xw, index=False, sheet_name=excel_safe("pairwise_changes_area", used))
        rrg_changes.to_excel(xw, index=False, sheet_name=excel_safe("RRG_encro_retreat", used))
        # quick counts + global tops
        rrg_quick_counts.to_excel(xw, index=False, sheet_name=excel_safe("RRG_quick_counts", used))
        rrg_top_enc.to_excel(xw, index=False, sheet_name=excel_safe("RRG_top_encroachment", used))
        rrg_top_ret.to_excel(xw, index=False, sheet_name=excel_safe("RRG_top_retreat", used))
        # consecutive pair tops
        for name, dfp in pair_top_sheets.items():
            dfp.to_excel(xw, index=False, sheet_name=excel_safe(name, used))
        # continuous/flip
        if not rrg_continuous_enc.empty:
            rrg_continuous_enc.to_excel(xw, index=False, sheet_name=excel_safe("RRG_continuous_encroachers", used))
        if not rrg_continuous_ret.empty:
            rrg_continuous_ret.to_excel(xw, index=False, sheet_name=excel_safe("RRG_continuous_retreaters", used))
        if not rrg_flip_enc_to_ret.empty:
            rrg_flip_enc_to_ret.to_excel(xw, index=False, sheet_name=excel_safe("RRG_flip_enc_to_ret", used))
        if not rrg_flip_ret_to_enc.empty:
            rrg_flip_ret_to_enc.to_excel(xw, index=False, sheet_name=excel_safe("RRG_flip_ret_to_enc", used))

    print(f"✅ Saved wetland time-series workbook: {args.out_xlsx}")

if __name__ == "__main__":
    main()
