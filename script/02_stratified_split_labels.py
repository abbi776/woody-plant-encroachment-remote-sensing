#!/usr/bin/env python3
"""
Stratified 70/30 split of labeled polygons by ROI (Poly_num) and class (classname).

Example:
  python scripts/02_stratified_split_labels.py \
    --input /path/to/ROI_labels_525_samples_uniqueIDs.shp \
    --class-col classname \
    --roi-col Poly_num \
    --train-out data/processed/training_labels.shp \
    --test-out  data/processed/testing_labels.shp \
    --summary-out docs/tables/label_split_summary.csv \
    --train-frac 0.7 \
    --seed 42
"""

import argparse
import sys
import pandas as pd
import geopandas as gpd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input labeled shapefile")
    ap.add_argument("--class-col", default="classname", help="Class column name")
    ap.add_argument("--roi-col", default="Poly_num", help="ROI column name")
    ap.add_argument("--id-col", default="unique_id", help="Unique ID column")
    ap.add_argument("--train-out", required=True, help="Output shapefile for training set")
    ap.add_argument("--test-out", required=True, help="Output shapefile for test set")
    ap.add_argument("--summary-out", default=None, help="Optional CSV with per-group counts")
    ap.add_argument("--train-frac", type=float, default=0.7, help="Train fraction (0–1)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--expect-per-group", type=int, default=None,
                    help="Optional: assert each ROI×class group has this many samples (e.g., 35)")
    args = ap.parse_args()

    # Load
    gdf = gpd.read_file(args.input)
    for col in (args.id_col, args.roi_col, args.class_col):
        if col not in gdf.columns:
            sys.exit(f"ERROR: Missing required column '{col}' in {args.input}")

    # Optional strict size check
    if args.expect_per_group is not None:
        bad = (
            gdf.groupby([args.roi_col, args.class_col])[args.id_col]
            .count()
            .reset_index(name="n")
        )
        bad = bad[bad["n"] != args.expect_per_group]
        if not bad.empty:
            sys.exit(
                "ERROR: Some ROI×class groups do not match expected size:\n"
                + bad.to_string(index=False)
            )

    train_parts, test_parts, rows = [], [], []
    # Stratified by ROI × class
    for (roi, cls), subset in gdf.groupby([args.roi_col, args.class_col]):
        # Shuffle
        subset = subset.sample(frac=1.0, random_state=args.seed)
        n = len(subset)
        n_train = max(1, int(round(args.train_frac * n)))
        train = subset.iloc[:n_train]
        test = subset.iloc[n_train:]

        train_parts.append(train)
        test_parts.append(test)

        rows.append({
            "ROI": roi,
            "class": cls,
            "total": n,
            "train": len(train),
            "test": len(test)
        })

    train_gdf = gpd.GeoDataFrame(pd.concat(train_parts, ignore_index=True), crs=gdf.crs)
    test_gdf  = gpd.GeoDataFrame(pd.concat(test_parts,  ignore_index=True), crs=gdf.crs)

    # Save shapefiles
    train_gdf.to_file(args.train_out)
    test_gdf.to_file(args.test_out)

    # Summary CSV (optional)
    if args.summary_out:
        pd.DataFrame(rows).sort_values(["ROI", "class"]).to_csv(args.summary_out, index=False)

    print("✅ Stratified split complete")
    print(f"Training samples: {len(train_gdf)}")
    print(f"Testing samples : {len(test_gdf)}")
    if args.summary_out:
        print(f"Summary written: {args.summary_out}")

if __name__ == "__main__":
    main()
