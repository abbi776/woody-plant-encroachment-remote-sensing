#!/usr/bin/env python3
"""
Apply a trained XGBoost model (.pkl) to a multi-band feature raster and save a classified map.
Assumes the raster band order matches the training feature order.
"""

import argparse, os, joblib
import numpy as np, rasterio

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="Feature stack .tif")
    ap.add_argument("--model", required=True, help="XGB .pkl from training")
    ap.add_argument("--out", required=True, help="Output classified .tif")
    ap.add_argument("--nodata", type=int, default=-1)
    args = ap.parse_args()

    print("🧠 Loading model…")
    clf = joblib.load(args.model)

    print("📊 Reading features…")
    with rasterio.open(args.features) as src:
        prof = src.profile.copy()
        arr = src.read().astype("float32")    # (bands, H, W)
        b, h, w = arr.shape
        nodata_val = src.nodata

    if nodata_val is not None and not np.isnan(nodata_val):
        arr[arr == nodata_val] = np.nan
    flat = arr.reshape(b, -1).T  # (pixels, bands)

    valid_mask = np.all(np.isfinite(flat), axis=1)
    X = flat[valid_mask]
    print(f"✅ Valid pixels: {X.shape[0]} / {flat.shape[0]}")

    y = np.full(flat.shape[0], args.nodata, dtype="int16")
    if X.size > 0:
        y_pred = clf.predict(X)
        y[valid_mask] = y_pred.astype("int16")

    out_img = y.reshape(h, w)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    prof.update(count=1, dtype="int16", compress="lzw", nodata=args.nodata)
    with rasterio.open(args.out, "w", **prof) as dst:
        dst.write(out_img, 1)
    print(f"✅ Saved classified map: {args.out}")

if __name__ == "__main__":
    main()
