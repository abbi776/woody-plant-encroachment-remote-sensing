#!/usr/bin/env python3
"""
Extract seasonal per-polygon features for the TRAIN set.

- Reads seasonal multi-band TIFFs (expected order: Blue, Green, Red, RE1, RE2, RE3, NIR1, NIR2, SWIR1, SWIR2)
- Reprojects rasters to EPSG:3577 on the fly if needed
- Computes band means inside polygons + indices (NDVI, EVI, NDWI, LSWI, MSAVI2, DBSI, GNDVI, TVI)
- Adds terrain vars (elevation, slope, curvature)
- Saves CSV/XLSX

Usage (defaults provided; override as needed):
  python scripts/03_extract_seasonal_features_train.py \
    --labels data/processed/training_labels.shp \
    --season spring data/processed/spring_2024_combined_ROI.tif \
    --season summer data/processed/summer_2024_combined_ROI.tif \
    --season autumn data/processed/autumn_2025_combined_ROI.tif \
    --season winter data/processed/winter_2025_combined_ROI.tif \
    --elevation data/raw/ROI_DEM_10m.tif \
    --slope data/raw/ROI_slope_10m.tif \
    --curvature data/raw/ROI_curvature_10m.tif \
    --out docs/tables/seasonal_features_train.xlsx
"""
import argparse, os, sys
import numpy as np, pandas as pd, geopandas as gpd
import rasterio, rasterio.mask
from rasterio.io import MemoryFile
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import mapping

REFERENCE_CRS = "EPSG:3577"
BAND_KEYS = [
    'nbart_blue','nbart_green','nbart_red',
    'nbart_red_edge_1','nbart_red_edge_2','nbart_red_edge_3',
    'nbart_nir_1','nbart_nir_2','nbart_swir_1','nbart_swir_2'
]

def reproject_to_3577(src):
    if src.crs and src.crs.to_string() == REFERENCE_CRS:
        return src
    if src.crs is None:
        raise ValueError("Source raster has no CRS set.")
    dst_crs = REFERENCE_CRS
    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds
    )
    kwargs = src.meta.copy()
    kwargs.update({'crs': dst_crs, 'transform': transform, 'width': width, 'height': height})
    memfile = MemoryFile()
    dst = memfile.open(**kwargs)
    for i in range(1, src.count + 1):
        reproject(
            source=rasterio.band(src, i),
            destination=rasterio.band(dst, i),
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear
        )
    return dst

def compute_scalar_indices(b):
    blue, green, red = b["nbart_blue"], b["nbart_green"], b["nbart_red"]
    nir1, swir1, swir2 = b["nbart_nir_1"], b["nbart_swir_1"], b["nbart_swir_2"]
    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi  = (nir1 - red) / (nir1 + red)
        evi   = 2.5 * (nir1 - red) / (nir1 + 6*red - 7.5*blue + 1)
        ndwi  = (green - nir1) / (green + nir1)
        lswi  = (nir1 - swir2) / (nir1 + swir2)
        arg   = (2*nir1 + 1)**2 - 8*(nir1 - red)
        msavi2= np.where(arg >= 0, (2*nir1 + 1 - np.sqrt(arg))/2, np.nan)
        dbsi  = ((swir1 - green)/(swir1 + green)) - ndvi
        gndvi = (nir1 - green) / (nir1 + green)
        tvi   = np.sqrt(ndvi + 0.5)
    to_f = lambda x: float(np.asarray(x))
    return {"NDVI":to_f(ndvi),"EVI":to_f(evi),"NDWI":to_f(ndwi),
            "LSWI":to_f(lswi),"MSAVI2":to_f(msavi2),"DBSI":to_f(dbsi),
            "GNDVI":to_f(gndvi),"TVI":to_f(tvi)}

def zonal_mean(src, geom):
    out, _ = rasterio.mask.mask(src, [geom], crop=True)
    arr = out.astype("float32")
    nodata = src.nodata
    if nodata is not None:
        arr[arr == nodata] = np.nan
    means = np.nanmean(arr, axis=(1, 2))
    return None if np.all(np.isnan(means)) else means

def extract_features(labels_path, seasons, terrain, out_path, id_col, class_col, classval_col, valid_frac=0.7):
    gdf = gpd.read_file(labels_path).to_crs(REFERENCE_CRS)
    rows = {}
    # Seasons
    for season_name, tif_path in seasons.items():
        if not os.path.exists(tif_path):
            print(f"⚠️ Missing raster: {tif_path}"); continue
        with rasterio.open(tif_path) as src0:
            src = reproject_to_3577(src0)
            for _, r in gdf.iterrows():
                uid, geom = r[id_col], mapping(r.geometry)
                try:
                    means = zonal_mean(src, geom)
                    if means is None: 
                        continue
                    # require some valid data
                    if np.isfinite(means).sum() < valid_frac * len(means):
                        continue
                    band_dict = {}
                    for i, key in enumerate(BAND_KEYS, start=1):
                        band_dict[key] = means[i-1] if i <= src.count else np.nan
                    idx = compute_scalar_indices(band_dict)
                    if uid not in rows:
                        rows[uid] = {id_col: uid, class_col: r[class_col], classval_col: r[classval_col]}
                    for k, v in band_dict.items():
                        rows[uid][f"{season_name}_{k}"] = v
                    for k, v in idx.items():
                        rows[uid][f"{season_name}_{k}"] = v
                except Exception as e:
                    print(f"❌ {uid} / {season_name}: {e}")
            if src is not src0: src.close()
    # Terrain
    for name, tif_path in terrain.items():
        if not tif_path or not os.path.exists(tif_path): 
            continue
        with rasterio.open(tif_path) as src0:
            src = reproject_to_3577(src0)
            for _, r in gdf.iterrows():
                uid, geom = r[id_col], mapping(r.geometry)
                try:
                    m = zonal_mean(src, geom)
                    if m is not None and uid in rows:
                        rows[uid][name] = float(np.nanmean(m))
                except Exception as e:
                    print(f"⚠️ Terrain {name} / {uid}: {e}")
            if src is not src0: src.close()
    df = pd.DataFrame.from_dict(rows, orient="index").reset_index(drop=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ext = os.path.splitext(out_path)[1].lower()
    if ext in (".xlsx",".xls"):
        try:
            import openpyxl  # noqa
            df.to_excel(out_path, index=False)
        except Exception:
            print("⚠️ Excel writer missing; saving CSV instead.")
            df.to_csv(os.path.splitext(out_path)[0]+".csv", index=False)
    else:
        df.to_csv(out_path, index=False)
    print(f"✅ Saved features: {out_path}  (rows={len(df)}, cols={len(df.columns)})")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/processed/training_labels.shp")
    ap.add_argument("--id-col", default="unique_id")
    ap.add_argument("--class-col", default="classname")
    ap.add_argument("--classval-col", default="classvalue")
    ap.add_argument("--season", action="append", nargs=2, metavar=("NAME","TIF"))
    ap.add_argument("--elevation", default="data/raw/ROI_DEM_10m.tif")
    ap.add_argument("--slope", default="data/raw/ROI_slope_10m.tif")
    ap.add_argument("--curvature", default="data/raw/ROI_curvature_10m.tif")
    ap.add_argument("--out", default="docs/tables/seasonal_features_train.xlsx")
    ap.add_argument("--valid-frac", type=float, default=0.7)
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if not args.season:
        sys.exit("ERROR: Provide at least one --season <name> <tif> pair.")
    seasons = {name: path for name, path in args.season}
    terrain = {"elevation": args.elevation, "slope": args.slope, "curvature": args.curvature}
    extract_features(args.labels, seasons, terrain, args.out, args.id_col, args.class_col, args.classval_col, args.valid_frac)
