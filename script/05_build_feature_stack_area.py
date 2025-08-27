#!/usr/bin/env python3
"""
Mask seasonal rasters by wetland polygons → compute indices per season → add terrain → write a multi-band feature stack.

Output band names (as descriptions):
  <season>_<band>  (Blue..SWIR2)
  <season>_<index> (NDVI,EVI,NDWI,LSWI,MSAVI2,DBSI,GNDVI,TVI)
  slope[, elevation, curvature if provided]

Assumes EPSG:3577, reprojects if needed.
"""
import argparse, os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping
from collections import OrderedDict

REFERENCE_CRS = "EPSG:3577"
BANDS = ['nbart_blue','nbart_green','nbart_red',
         'nbart_red_edge_1','nbart_red_edge_2','nbart_red_edge_3',
         'nbart_nir_1','nbart_nir_2','nbart_swir_1','nbart_swir_2']
INDEX_NAMES = ["NDVI","EVI","NDWI","LSWI","MSAVI2","DBSI","GNDVI","TVI"]

def compute_scalar_indices_vec(b_stack):
    """
    b_stack: (10, H, W) in the order BANDS.
    Returns indices (8, H, W).
    """
    blue, green, red = b_stack[0], b_stack[1], b_stack[2]
    nir1, swir1, swir2 = b_stack[6], b_stack[8], b_stack[9]

    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = (nir1 - red) / (nir1 + red)
        evi  = 2.5 * (nir1 - red) / (nir1 + 6*red - 7.5*blue + 1)
        ndwi = (green - nir1) / (green + nir1)
        lswi = (nir1 - swir2) / (nir1 + swir2)
        arg  = (2*nir1 + 1)**2 - 8*(nir1 - red)
        msavi2 = np.where(arg >= 0, (2*nir1 + 1 - np.sqrt(arg))/2, np.nan)
        dbsi = ((swir1 - green) / (swir1 + green)) - ndvi
        gndvi = (nir1 - green) / (nir1 + green)
        tvi   = np.sqrt(ndvi + 0.5)

    return np.stack([ndvi, evi, ndwi, lswi, msavi2, dbsi, gndvi, tvi], axis=0)

def read_mask(raster_path, geoms):
    with rasterio.open(raster_path) as src:
        if src.crs and src.crs.to_string() != REFERENCE_CRS:
            raise ValueError(f"{raster_path} not in EPSG:3577. Reproject upstream.")
        out, transform = mask(src, geoms, crop=True)
        out = out.astype("float32")
        if src.nodata is not None:
            out[out == src.nodata] = np.nan
        profile = src.profile.copy()
        profile.update(transform=transform, height=out.shape[1], width=out.shape[2])
    return out, profile

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wetlands", required=True, help="Wetland polygons (shp/geojson) in EPSG:3577")
    ap.add_argument("--season", action="append", nargs=2, metavar=("NAME","TIF"),
                    help="Season name and path. Repeatable.")
    ap.add_argument("--slope", required=True, help="Slope raster")
    ap.add_argument("--elevation", default=None)
    ap.add_argument("--curvature", default=None)
    ap.add_argument("--out", required=True, help="Output multi-band feature stack .tif")
    args = ap.parse_args()

    gdf = gpd.read_file(args.wetlands)
    gdf = gdf.to_crs(REFERENCE_CRS)
    geoms = [mapping(geom) for geom in gdf.geometry]

    stacked = OrderedDict()
    profile_ref = None
    transform_ref = None

    # Seasons: raw bands + indices
    for season, tif in args.season or []:
        print(f"📥 {season}: {tif}")
        bands_stack, profile = read_mask(tif, geoms)
        if profile_ref is None:
            profile_ref = profile.copy()
            transform_ref = profile['transform']

        # raw bands
        for i, bname in enumerate(BANDS):
            stacked[f"{season}_{bname}"] = bands_stack[i]

        # indices
        idx_stack = compute_scalar_indices_vec(bands_stack)
        for i, iname in enumerate(INDEX_NAMES):
            stacked[f"{season}_{iname}"] = idx_stack[i]

    # Terrain rasters (optional elevation/curvature)
    for name, path in [("slope", args.slope), ("elevation", args.elevation), ("curvature", args.curvature)]:
        if path:
            print(f"📥 terrain: {name}: {path}")
            arr, _ = read_mask(path, geoms)
            stacked[name] = arr[0]

    # write
    print("💾 Writing feature stack...")
    sample = next(iter(stacked.values()))
    height, width = sample.shape
    profile_ref.update(
        height=height, width=width, transform=transform_ref,
        count=len(stacked), dtype="float32", compress="lzw", nodata=np.nan, tiled=True
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with rasterio.open(args.out, "w", **profile_ref) as dst:
        for i, (name, arr) in enumerate(stacked.items(), 1):
            dst.write(arr, i)
            dst.set_band_description(i, name)
    print(f"✅ Saved: {args.out}")

if __name__ == "__main__":
    main()
