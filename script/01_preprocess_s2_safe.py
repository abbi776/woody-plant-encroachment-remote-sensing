#!/usr/bin/env python3
"""
Preprocess Sentinel-2 SAFE.zip: unzip → extract bands → resample (10m) → stack
Usage:
    python scripts/01_preprocess_s2_safe.py --zip_file <path> --output <tif>
"""

import os
import zipfile
import argparse
from glob import glob
import rasterio
from osgeo import gdal

# ---------------------- FUNCTIONS ----------------------

def unzip_safe(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    safe_dirs = [f.path for f in os.scandir(extract_to) if f.name.endswith('.SAFE')]
    if not safe_dirs:
        raise FileNotFoundError("No .SAFE directory found after extraction.")
    return safe_dirs[0]

def get_band_paths(safe_dir, bands):
    jp2_files = glob(os.path.join(safe_dir, '**', '*.jp2'), recursive=True)
    band_paths = {}
    for band in bands:
        for file in jp2_files:
            fname = os.path.basename(file)
            if f'B{band}' in fname and ('R10m' in file or 'R20m' in file or 'R60m' in file):
                band_paths[band] = file
                break
    return band_paths

def resample_to_10m(src_path, target_res=10):
    ds = gdal.Open(src_path)
    resampled_path = src_path.replace('.jp2', f'_10m.tif')
    gdal.Warp(
        resampled_path, ds,
        xRes=target_res, yRes=target_res,
        resampleAlg='near', format='GTiff'
    )
    return resampled_path

def stack_bands(band_files, bands_order, output_tif):
    band_name_map = {
        '02': 'B02 (Blue)',
        '03': 'B03 (Green)',
        '04': 'B04 (Red)',
        '05': 'B05 (Red Edge 1)',
        '06': 'B06 (Red Edge 2)',
        '07': 'B07 (Red Edge 3)',
        '08': 'B08 (NIR 1)',
        '8A': 'B8A (NIR 2)',
        '11': 'B11 (SWIR 1)',
        '12': 'B12 (SWIR 2)'
    }
    datasets = [rasterio.open(band_files[band]) for band in bands_order]
    meta = datasets[0].meta.copy()
    meta.update(count=len(datasets))

    with rasterio.open(output_tif, 'w', **meta) as dst:
        for idx, (band, src) in enumerate(zip(bands_order, datasets), start=1):
            dst.write(src.read(1), idx)
            dst.set_band_description(idx, band_name_map.get(band, f'Band {band}'))

    print(f"✅ Saved stacked TIFF: {output_tif}")

# ---------------------- MAIN ----------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip_file", required=True, help="Path to .SAFE.zip")
    parser.add_argument("--output", required=True, help="Output stacked .tif")
    parser.add_argument("--workdir", default="data/interim", help="Where to unzip SAFE files")
    args = parser.parse_args()

    bands_to_extract = ['02','03','04','05','06','07','08','8A','11','12']

    # 1) Unzip
    safe_dir = unzip_safe(args.zip_file, args.workdir)

    # 2) Get band paths
    band_paths = get_band_paths(safe_dir, bands_to_extract)

    # 3) Resample to 10m
    resampled_band_paths = {b: resample_to_10m(band_paths[b]) for b in bands_to_extract}

    # 4) Stack into single TIFF
    stack_bands(resampled_band_paths, bands_to_extract, args.output)
