#!/usr/bin/env python3
"""
Plot a classified raster (0=RRG, 1=NFFP, 2=Water) with a legend.
"""
import argparse, rasterio, numpy as np, matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm

CLASS_COLORS = {0: "red", 1: "green", 2: "blue"}  # RRG red per your figure style

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raster", required=True)
    ap.add_argument("--title", default="Classified Wetlands Map")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with rasterio.open(args.raster) as src:
        img = src.read(1).astype("float32")
        if src.nodata is not None:
            img[img == src.nodata] = np.nan

    cmap = ListedColormap([CLASS_COLORS[i] for i in sorted(CLASS_COLORS)])
    bounds = [-0.5, 0.5, 1.5, 2.5]  # classes 0,1,2
    norm = BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap=cmap, norm=norm)
    plt.title(args.title); plt.axis("off")

    legend = [
        mpatches.Patch(color=CLASS_COLORS[0], label="River Red Gum"),
        mpatches.Patch(color=CLASS_COLORS[1], label="Non-Forest Floodplain"),
        mpatches.Patch(color=CLASS_COLORS[2], label="Water"),
    ]
    plt.legend(handles=legend, loc="lower right")
    if args.out:
        plt.tight_layout(); plt.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()
