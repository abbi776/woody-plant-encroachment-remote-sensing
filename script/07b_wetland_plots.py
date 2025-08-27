#!/usr/bin/env python3
"""
Wetland-level plots from 07_wetland_stats.py Excel output.
"""
import os, argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-wetlands", type=int, default=None, help="Limit plots for large sets")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df_long = pd.read_excel(args.excel, sheet_name="long_all_years")

    if df_long.empty:
        print("No data to plot."); return

    # Clean
    df_long["Year"] = pd.to_numeric(df_long["Year"], errors="coerce").astype("Int64")
    df_long = df_long.dropna(subset=["Year"]).copy()
    df_long["Year"] = df_long["Year"].astype(int)

    sns.set(style="whitegrid"); plt.rcParams["figure.dpi"] = 120

    wetlands = df_long["WetlandID"].unique().tolist()
    if args.max_wetlands: wetlands = wetlands[:args.max_wetlands]

    # 1) Line chart per wetland (area_ha by class over years)
    for wid in wetlands:
        g = df_long[df_long["WetlandID"] == wid].sort_values(["class_label","Year"])
        plt.figure(figsize=(8,5))
        sns.lineplot(data=g, x="Year", y="area_ha", hue="class_label", marker="o")
        plt.title(f"{wid} – Class Areas Over Time"); plt.xlabel("Year"); plt.ylabel("Area (ha)")
        plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, f"{wid}_timeseries_by_class.png")); plt.close()

    # 2) Heatmap (class × year) per wetland
    for wid in wetlands:
        g = df_long[df_long["WetlandID"] == wid]
        mat = (g.pivot_table(index="class_label", columns="Year", values="area_ha", aggfunc="sum").sort_index())
        if mat.empty: continue
        plt.figure(figsize=(7, 3 + 0.25*len(mat.index)))
        sns.heatmap(mat, annot=True, fmt=".1f", linewidths=0.5, cbar_kws={"label": "Area (ha)"})
        plt.title(f"{wid} – Area by Class and Year"); plt.xlabel("Year"); plt.ylabel("Class")
        plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, f"{wid}_heatmap_area.png")); plt.close()

    print("✅ Wetland plots saved to:", args.out_dir)

if __name__ == "__main__":
    main()
