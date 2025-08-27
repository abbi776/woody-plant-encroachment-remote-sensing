#!/usr/bin/env python3
"""
Create per-ROI time series, grouped bars, stacked area, consecutive deltas,
and heatmaps from the Excel produced by 07_area_stats.py.
"""

import os, argparse
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", required=True, help="Path to wetland_class_areas_timeseries.xlsx")
    ap.add_argument("--out-dir", required=True, help="Directory to save plots")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df_long = pd.read_excel(args.excel, sheet_name="areas_long")
    df_wide = pd.read_excel(args.excel, sheet_name="areas_wide")
    df_pair  = pd.read_excel(args.excel, sheet_name="pairwise_consecutive")

    if df_long.empty:
        print("No data in areas_long; nothing to plot."); return

    # Clean types
    df_long["year"] = pd.to_numeric(df_long["year"], errors="coerce").astype("Int64")
    df_long = df_long.dropna(subset=["year"]).copy()
    df_long["year"] = df_long["year"].astype(int)

    sns.set(style="whitegrid")
    plt.rcParams["figure.dpi"] = 120

    # 1) Line charts per ROI
    for roi, g in df_long.groupby("ROI"):
        g = g.sort_values(["class_label","year"])
        plt.figure(figsize=(8,5))
        sns.lineplot(data=g, x="year", y="area_ha", hue="class_label", marker="o")
        plt.title(f"{roi} – Class Areas Over Time")
        plt.xlabel("Year"); plt.ylabel("Area (ha)")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f"{roi}_timeseries_by_class.png"))
        plt.close()

    # 1b) Grouped bars per ROI
    for roi, g in df_long.groupby("ROI"):
        g = g.sort_values(["class_label","year"]).copy()
        class_order = sorted(g["class_label"].unique())
        year_order = sorted(g["year"].unique())
        g["year"] = pd.Categorical(g["year"], categories=year_order, ordered=True)
        plt.figure(figsize=(9,5.5))
        sns.barplot(data=g, x="class_label", y="area_ha", hue="year", order=class_order)
        plt.title(f"{roi} – Class Area Comparison ({year_order[0]}–{year_order[-1]})")
        plt.ylabel("Area (ha)"); plt.xlabel("Class"); plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f"{roi}_bar_comparison_multi_year.png"))
        plt.close()

    # 2) Stacked area per ROI
    for roi, g in df_long.groupby("ROI"):
        mat = (g.pivot_table(index="year", columns="class_label", values="area_ha", aggfunc="sum").sort_index())
        if mat.empty: continue
        plt.figure(figsize=(8,5))
        mat = mat.reindex(sorted(mat.columns), axis=1)
        plt.stackplot(mat.index, mat.T.values, labels=mat.columns)
        plt.title(f"{roi} – Stacked Class Areas Over Time")
        plt.xlabel("Year"); plt.ylabel("Area (ha)")
        plt.legend(loc="upper left", bbox_to_anchor=(1.02,1), borderaxespad=0.)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f"{roi}_stacked_area.png"))
        plt.close()

    # 3) Consecutive deltas per ROI × class
    if not df_pair.empty:
        if df_pair["percent_change"].dtype == object:
            df_pair["percent_change_num"] = pd.to_numeric(df_pair["percent_change"], errors="coerce")
        else:
            df_pair["percent_change_num"] = df_pair["percent_change"]

        for roi, g in df_pair.groupby("ROI"):
            g = g.copy(); g["interval"] = g["year_from"].astype(str)+"→"+g["year_to"].astype(str)
            for cls_label, gc in g.groupby("class_label"):
                gc = gc.sort_values("year_from")
                plt.figure(figsize=(8,5))
                order = list(gc["interval"])
                sns.barplot(data=gc, x="interval", y="change_ha", order=order)
                plt.axhline(0, linewidth=1, color="black")
                plt.title(f"{roi} – Change in {cls_label} (Consecutive Years)")
                plt.xlabel("Interval"); plt.ylabel("Change (ha)")
                for i, row in enumerate(gc.itertuples()):
                    val, pct = row.change_ha, row.percent_change_num
                    txt = f"{val:.1f} ha" + (f"\n({pct:.1f}%)" if pd.notna(pct) else "")
                    ytxt = val + (0.02*np.sign(val) if val != 0 else 0.02)
                    plt.text(i, ytxt, txt, ha="center",
                             va="bottom" if val >= 0 else "top", fontsize=8)
                plt.tight_layout()
                fn = f"{roi}_delta_{cls_label.replace(' ','_')}.png"
                plt.savefig(os.path.join(args.out_dir, fn)); plt.close()

    # 4) Heatmap per ROI
    for roi, g in df_long.groupby("ROI"):
        years_sorted = sorted(g["year"].unique())
        classes_sorted = sorted(g["class_label"].unique())
        mat = (g.pivot_table(index="class_label", columns="year", values="area_ha", aggfunc="sum")
                 .reindex(classes_sorted).reindex(years_sorted, axis=1))
        if mat.empty: continue
        plt.figure(figsize=(8, 4 + 0.2*len(mat.index)))
        sns.heatmap(mat, annot=True, fmt=".1f", linewidths=0.5, cbar_kws={"label":"Area (ha)"})
        plt.title(f"{roi} – Area by Class and Year"); plt.xlabel("Year"); plt.ylabel("Class")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f"{roi}_heatmap_area.png")); plt.close()

    print("✅ Plots saved to:", args.out_dir)

if __name__ == "__main__":
    main()
