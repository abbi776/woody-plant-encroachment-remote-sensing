#!/usr/bin/env python3
# Create panels A–D as separate figures with matched dimensions.
import os, re, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def natural_key(s):
    parts = re.split(r'(\d+)', str(s))
    return [int(p) if p.isdigit() else p for p in parts]

def require_cols(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}\nHave: {list(df.columns)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True,
                    help="CSV with per-ROI areas for RRG and NFFP (both RF and XGB).")
    ap.add_argument("--out-dir", default="figures",
                    help="Output directory for panel figures.")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ---------- Load & prep ----------
    df = pd.read_csv(args.csv)
    df = df[df["Class"].isin(["River Red Gum", "Non-Forest Floodplain"])].copy()

    years = [2016, 2018, 2025]
    num_cols = [f"{y} (XGB)" for y in years] + [f"{y} (RF)" for y in years]
    require_cols(df, ["ROI", "Class"] + num_cols)

    # Coerce numerics (robust)
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Compute % change if missing
    if "% Δ 2016–2025 (XGB)" not in df.columns:
        df["% Δ 2016–2025 (XGB)"] = (df["2025 (XGB)"] - df["2016 (XGB)"]) / df["2016 (XGB)"] * 100.0
    if "% Δ 2016–2025 (RF)" not in df.columns:
        df["% Δ 2016–2025 (RF)"]  = (df["2025 (RF)"]  - df["2016 (RF)"])  / df["2016 (RF)"]  * 100.0

    df.sort_values(["ROI", "Class"], key=lambda s: s.map(natural_key), inplace=True)

    def ycols(model): return [f"{y} ({model})" for y in years]
    rois = sorted(df["ROI"].unique(), key=natural_key)

    # ---------- Styling ----------
    # Force consistent colors for models across all panels
    COLOR_XGB = "#1f77b4"  # blue
    COLOR_RF  = "#ff7f0e"  # orange

    xtick_labels_2yrs = ["2016", "2018", "2024-2025"]

    FIGSIZE_AB = (7.6, 4.8)   # A & B identical
    FIGSIZE_CD = (7.0, 4.8)   # C & D identical
    RIGHT_SPACE = 0.80
    LEGEND_KW = dict(frameon=True, edgecolor="black", fontsize=8,
                     loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, ncol=1)

    # ---------------- Panel A: RRG per ROI (lines) ----------------
    figA, axA = plt.subplots(figsize=FIGSIZE_AB)
    figA.subplots_adjust(right=RIGHT_SPACE, left=0.10, bottom=0.15, top=0.88)
    for roi in rois:
        s = df[(df["ROI"] == roi) & (df["Class"] == "River Red Gum")]
        if s.empty: continue
        axA.plot(years, s[ycols("XGB")].values.flatten(), marker="o", color=COLOR_XGB, label=f"{roi} (XGBoost)")
        axA.plot(years, s[ycols("RF")].values.flatten(),  marker="o", linestyle="--", color=COLOR_RF, label=f"{roi} (RF)")
    axA.set_title("(a) RRG — per Sub-region (XGBoost & RF)")
    axA.set_xlabel("Year"); axA.set_ylabel("Area (ha)")
    axA.set_xticks(years); axA.set_xticklabels(xtick_labels_2yrs); axA.legend(**LEGEND_KW)
    for ext in ("png","pdf","svg"):
        figA.savefig(os.path.join(args.out_dir, f"panel_A_RRG_per_ROI.{ext}"), dpi=300)
    plt.close(figA)

    # ---------------- Panel B: NFFP per ROI (lines) ----------------
    figB, axB = plt.subplots(figsize=FIGSIZE_AB)
    figB.subplots_adjust(right=RIGHT_SPACE, left=0.10, bottom=0.15, top=0.88)
    for roi in rois:
        s = df[(df["ROI"] == roi) & (df["Class"] == "Non-Forest Floodplain")]
        if s.empty: continue
        axB.plot(years, s[ycols("XGB")].values.flatten(), marker="s", color=COLOR_XGB, label=f"{roi} (XGBoost)")
        axB.plot(years, s[ycols("RF")].values.flatten(),  marker="s", linestyle="--", color=COLOR_RF,  label=f"{roi} (RF)")
    axB.set_title("(b) NFFP — per Sub-region (XGBoost & RF)")
    axB.set_xlabel("Year"); axB.set_ylabel("Area (ha)")
    axB.set_xticks(years); axB.set_xticklabels(xtick_labels_2yrs); axB.legend(**LEGEND_KW)
    for ext in ("png","pdf","svg"):
        figB.savefig(os.path.join(args.out_dir, f"panel_B_NFFP_per_ROI.{ext}"), dpi=300)
    plt.close(figB)

    # ---------------- Panel C: % change bars ----------------
    # Build matrix safely (skip ROIs without both classes)
    rows = []
    for roi in rois:
        rrg = df[(df["ROI"]==roi) & (df["Class"]=="River Red Gum")]
        nff = df[(df["ROI"]==roi) & (df["Class"]=="Non-Forest Floodplain")]
        if rrg.empty or nff.empty: continue
        rows.append([
            float(rrg["% Δ 2016–2025 (XGB)"].values[0]),
            float(rrg["% Δ 2016–2025 (RF)"].values[0]),
            float(nff["% Δ 2016–2025 (XGB)"].values[0]),
            float(nff["% Δ 2016–2025 (RF)"].values[0]),
            roi,
        ])
    if rows:
        B = np.array(rows, dtype=float)
        rois_used = [r[-1] for r in rows]
        x = np.arange(len(rois_used)); w = 0.2

        figC, axC = plt.subplots(figsize=FIGSIZE_CD)
        axC.bar(x - 1.5*w, B[:,0], width=w, label="RRG (XGBoost)", color=COLOR_XGB)
        axC.bar(x - 0.5*w, B[:,1], width=w, label="RRG (RF)",      color=COLOR_RF)
        axC.bar(x + 0.5*w, B[:,2], width=w, label="NFFP (XGBoost)", color=COLOR_XGB, alpha=0.5)
        axC.bar(x + 1.5*w, B[:,3], width=w, label="NFFP (RF)",      color=COLOR_RF,  alpha=0.5)
        axC.axhline(0, linewidth=1, color="k")
        axC.set_title("(c) % Change 2016–2025 by Sub-regions")
        axC.set_xlabel("Sub-regions"); axC.set_ylabel("% Change")
        axC.set_xticks(x, rois_used); axC.legend(ncol=2, frameon=True, edgecolor="black", fontsize=9, loc="best")
        figC.tight_layout()
        for ext in ("png","pdf","svg"):
            figC.savefig(os.path.join(args.out_dir, f"panel_C_pct_change.{ext}"), dpi=300)
        plt.close(figC)

    # ---------------- Panel D: scatter of % change ----------------
    def pct_pair(model):
        rrg = df[df["Class"]=="River Red Gum"][["ROI", f"% Δ 2016–2025 ({model})"]].rename(
            columns={f"% Δ 2016–2025 ({model})":"RRG_pct"})
        nff = df[df["Class"]=="Non-Forest Floodplain"][["ROI", f"% Δ 2016–2025 ({model})"]].rename(
            columns={f"% Δ 2016–2025 ({model})":"NFF_pct"})
        return pd.merge(rrg, nff, on="ROI")

    C_xgb, C_rf = pct_pair("XGB"), pct_pair("RF")

    figD, axD = plt.subplots(figsize=FIGSIZE_CD)
    axD.scatter(C_xgb["NFF_pct"], C_xgb["RRG_pct"], s=60, marker="o", color=COLOR_XGB, label="XGBoost")
    axD.scatter(C_rf["NFF_pct"],  C_rf["RRG_pct"],  s=60, marker="^", color=COLOR_RF,  label="RF")

    # ROI labels (offset a bit)
    for _, r in C_xgb.iterrows():
        axD.text(r["NFF_pct"]+0.3, r["RRG_pct"], r["ROI"], fontsize=9, color=COLOR_XGB)
    for _, r in C_rf.iterrows():
        axD.text(r["NFF_pct"]+0.3, r["RRG_pct"], r["ROI"], fontsize=9, color=COLOR_RF)

    axD.axhline(0, linestyle="--", linewidth=1, color="k")
    axD.axvline(0, linestyle="--", linewidth=1, color="k")
    axD.set_title("(d) RRG vs NFFP % Change")
    axD.set_xlabel("NFFP % Change (2016–2025)")
    axD.set_ylabel("RRG % Change (2016–2025)")
    axD.legend(frameon=True, edgecolor="black", fontsize=9, loc="best")
    figD.tight_layout()
    for ext in ("png","pdf","svg"):
        figD.savefig(os.path.join(args.out_dir, f"panel_D_scatter.{ext}"), dpi=300)
    plt.close(figD)

    print("✅ Saved panels to:", os.path.abspath(args.out_dir))

if __name__ == "__main__":
    main()
