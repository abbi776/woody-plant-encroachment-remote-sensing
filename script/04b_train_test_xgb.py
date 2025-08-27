#!/usr/bin/env python3
"""
Train & test XGBoost (multi-class) with RandomizedSearchCV.
Saves: best params + report CSV, confusion matrix PNG, feature importance PNG, and model .pkl
"""

import argparse, os, joblib
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    # ==== Load data ====
    train = pd.read_excel(args.train)
    valid = pd.read_excel(args.valid)

    X_train = train.drop(columns=["unique_id","classname","classvalue"])
    y_train = train["classvalue"]
    X_valid = valid.drop(columns=["unique_id","classname","classvalue"])
    y_valid = valid["classvalue"]

    # ==== Search space (tuned for small/medium tabular) ====
    param_dist = {
        "n_estimators":      [200, 300, 400, 600, 800],
        "max_depth":         [3, 4, 5, 6, 8],
        "learning_rate":     [0.03, 0.05, 0.07, 0.1],
        "subsample":         [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree":  [0.6, 0.7, 0.8, 1.0],
        "min_child_weight":  [1, 3, 5, 7],
        "gamma":             [0, 0.5, 1.0],
        "reg_lambda":        [1.0, 1.5, 2.0, 3.0],
        "reg_alpha":         [0.0, 0.1, 0.5],
    }

    print("⚡ Starting Randomized Search (XGBoost)…")
    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=len(np.unique(y_train)),
        eval_metric="mlogloss",
        tree_method="hist",   # change to "gpu_hist" if you have CUDA
        n_jobs=-1,
        random_state=42,
        verbosity=0,
    )

    search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_dist,
        n_iter=args.n_iter,
        cv=args.cv,
        n_jobs=-1,
        scoring="accuracy",
        verbose=1,
        random_state=42,
        error_score="raise",
    )
    search.fit(X_train, y_train)

    print("\n✅ Best Parameters:", search.best_params_)
    print("✅ Best CV Accuracy: {:.4f}".format(search.best_score_))
    clf = search.best_estimator_

    # ==== Validation metrics ====
    y_pred = clf.predict(X_valid)
    report_dict = classification_report(y_valid, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()

    # Save combined params + report
    params_df = pd.DataFrame.from_dict(search.best_params_, orient="index", columns=["Best Parameter Value"])
    params_df.loc["Best CV Accuracy"] = search.best_score_
    results_csv = os.path.join(args.out_dir, "xgb_results.csv")
    pd.concat([params_df, pd.DataFrame([[""]], index=[""], columns=["Best Parameter Value"]), report_df]) \
      .to_csv(results_csv)
    print(f"📄 Results saved to {results_csv}")

    # Confusion matrix
    cm = confusion_matrix(y_valid, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=np.unique(y_valid), yticklabels=np.unique(y_valid))
    plt.title("XGBoost – Confusion Matrix"); plt.xlabel("Predicted"); plt.ylabel("True")
    cm_path = os.path.join(args.out_dir, "confusion_matrix_xgb.png")
    plt.tight_layout(); plt.savefig(cm_path, dpi=300); plt.close()
    print(f"🖼️ Confusion matrix saved to {cm_path}")

    # Feature importances (gain-based)
    importances = clf.feature_importances_
    feat_names = X_train.columns.to_numpy()
    idx = np.argsort(importances)[-15:]
    top_feats, top_imps = feat_names[idx], importances[idx]

    plt.figure(figsize=(8,6))
    colors = plt.cm.Blues(np.linspace(0.4, 1, len(top_feats)))
    plt.barh(top_feats[::-1], top_imps[::-1], color=colors[::-1])
    plt.xlabel("Importance"); plt.title("XGBoost — Top 15 Features")
    imp_path = os.path.join(args.out_dir, "xgb_feature_importance.png")
    plt.tight_layout(); plt.savefig(imp_path, dpi=300); plt.close()
    print(f"🖼️ Feature importance saved to {imp_path}")

    # Save model
    model_path = os.path.join(args.model_dir, "best_xgb_model.pkl")
    joblib.dump(clf, model_path)
    print(f"💾 Model saved to {model_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="docs/tables/seasonal_features_train.xlsx")
    ap.add_argument("--valid", default="docs/tables/seasonal_features_test.xlsx")
    ap.add_argument("--out-dir", default="docs/results/xgb")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--n-iter", type=int, default=60, dest="n_iter")
    ap.add_argument("--cv", type=int, default=3)
    args = ap.parse_args()
    main(args)
