#!/usr/bin/env python3
"""
Train a Random Forest classifier with grid search + CV,
evaluate on validation set, save reports/plots/model.
"""

import argparse, os, joblib
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

def main(args):
    # ==== Load Data ====
    train = pd.read_excel(args.train)
    valid = pd.read_excel(args.valid)

    X_train = train.drop(columns=["unique_id","classname","classvalue"])
    y_train = train["classvalue"]
    X_valid = valid.drop(columns=["unique_id","classname","classvalue"])
    y_valid = valid["classvalue"]

    # ==== Define Parameter Grid ====
    param_grid = {
        'n_estimators': [100,200,300],
        'max_depth': [None,10,20,30],
        'min_samples_split': [2,5,10],
        'min_samples_leaf': [1,2,4],
        'max_features': ['sqrt','log2']
    }

    # ==== Grid Search ====
    print("🔍 Starting Grid Search...")
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring='accuracy',
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    # ==== Best Estimator ====
    print("\n✅ Best Parameters:", grid_search.best_params_)
    print("✅ Best CV Accuracy: {:.4f}".format(grid_search.best_score_))
    clf = grid_search.best_estimator_

    # ==== Evaluate on Validation Set ====
    y_pred = clf.predict(X_valid)
    report_dict = classification_report(y_valid, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()

    # ==== Confusion Matrix ====
    cm = confusion_matrix(y_valid, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Random Forest - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    os.makedirs(args.out_dir, exist_ok=True)
    cm_path = os.path.join(args.out_dir,"confusion_matrix_rf.png")
    plt.tight_layout(); plt.savefig(cm_path,dpi=300); plt.close()
    print(f"🖼️ Confusion matrix saved to {cm_path}")

    # ==== Feature Importance Plot ====
    importances = clf.feature_importances_
    feat_names = X_train.columns
    idx_sorted = np.argsort(importances)[-15:]
    top_feats, top_imps = feat_names[idx_sorted], importances[idx_sorted]

    colors = plt.cm.Blues(np.linspace(0.4,1,len(top_feats)))
    plt.figure(figsize=(8,6))
    plt.barh(top_feats, top_imps, color=colors)
    plt.xlabel("Importance")
    plt.title("Top 15 Most Important Features")
    plt.tight_layout()
    imp_path = os.path.join(args.out_dir,"rf_feature_importance.png")
    plt.savefig(imp_path,dpi=300); plt.close()
    print(f"🖼️ Feature importance plot saved to {imp_path}")

    # ==== Save Combined Report ====
    best_params_df = pd.DataFrame.from_dict(
        grid_search.best_params_, orient='index', columns=['Best Parameter Value']
    )
    best_score_df = pd.DataFrame({'Best Parameter Value':[grid_search.best_score_]}, index=['Best CV Score'])
    params_df = pd.concat([best_params_df,best_score_df])
    final_df = pd.concat([params_df, report_df])
    csv_path = os.path.join(args.out_dir,"rf_results.csv")
    final_df.to_csv(csv_path)
    print(f"📄 Results saved to {csv_path}")

    # ==== Save Model ====
    model_path = os.path.join(args.model_dir,"best_rf_model.pkl")
    os.makedirs(args.model_dir, exist_ok=True)
    joblib.dump(clf, model_path)
    print(f"💾 Model saved to {model_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="docs/tables/seasonal_features_train.xlsx")
    ap.add_argument("--valid", default="docs/tables/seasonal_features_valid.xlsx")
    ap.add_argument("--out-dir", default="docs/results/rf")
    ap.add_argument("--model-dir", default="models")
    args = ap.parse_args()
    main(args)
