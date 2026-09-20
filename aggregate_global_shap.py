import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

PROJECT = Path(".")
RESULTS = PROJECT / "results"
MODEL_DIR = PROJECT / "models" / "final_t20_suitability_model"
XAI_DIR = RESULTS / "phase10_xai_fairness" / "shap"

DATA_FILE = RESULTS / "phase7_player_match_features_split.csv"
MODEL_FILE = MODEL_DIR / "final_hist_gradient_boosting.pkl"
PREPROCESSOR_FILE = MODEL_DIR / "final_preprocessor.pkl"
FEATURE_DEF_FILE = MODEL_DIR / "feature_definition.pkl"

with open(FEATURE_DEF_FILE, "rb") as f:
    feature_definition = pickle.load(f)

candidate_features = feature_definition["candidate_features"]

df = pd.read_csv(
    DATA_FILE,
    low_memory=False
)

model = joblib.load(MODEL_FILE)
preprocessor = joblib.load(PREPROCESSOR_FILE)

test = (
    df[df["data_split"] == "test"]
    .sort_values(
        ["match_date", "match_id", "player_id"]
    )
    .head(2103)
    .copy()
)

X_source = test.rename(
    columns={
        "country_x": "country",
        "playing_role_clean_x": "playing_role_clean",
        "source_role_x": "source_role"
    }
)

X_raw = X_source[candidate_features]

X_processed = preprocessor.transform(
    X_raw
)

feature_names = list(
    preprocessor.get_feature_names_out()
)

explainer = shap.TreeExplainer(model)

shap_values = np.asarray(
    explainer.shap_values(X_processed)
)

if shap_values.ndim != 2:
    raise RuntimeError(
        f"Unexpected SHAP shape: {shap_values.shape}"
    )

processed_importance = np.mean(
    np.abs(shap_values),
    axis=0
)

processed_df = pd.DataFrame(
    {
        "processed_feature": feature_names,
        "mean_abs_shap": processed_importance
    }
)

# Save the detailed transformed-feature artifact
processed_df.sort_values(
    "mean_abs_shap",
    ascending=False
).to_csv(
    XAI_DIR / "shap_global_importance_processed.csv",
    index=False
)

# Map transformed features back to the original 63 features
raw_totals = {
    feature: 0.0
    for feature in candidate_features
}

for feature_name, value in zip(
    feature_names,
    processed_importance
):

    base = feature_name

    if "__" in base:
        base = base.split(
            "__",
            1
        )[1]

    matched = None

    # Exact numeric feature
    if base in raw_totals:
        matched = base

    else:
        # One-hot categorical feature
        candidates = [
            f
            for f in candidate_features
            if base.startswith(
                f + "_"
            )
        ]

        if candidates:
            matched = max(
                candidates,
                key=len
            )

    if matched is not None:
        raw_totals[matched] += float(value)

raw_df = (
    pd.DataFrame(
        {
            "feature": list(
                raw_totals.keys()
            ),
            "mean_abs_shap": list(
                raw_totals.values()
            )
        }
    )
    .sort_values(
        "mean_abs_shap",
        ascending=False
    )
    .reset_index(drop=True)
)

raw_df["rank"] = (
    np.arange(
        len(raw_df)
    ) + 1
)

raw_csv = (
    XAI_DIR /
    "shap_global_importance.csv"
)

raw_df.to_csv(
    raw_csv,
    index=False
)

top20 = (
    raw_df
    .head(20)
    .sort_values(
        "mean_abs_shap",
        ascending=True
    )
)

plt.figure(
    figsize=(11, 9)
)

plt.barh(
    top20["feature"],
    top20["mean_abs_shap"]
)

plt.xlabel(
    "Mean absolute SHAP value"
)

plt.ylabel(
    "Original model feature"
)

plt.title(
    "Global SHAP Feature Importance — T20 Best XI"
)

plt.tight_layout()

plot_path = (
    XAI_DIR /
    "shap_global_top20_bar.png"
)

plt.savefig(
    plot_path,
    dpi=180,
    bbox_inches="tight"
)

plt.close()

print("\n=== ORIGINAL-FEATURE SHAP TOP 10 ===")

print(
    raw_df[
        [
            "rank",
            "feature",
            "mean_abs_shap"
        ]
    ]
    .head(10)
    .to_string(index=False)
)

print("\nSaved:")
print(raw_csv)
print(plot_path)
print(
    XAI_DIR /
    "shap_global_importance_processed.csv"
)
