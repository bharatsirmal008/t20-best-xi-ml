import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

DATA = Path(r".\results\phase7_player_match_features_split.csv")
FEATURES = Path(r".\models\final_t20_suitability_model\feature_lists.json")
OUT = Path(r".\results\phase10_xai_fairness\drift_analysis")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA, low_memory=False)

with open(FEATURES, "r", encoding="utf-8") as f:
    feature_def = json.load(f)

def find_list(obj, wanted_key):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.lower() == wanted_key.lower() and isinstance(v, list):
                return v
            found = find_list(v, wanted_key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = find_list(item, wanted_key)
            if found is not None:
                return found
    return None

raw_features = find_list(feature_def, "raw_features") or []
numeric_features = find_list(feature_def, "numeric_features") or []
categorical_features = find_list(feature_def, "categorical_features") or []

if not raw_features:
    raw_features = [c for c in df.columns if c.startswith("pre_")]
    for c in ["venue_standard", "team_id", "opponent_team_id"]:
        if c in df.columns and c not in raw_features:
            raw_features.append(c)

train = df[df["data_split"] == "train"].copy()
test = df[df["data_split"] == "test"].copy()

print("TRAIN:", len(train))
print("TEST :", len(test))
print("Raw features checked:", len(raw_features))

def psi_numeric(train_s, test_s, bins=10):
    a = pd.to_numeric(train_s, errors="coerce").dropna().to_numpy()
    b = pd.to_numeric(test_s, errors="coerce").dropna().to_numpy()

    if len(a) == 0 or len(b) == 0:
        return np.nan

    edges = np.unique(np.nanquantile(a, np.linspace(0, 1, bins + 1)))

    if len(edges) < 3:
        return 0.0

    a_bin = np.digitize(a, edges[1:-1], right=False)
    b_bin = np.digitize(b, edges[1:-1], right=False)

    pa = np.bincount(a_bin, minlength=len(edges))
    pb = np.bincount(b_bin, minlength=len(edges))

    pa = pa / max(pa.sum(), 1)
    pb = pb / max(pb.sum(), 1)

    eps = 1e-6
    pa = np.clip(pa, eps, None)
    pb = np.clip(pb, eps, None)

    return float(np.sum((pb - pa) * np.log(pb / pa)))

def psi_categorical(train_s, test_s):
    a = train_s.astype("string").fillna("<MISSING>")
    b = test_s.astype("string").fillna("<MISSING>")

    cats = sorted(set(a.unique()).union(set(b.unique())))

    pa = a.value_counts(normalize=True).reindex(cats, fill_value=0)
    pb = b.value_counts(normalize=True).reindex(cats, fill_value=0)

    eps = 1e-6
    pa = np.clip(pa.to_numpy(), eps, None)
    pb = np.clip(pb.to_numpy(), eps, None)

    return float(np.sum((pb - pa) * np.log(pb / pa)))

def psi_level(v):
    if pd.isna(v):
        return "Unavailable"
    if v < 0.10:
        return "Low"
    if v < 0.25:
        return "Moderate"
    return "High"

rows = []

for feature in raw_features:
    if feature not in df.columns:
        continue

    train_s = train[feature]
    test_s = test[feature]

    train_missing = float(train_s.isna().mean())
    test_missing = float(test_s.isna().mean())
    missing_shift = abs(test_missing - train_missing)

    is_cat = feature in categorical_features

    if not is_cat and feature not in numeric_features:
        is_cat = not pd.api.types.is_numeric_dtype(df[feature])

    if is_cat:
        psi = psi_categorical(train_s, test_s)
        ks_stat = np.nan
        ks_p = np.nan
        feature_type = "categorical"
    else:
        psi = psi_numeric(train_s, test_s)
        a = pd.to_numeric(train_s, errors="coerce").dropna()
        b = pd.to_numeric(test_s, errors="coerce").dropna()

        if len(a) > 0 and len(b) > 0:
            ks = ks_2samp(a, b)
            ks_stat = float(ks.statistic)
            ks_p = float(ks.pvalue)
        else:
            ks_stat = np.nan
            ks_p = np.nan

        feature_type = "numeric"

    rows.append({
        "feature": feature,
        "type": feature_type,
        "psi": psi,
        "psi_level": psi_level(psi),
        "train_missing_pct": train_missing * 100,
        "test_missing_pct": test_missing * 100,
        "missing_shift_pct": missing_shift * 100,
        "ks_statistic": ks_stat,
        "ks_pvalue": ks_p
    })

result = pd.DataFrame(rows)

def final_level(row):
    psi = row["psi"]
    miss = row["missing_shift_pct"]

    if (pd.notna(psi) and psi >= 0.25) or miss >= 5:
        return "High"
    if (pd.notna(psi) and psi >= 0.10) or miss >= 2:
        return "Moderate"
    return "Low"

result["drift_level"] = result.apply(final_level, axis=1)
result = result.sort_values(["psi", "missing_shift_pct"], ascending=False)

csv_path = OUT / "train_vs_test_drift_feature_summary.csv"
json_path = OUT / "train_vs_test_drift_summary.json"

result.to_csv(csv_path, index=False)

summary = {
    "comparison": "train_vs_test",
    "train_rows": int(len(train)),
    "test_rows": int(len(test)),
    "train_date_min": str(train["match_date"].min()),
    "train_date_max": str(train["match_date"].max()),
    "test_date_min": str(test["match_date"].min()),
    "test_date_max": str(test["match_date"].max()),
    "features_checked": int(len(result)),
    "low_drift_features": int((result["drift_level"] == "Low").sum()),
    "moderate_drift_features": int((result["drift_level"] == "Moderate").sum()),
    "high_drift_features": int((result["drift_level"] == "High").sum()),
    "max_psi": float(result["psi"].max()) if len(result) else None,
    "mean_psi": float(result["psi"].mean()) if len(result) else None
}

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\n=== DRIFT SUMMARY ===")
print(json.dumps(summary, indent=2))

print("\n=== TOP 15 FEATURES BY PSI ===")
print(
    result[
        [
            "feature",
            "type",
            "psi",
            "psi_level",
            "missing_shift_pct",
            "ks_statistic"
        ]
    ].head(15).to_string(index=False)
)

print("\nSaved:")
print(csv_path)
print(json_path)
