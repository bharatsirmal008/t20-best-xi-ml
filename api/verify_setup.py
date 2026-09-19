from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

required_files = {
    # files/
    "players_performance":
        BASE / "files" / "players_performance_final.csv",

    "venue_context":
        BASE / "files" / "venue_context_final.csv",

    # results/
    "phase7_features":
        BASE / "results" / "phase7_player_match_features_split.csv",

    "phase7_training":
        BASE / "results" / "phase7_historical_player_match_training.csv",

    "phase9_team_catalog":
        BASE / "results" / "phase9_team_catalog.csv",

    "phase9_country_catalog":
        BASE / "results" / "phase9_country_catalog.csv",

    # model/
    "model":
        BASE / "models" / "final_t20_suitability_model"
        / "final_hist_gradient_boosting.pkl",

    "preprocessor":
        BASE / "models" / "final_t20_suitability_model"
        / "final_preprocessor.pkl",

    "feature_definition":
        BASE / "models" / "final_t20_suitability_model"
        / "feature_definition.pkl",

    "model_config":
        BASE / "models" / "final_t20_suitability_model"
        / "model_config.json",

    "feature_lists":
        BASE / "models" / "final_t20_suitability_model"
        / "feature_lists.json",

    "training_metadata":
        BASE / "models" / "final_t20_suitability_model"
        / "training_metadata.csv",
}

print("=" * 70)
print("T20/T20I API DEPLOYMENT - LOCAL FILE CHECK")
print("=" * 70)

all_ok = True

for name, path in required_files.items():

    exists = path.exists()

    print(
        f"{name:<25} : "
        f"{'FOUND ✅' if exists else 'MISSING ❌'}"
    )

    if not exists:
        print(f"    {path}")
        all_ok = False

print("\n" + "=" * 70)

if all_ok:
    print("✅ ALL REQUIRED DEPLOYMENT FILES FOUND")
    print("✅ LOCAL SETUP PASS")
else:
    print("❌ SOME REQUIRED FILES ARE MISSING")
    print("❌ LOCAL SETUP FAIL")