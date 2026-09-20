import os
import re
import pickle
from pathlib import Path

import joblib
import json
import numpy as np
import pandas as pd
import streamlit as st

from scipy.optimize import milp, LinearConstraint, Bounds


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="T20 Best Playing XI Predictor",
    page_icon="🏏",
    layout="wide"
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

DATA_DIR = PROJECT_DIR / "files"
RESULT_DIR = PROJECT_DIR / "results"
MODEL_DIR = (
    PROJECT_DIR
    / "models"
    / "final_t20_suitability_model"
)

PLAYER_FILE = (
    DATA_DIR
    / "players_performance_final.csv"
)

FEATURE_FILE = (
    RESULT_DIR
    / "phase7_player_match_features_split.csv"
)

TRAINING_FILE = (
    RESULT_DIR
    / "phase7_historical_player_match_training.csv"
)

TEAM_CATALOG_FILE = (
    RESULT_DIR
    / "phase9_team_catalog.csv"
)

COUNTRY_CATALOG_FILE = (
    RESULT_DIR
    / "phase9_country_catalog.csv"
)

MODEL_FILE = (
    MODEL_DIR
    / "final_hist_gradient_boosting.pkl"
)

PREPROCESSOR_FILE = (
    MODEL_DIR
    / "final_preprocessor.pkl"
)

FEATURE_DEF_FILE = (
    MODEL_DIR
    / "feature_definition.pkl"
)


# ============================================================
# PHASE 10 XAI / FAIRNESS ARTIFACT PATHS
# ============================================================

XAI_DIR_CANDIDATES = [
    PROJECT_DIR / "results" / "phase10_xai_fairness",
    PROJECT_DIR.parent / "results" / "phase10_xai_fairness"
]

XAI_DIR = next(
    (
        p
        for p in XAI_DIR_CANDIDATES
        if p.exists()
    ),
    PROJECT_DIR / "results" / "phase10_xai_fairness"
)

SHAP_DIR = XAI_DIR / "shap"
LIME_DIR = XAI_DIR / "lime"
FAIRNESS_DIR = XAI_DIR / "fairness"
MITIGATION_DIR = XAI_DIR / "mitigation"



# ============================================================
# ARTIFACT LOADING
# ============================================================

@st.cache_resource
def load_artifacts():

    players = pd.read_csv(
        PLAYER_FILE
    )

    historical = pd.read_csv(
        FEATURE_FILE
    )

    training = pd.read_csv(
        TRAINING_FILE
    )

    team_catalog = pd.read_csv(
        TEAM_CATALOG_FILE
    )

    country_catalog = pd.read_csv(
        COUNTRY_CATALOG_FILE
    )

    model = joblib.load(
        MODEL_FILE
    )

    preprocessor = joblib.load(
        PREPROCESSOR_FILE
    )

    with open(
        FEATURE_DEF_FILE,
        "rb"
    ) as f:

        feature_definition = pickle.load(
            f
        )

    return (
        players,
        historical,
        training,
        team_catalog,
        country_catalog,
        model,
        preprocessor,
        feature_definition
    )


# ============================================================
# HELPERS
# ============================================================

def extract_features(
    definition,
    keys,
    fallback=None
):

    if isinstance(
        definition,
        dict
    ):

        for key in keys:

            if key in definition:

                value = definition[key]

                if isinstance(
                    value,
                    (list, tuple)
                ):

                    return list(value)

    return fallback


def clean_category(series):

    values = (
        series
        .astype(object)
        .to_numpy()
    )

    cleaned = np.empty(
        len(values),
        dtype=object
    )

    for i, value in enumerate(values):

        if pd.isna(value):

            cleaned[i] = np.nan

        else:

            cleaned[i] = str(value)

    return pd.Series(
        cleaned,
        index=series.index,
        dtype=object
    )


def prepare_data():

    (
        players,
        historical,
        training,
        team_catalog,
        country_catalog,
        model,
        preprocessor,
        feature_definition
    ) = load_artifacts()

    # --------------------------------------------
    # Player master
    # --------------------------------------------

    players["ID"] = pd.to_numeric(
        players["ID"],
        errors="coerce"
    )

    players = players.dropna(
        subset=["ID"]
    ).copy()

    players["ID"] = (
        players["ID"]
        .astype(int)
    )

    players["country"] = (
        players["country"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------
    # Historical data
    # --------------------------------------------

    for canonical, candidates in {

        "country": [
            "country_x",
            "country_y"
        ],

        "playing_role_clean": [
            "playing_role_clean_x",
            "playing_role_clean_y"
        ],

        "source_role": [
            "source_role_x",
            "source_role_y"
        ]

    }.items():

        if canonical not in historical.columns:

            source = None

            for candidate in candidates:

                if candidate in historical.columns:

                    source = candidate
                    break

            if source is None:

                raise ValueError(
                    f"Could not reconstruct {canonical}"
                )

            historical[canonical] = (
                historical[source]
            )

    historical["player_id"] = pd.to_numeric(
        historical["player_id"],
        errors="coerce"
    )

    historical["team_id"] = pd.to_numeric(
        historical["team_id"],
        errors="coerce"
    )

    historical["match_date"] = pd.to_datetime(
        historical["match_date"],
        errors="coerce"
    )

    historical = historical.dropna(
        subset=["player_id"]
    ).copy()

    historical["player_id"] = (
        historical["player_id"]
        .astype(int)
    )

    # --------------------------------------------
    # Training membership
    # --------------------------------------------

    training["player_id"] = pd.to_numeric(
        training["player_id"],
        errors="coerce"
    )

    training["team_id"] = pd.to_numeric(
        training["team_id"],
        errors="coerce"
    )

    training = training.dropna(
        subset=[
            "player_id",
            "team_id"
        ]
    ).copy()

    training["player_id"] = (
        training["player_id"]
        .astype(int)
    )

    training["team_id"] = (
        training["team_id"]
        .astype(int)
    )

    team_membership = (
        training[
            [
                "team_id",
                "player_id"
            ]
        ]
        .drop_duplicates()
    )

    # --------------------------------------------
    # Feature definitions
    # --------------------------------------------

    raw_features = extract_features(
        feature_definition,
        [
            "raw_features",
            "feature_columns",
            "candidate_features",
            "features"
        ]
    )

    categorical_features = extract_features(
        feature_definition,
        [
            "categorical_features",
            "categorical"
        ]
    )

    numeric_features = extract_features(
        feature_definition,
        [
            "numeric_features",
            "numeric"
        ]
    )

    if raw_features is None:

        raise ValueError(
            "Could not recover raw model features."
        )

    if categorical_features is None:

        categorical_features = [
            "team_id",
            "opponent_team_id",
            "venue_standard",
            "city",
            "season",
            "country",
            "batting_style",
            "bowling_style",
            "playing_role_clean",
            "source_role"
        ]

    if numeric_features is None:

        numeric_features = [
            c
            for c in raw_features
            if c not in categorical_features
        ]

    return {
        "players": players,
        "historical": historical,
        "training": training,
        "team_catalog": team_catalog,
        "country_catalog": country_catalog,
        "team_membership": team_membership,
        "model": model,
        "preprocessor": preprocessor,
        "raw_features": raw_features,
        "categorical_features": categorical_features,
        "numeric_features": numeric_features
    }


# ============================================================
# ELIGIBILITY
# ============================================================

def get_eligible_squad(
    data,
    mode,
    selection_value,
    explicit_ids=None
):

    players = data["players"]
    membership = data["team_membership"]
    team_catalog = data["team_catalog"]
    country_catalog = data["country_catalog"]

    mode = (
        str(mode)
        .strip()
        .upper()
    )

    # --------------------------------------------
    # Explicit squad
    # --------------------------------------------

    if explicit_ids is not None:

        ids = [
            int(x)
            for x in explicit_ids
        ]

        ids = list(
            dict.fromkeys(ids)
        )

        if len(ids) < 11:

            raise ValueError(
                "Custom squad must contain at least 11 players."
            )

        if mode == "IPL":

            team_id = int(
                selection_value
            )

            team_members = set(
                membership.loc[
                    membership["team_id"]
                    == team_id,
                    "player_id"
                ]
                .astype(int)
            )

            outside = (
                set(ids)
                - team_members
            )

            if outside:

                raise ValueError(
                    "Custom squad contains players outside "
                    "the selected team."
                )

        else:

            country = str(
                selection_value
            ).strip()

            country_members = set(
                players.loc[
                    players["country"]
                    == country,
                    "ID"
                ]
                .astype(int)
            )

            outside = (
                set(ids)
                - country_members
            )

            if outside:

                raise ValueError(
                    "Custom squad contains players outside "
                    "the selected country."
                )

        squad = players[
            players["ID"].isin(ids)
        ].copy()

        return squad

    # --------------------------------------------
    # IPL
    # --------------------------------------------

    if mode == "IPL":

        team_id = int(
            selection_value
        )

        valid_team_ids = set(
            team_catalog["team_id"]
            .astype(int)
        )

        if team_id not in valid_team_ids:

            raise ValueError(
                "Invalid team ID."
            )

        ids = (
            membership.loc[
                membership["team_id"]
                == team_id,
                "player_id"
            ]
            .drop_duplicates()
            .astype(int)
            .tolist()
        )

        squad = players[
            players["ID"].isin(ids)
        ].copy()

        return squad

    # --------------------------------------------
    # COUNTRY
    # --------------------------------------------

    country = str(
        selection_value
    ).strip()

    valid_countries = set(
        country_catalog["country"]
        .astype(str)
        .str.strip()
    )

    if country not in valid_countries:

        raise ValueError(
            "Invalid country."
        )

    squad = players[
        players["country"]
        == country
    ].copy()

    return squad


# ============================================================
# HISTORICAL SNAPSHOT
# ============================================================

def get_snapshots(
    data,
    eligible_ids,
    mode,
    selection_value
):

    historical = data["historical"]

    rows = historical[
        historical["player_id"]
        .isin(eligible_ids)
    ].copy()

    if len(rows) == 0:

        raise ValueError(
            "No historical feature records found "
            "for eligible players."
        )

    if mode == "IPL":

        team_id = int(
            selection_value
        )

        team_rows = rows[
            rows["team_id"]
            == team_id
        ].copy()

        team_rows = (
            team_rows
            .sort_values(
                [
                    "player_id",
                    "match_date"
                ]
            )
            .groupby(
                "player_id",
                as_index=False
            )
            .tail(1)
        )

        selected = set(
            team_rows["player_id"]
            .astype(int)
        )

        missing = [
            pid
            for pid in eligible_ids
            if pid not in selected
        ]

        if missing:

            fallback = (
                rows[
                    rows["player_id"]
                    .isin(missing)
                ]
                .sort_values(
                    [
                        "player_id",
                        "match_date"
                    ]
                )
                .groupby(
                    "player_id",
                    as_index=False
                )
                .tail(1)
            )

            snapshots = pd.concat(
                [
                    team_rows,
                    fallback
                ],
                ignore_index=True
            )

            snapshots = (
                snapshots
                .drop_duplicates(
                    "player_id"
                )
            )

        else:

            snapshots = team_rows

        snapshots["team_id"] = team_id

    else:

        snapshots = (
            rows
            .sort_values(
                [
                    "player_id",
                    "match_date"
                ]
            )
            .groupby(
                "player_id",
                as_index=False
            )
            .tail(1)
        )

    return snapshots


# ============================================================
# ML SCORING
# ============================================================

def score_squad(
    data,
    squad,
    mode,
    selection_value
):

    model = data["model"]
    preprocessor = data["preprocessor"]

    raw_features = data["raw_features"]
    categorical_features = (
        data["categorical_features"]
    )
    numeric_features = (
        data["numeric_features"]
    )

    eligible_ids = set(
        squad["ID"]
        .astype(int)
    )

    snapshots = get_snapshots(
        data,
        eligible_ids,
        mode,
        selection_value
    )

    identity = squad[
        [
            c
            for c in [
                "ID",
                "name",
                "country",
                "playing_role_clean"
            ]
            if c in squad.columns
        ]
    ].copy()

    identity = identity.rename(
        columns={
            "ID": "player_id"
        }
    )

    model_input = identity.merge(
        snapshots,
        on="player_id",
        how="left",
        suffixes=(
            "_master",
            "_snapshot"
        )
    )

    # --------------------------------------------
    # Rebuild model features
    # --------------------------------------------

    for col in raw_features:

        if col in model_input.columns:
            continue

        snapshot_col = (
            f"{col}_snapshot"
        )

        master_col = (
            f"{col}_master"
        )

        if snapshot_col in model_input.columns:

            model_input[col] = (
                model_input[
                    snapshot_col
                ]
            )

        elif master_col in model_input.columns:

            model_input[col] = (
                model_input[
                    master_col
                ]
            )

        else:

            model_input[col] = np.nan

    # --------------------------------------------
    # Correct dtypes
    # --------------------------------------------

    for col in categorical_features:

        if col in model_input.columns:

            model_input[col] = (
                clean_category(
                    model_input[col]
                )
            )

    for col in numeric_features:

        if col in model_input.columns:

            model_input[col] = pd.to_numeric(
                model_input[col],
                errors="coerce"
            )

    X_raw = model_input[
        raw_features
    ].copy()

    for col in categorical_features:

        if col in X_raw.columns:

            X_raw[col] = (
                clean_category(
                    X_raw[col]
                )
            )

    X_processed = (
        preprocessor.transform(
            X_raw
        )
    )

    scores = model.predict(
        X_processed
    )

    scores = np.asarray(
        scores,
        dtype=float
    )

    scores = np.clip(
        scores,
        0,
        1
    )

    result = squad.copy()

    prediction_map = dict(
        zip(
            model_input[
                "player_id"
            ].astype(int),
            scores
        )
    )

    result["suitability_score"] = (
        result["ID"]
        .astype(int)
        .map(
            prediction_map
        )
    )

    result = result.sort_values(
        "suitability_score",
        ascending=False
    ).reset_index(
        drop=True
    )

    result["eligible_rank"] = (
        np.arange(
            1,
            len(result) + 1
        )
    )

    return result


# ============================================================
# OPTIMIZER
# ============================================================

def optimize_xi(
    scored,
    mode
):

    df = scored[
        scored["suitability_score"]
        .notna()
    ].copy()

    if len(df) < 11:

        raise ValueError(
            "At least 11 scored players are required."
        )

    role = (
        df["playing_role_clean"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["is_batter_flag"] = (
        role.isin(
            [
                "batsman",
                "wicketkeeper-batsman"
            ]
        )
    ).astype(int)

    df["is_bowler_flag"] = (
        role == "bowler"
    ).astype(int)

    df["is_allrounder_flag"] = (
        role == "allrounder"
    ).astype(int)

    if "is_wicketkeeper" in df.columns:

        keeper = pd.to_numeric(
            df["is_wicketkeeper"],
            errors="coerce"
        ).fillna(0)

        df["is_keeper_flag"] = (
            keeper > 0
        ).astype(int)

    else:

        df["is_keeper_flag"] = (
            role.str.contains(
                "wicketkeeper"
            )
        ).astype(int)

    df["is_overseas_flag"] = (
        df["country"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .str.lower()
        .ne("india")
    ).astype(int)

    max_overseas = (
        4
        if mode == "IPL"
        else None
    )

    n = len(df)

    objective = -df[
        "suitability_score"
    ].to_numpy(
        dtype=float
    )

    integrality = np.ones(
        n,
        dtype=int
    )

    bounds = Bounds(
        np.zeros(n),
        np.ones(n)
    )

    rows = [
        np.ones(n),
        df["is_batter_flag"].to_numpy(float),
        df["is_bowler_flag"].to_numpy(float),
        df["is_allrounder_flag"].to_numpy(float),
        df["is_keeper_flag"].to_numpy(float)
    ]

    lower = [
        11,
        3,
        3,
        0,
        1
    ]

    upper = [
        11,
        7,
        6,
        4,
        2
    ]

    if max_overseas is not None:

        rows.append(
            df[
                "is_overseas_flag"
            ].to_numpy(float)
        )

        lower.append(0)
        upper.append(4)

    result = milp(
        c=objective,
        integrality=integrality,
        bounds=bounds,
        constraints=LinearConstraint(
            np.vstack(rows),
            np.asarray(
                lower,
                dtype=float
            ),
            np.asarray(
                upper,
                dtype=float
            )
        )
    )

    if not result.success:

        raise ValueError(
            f"Optimizer failed: {result.message}"
        )

    xi = df[
        result.x > 0.5
    ].copy()

    if len(xi) != 11:

        raise ValueError(
            "Optimizer did not return exactly 11 players."
        )

    xi = xi.sort_values(
        "suitability_score",
        ascending=False
    ).reset_index(
        drop=True
    )

    xi.insert(
        0,
        "xi_position",
        np.arange(1, 12)
    )

    composition = {
        "xi_size": len(xi),
        "batters": int(
            xi["is_batter_flag"].sum()
        ),
        "bowlers": int(
            xi["is_bowler_flag"].sum()
        ),
        "allrounders": int(
            xi["is_allrounder_flag"].sum()
        ),
        "wicketkeepers": int(
            xi["is_keeper_flag"].sum()
        ),
        "overseas": int(
            xi["is_overseas_flag"].sum()
        ),
        "total_suitability": float(
            xi["suitability_score"].sum()
        ),
        "mean_suitability": float(
            xi["suitability_score"].mean()
        )
    }

    return xi, composition


# ============================================================
# END-TO-END PREDICTION
# ============================================================

def predict_best_xi(
    data,
    mode,
    selection_value,
    explicit_ids=None
):

    squad = get_eligible_squad(
        data,
        mode,
        selection_value,
        explicit_ids
    )

    if len(squad) < 11:

        raise ValueError(
            f"Eligible squad contains only {len(squad)} "
            "players. At least 11 are required."
        )

    scored = score_squad(
        data,
        squad,
        mode,
        selection_value
    )

    xi, composition = optimize_xi(
        scored,
        mode
    )

    # Strict eligibility audit
    eligible_ids = set(
        squad["ID"]
        .astype(int)
    )

    selected_ids = set(
        xi["ID"]
        .astype(int)
    )

    outside = (
        selected_ids
        - eligible_ids
    )

    if outside:

        raise ValueError(
            "Optimizer selected a player outside "
            "the eligible squad."
        )

    return (
        squad,
        scored,
        xi,
        composition
    )




# ============================================================
# UI
# ============================================================

# ------------------------------------------------------------
# TEAM DISPLAY NAMES
# ------------------------------------------------------------

CURRENT_IPL_TEAM_IDS = [
    3,     # Mumbai Indians
    129,   # Chennai Super Kings
    1,     # Royal Challengers Bengaluru
    6,     # Kolkata Knight Riders
    252,   # Delhi Capitals
    134,   # Rajasthan Royals
    2,     # Sunrisers Hyderabad
    494,   # Punjab Kings
    614,   # Lucknow Super Giants
    615    # Gujarat Titans
]

IPL_TEAM_NAMES = {
    3: "Mumbai Indians",
    129: "Chennai Super Kings",
    1: "Royal Challengers Bengaluru",
    6: "Kolkata Knight Riders",
    252: "Delhi Capitals",
    134: "Rajasthan Royals",
    2: "Sunrisers Hyderabad",
    494: "Punjab Kings",
    614: "Lucknow Super Giants",
    615: "Gujarat Titans"
}

# ------------------------------------------------------------
# VENUE FILE
# ------------------------------------------------------------

VENUE_FILE = (
    PROJECT_DIR
    / "results"
    / "venue_context_final.csv"
)

@st.cache_data
def load_venue_options():

    if VENUE_FILE.exists():

        venue_df = pd.read_csv(
            VENUE_FILE
        )

        if "venue_standard" in venue_df.columns:

            return (
                venue_df[
                    "venue_standard"
                ]
                .dropna()
                .astype(str)
                .str.strip()
                .drop_duplicates()
                .sort_values()
                .tolist()
            )

    return []


# ------------------------------------------------------------
# NORMALIZE PLAYER NAME
# ------------------------------------------------------------

def normalize_player_name(value):

    value = str(
        value
    ).strip().lower()

    value = re.sub(
        r"[^a-z0-9]+",
        " ",
        value
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value.strip()


# ------------------------------------------------------------
# UPLOADED SQUAD PARSER
# ------------------------------------------------------------

def parse_uploaded_squad(
    uploaded_file,
    players_master
):

    if uploaded_file is None:

        raise ValueError(
            "Please upload a squad file."
        )

    filename = (
        uploaded_file.name
        .lower()
        .strip()
    )

    if filename.endswith(
        ".csv"
    ):

        uploaded_df = pd.read_csv(
            uploaded_file
        )

    elif filename.endswith(
        (".xlsx", ".xls")
    ):

        uploaded_df = pd.read_excel(
            uploaded_file
        )

    elif filename.endswith(
        ".txt"
    ):

        raw_text = (
            uploaded_file
            .getvalue()
            .decode(
                "utf-8",
                errors="ignore"
            )
        )

        lines = [
            line.strip()
            for line in raw_text.splitlines()
            if line.strip()
        ]

        uploaded_df = pd.DataFrame(
            {
                "name": lines
            }
        )

    else:

        raise ValueError(
            "Supported files: CSV, XLSX, XLS, TXT."
        )

    if uploaded_df.empty:

        raise ValueError(
            "Uploaded squad is empty."
        )

    uploaded_df.columns = [
        str(c).strip()
        for c in uploaded_df.columns
    ]

    id_column = None

    for candidate in [
        "ID",
        "id",
        "player_id",
        "Player ID",
        "playerid"
    ]:

        if candidate in uploaded_df.columns:

            id_column = candidate
            break

    name_column = None

    for candidate in [
        "name",
        "Name",
        "player_name",
        "Player Name",
        "player",
        "Player"
    ]:

        if candidate in uploaded_df.columns:

            name_column = candidate
            break

    if (
        id_column is None
        and name_column is None
        and len(uploaded_df.columns) == 1
    ):

        name_column = uploaded_df.columns[0]

    resolved_ids = []
    matched_names = []
    unmatched_names = []
    ambiguous_names = []

    # --------------------------------------------------------
    # ID matching
    # --------------------------------------------------------

    if id_column is not None:

        ids = pd.to_numeric(
            uploaded_df[
                id_column
            ],
            errors="coerce"
        )

        for raw_id in ids.dropna():

            pid = int(raw_id)

            matches = players_master[
                players_master["ID"]
                == pid
            ]

            if len(matches) == 1:

                resolved_ids.append(
                    pid
                )

                matched_names.append(
                    str(
                        matches.iloc[0][
                            "name"
                        ]
                    )
                )

            else:

                unmatched_names.append(
                    str(raw_id)
                )

    # --------------------------------------------------------
    # NAME matching
    # --------------------------------------------------------

    elif name_column is not None:

        master = players_master.copy()

        master[
            "_normalized_name"
        ] = (
            master["name"]
            .map(
                normalize_player_name
            )
        )

        grouped = (
            master
            .groupby(
                "_normalized_name"
            )["ID"]
            .apply(list)
            .to_dict()
        )

        for raw_name in uploaded_df[
            name_column
        ].dropna():

            raw_name = str(
                raw_name
            ).strip()

            normalized = (
                normalize_player_name(
                    raw_name
                )
            )

            matches = grouped.get(
                normalized,
                []
            )

            if len(matches) == 1:

                pid = int(
                    matches[0]
                )

                resolved_ids.append(
                    pid
                )

                matched = players_master[
                    players_master["ID"]
                    == pid
                ]

                matched_names.append(
                    str(
                        matched.iloc[0]["name"]
                    )
                )

            elif len(matches) > 1:

                ambiguous_names.append(
                    raw_name
                )

            else:

                unmatched_names.append(
                    raw_name
                )

    else:

        raise ValueError(
            "Upload must contain player ID "
            "or player name."
        )

    resolved_ids = list(
        dict.fromkeys(
            resolved_ids
        )
    )

    if len(resolved_ids) == 0:

        raise ValueError(
            "No uploaded players could be matched."
        )

    return {
        "player_ids": resolved_ids,
        "matched_names": matched_names,
        "unmatched_names": unmatched_names,
        "ambiguous_names": ambiguous_names,
        "uploaded_rows": len(
            uploaded_df
        )
    }


# ------------------------------------------------------------
# CONTEXT-AWARE ML SCORING
# ------------------------------------------------------------

def score_squad_with_match_context(
    data,
    squad,
    mode,
    selection_value,
    selected_venue,
    pitch_type,
    toss_winner,
    toss_decision
):

    model = data["model"]

    preprocessor = data[
        "preprocessor"
    ]

    raw_features = data[
        "raw_features"
    ]

    categorical_features = data[
        "categorical_features"
    ]

    numeric_features = data[
        "numeric_features"
    ]

    eligible_ids = set(
        squad["ID"]
        .astype(int)
    )

    snapshots = get_snapshots(
        data,
        eligible_ids,
        mode,
        selection_value
    )

    identity = squad[
        [
            c
            for c in [
                "ID",
                "name",
                "country",
                "playing_role_clean"
            ]
            if c in squad.columns
        ]
    ].copy()

    identity = identity.rename(
        columns={
            "ID": "player_id"
        }
    )

    model_input = identity.merge(
        snapshots,
        on="player_id",
        how="left",
        suffixes=(
            "_master",
            "_snapshot"
        )
    )

    for col in raw_features:

        if col in model_input.columns:
            continue

        snapshot_col = (
            f"{col}_snapshot"
        )

        master_col = (
            f"{col}_master"
        )

        if snapshot_col in model_input.columns:

            model_input[col] = (
                model_input[
                    snapshot_col
                ]
            )

        elif master_col in model_input.columns:

            model_input[col] = (
                model_input[
                    master_col
                ]
            )

        else:

            model_input[col] = np.nan

    # Venue is a trained model feature
    if (
        "venue_standard"
        in raw_features
    ):

        model_input[
            "venue_standard"
        ] = selected_venue

    for col in categorical_features:

        if col in model_input.columns:

            model_input[col] = (
                clean_category(
                    model_input[col]
                )
            )

    for col in numeric_features:

        if col in model_input.columns:

            model_input[col] = pd.to_numeric(
                model_input[col],
                errors="coerce"
            )

    X_raw = model_input[
        raw_features
    ].copy()

    for col in categorical_features:

        if col in X_raw.columns:

            X_raw[col] = (
                clean_category(
                    X_raw[col]
                )
            )

    X_processed = (
        preprocessor.transform(
            X_raw
        )
    )

    scores = model.predict(
        X_processed
    )

    scores = np.asarray(
        scores,
        dtype=float
    )

    scores = np.clip(
        scores,
        0,
        1
    )

    result = squad.copy()

    prediction_map = dict(
        zip(
            model_input[
                "player_id"
            ].astype(int),
            scores
        )
    )

    result[
        "suitability_score"
    ] = (
        result["ID"]
        .astype(int)
        .map(
            prediction_map
        )
    )

    result = result.sort_values(
        "suitability_score",
        ascending=False
    ).reset_index(
        drop=True
    )

    result[
        "eligible_rank"
    ] = np.arange(
        1,
        len(result) + 1
    )

    result[
        "selected_venue"
    ] = selected_venue

    result[
        "pitch_type"
    ] = pitch_type

    result[
        "toss_winner"
    ] = toss_winner

    result[
        "toss_decision"
    ] = toss_decision

    # ========================================================
    # PHASE 11 XAI TRACE
    # Preserve the exact model inputs produced by the
    # production scoring pipeline.
    # ========================================================

    # ========================================================
    # PHASE 11 XAI TRACE
    # Store only plain Python lists in DataFrame.attrs.
    # This keeps Streamlit dataframe serialization clean.
    # ========================================================

    result.attrs["xai_player_ids"] = (
        model_input["player_id"]
        .astype(int)
        .tolist()
    )

    result.attrs["xai_processed_features"] = (
        np.asarray(
            X_processed,
            dtype=float
        ).tolist()
    )

    try:

        result.attrs["xai_processed_feature_names"] = (
            preprocessor
            .get_feature_names_out()
            .tolist()
        )

    except Exception:

        result.attrs["xai_processed_feature_names"] = []

    return result


# ------------------------------------------------------------
# MANUAL XI VALIDATION
# ------------------------------------------------------------


# ============================================================
# PHASE 11 XAI HELPERS
# ============================================================

def _xai_get_row(
    scored,
    player_id
):
    """
    Recover the exact processed model row used for a player.
    """

    player_ids = scored.attrs.get(
        "xai_player_ids"
    )

    X_processed = scored.attrs.get(
        "xai_processed_features"
    )

    feature_names = scored.attrs.get(
        "xai_processed_feature_names",
        []
    )

    if player_ids is None:
        raise ValueError(
            "XAI player trace is unavailable."
        )

    if X_processed is None:
        raise ValueError(
            "XAI processed feature trace is unavailable."
        )

    player_ids = np.asarray(
        player_ids,
        dtype=int
    )

    matches = np.where(
        player_ids == int(player_id)
    )[0]

    if len(matches) == 0:
        raise ValueError(
            "Player is not present in the "
            "production scoring trace."
        )

    row_index = int(
        matches[0]
    )

    return (
        np.asarray(
            X_processed,
            dtype=float
        ),
        row_index,
        list(feature_names)
    )


def _xai_processed_to_raw(
    feature_names,
    data
):
    """
    Collapse 200 processed features back to the
    63 raw model features.
    """

    categorical = list(
        data.get(
            "categorical_features",
            []
        )
    )

    mapping = []

    for feature_name in feature_names:

        name = str(
            feature_name
        )

        if "__" in name:

            prefix, body = name.split(
                "__",
                1
            )

        else:

            prefix = ""
            body = name

        raw_name = None

        if prefix == "num":

            raw_name = body

        elif prefix == "cat":

            for col in categorical:

                if (
                    body == col
                    or
                    body.startswith(
                        f"{col}_"
                    )
                ):

                    raw_name = col
                    break

        if raw_name is None:

            raw_name = body

        mapping.append(
            raw_name
        )

    return mapping


def _xai_local_shap(
    scored,
    player_id,
    data
):
    """
    Compute local SHAP using the exact processed
    representation consumed by the production model.
    """

    import shap

    model = data["model"]

    X_processed, row_index, feature_names = (
        _xai_get_row(
            scored,
            player_id
        )
    )

    if not feature_names:

        feature_names = (
            data["preprocessor"]
            .get_feature_names_out()
            .tolist()
        )

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = (
        explainer.shap_values(
            X_processed
        )
    )

    shap_values = np.asarray(
        shap_values,
        dtype=float
    )

    if shap_values.ndim == 3:

        shap_values = shap_values[0]

    local_values = (
        shap_values[row_index]
    )

    raw_mapping = (
        _xai_processed_to_raw(
            feature_names,
            data
        )
    )

    contributions = {}

    for raw_name, value in zip(
        raw_mapping,
        local_values
    ):

        contributions[
            raw_name
        ] = (
            contributions.get(
                raw_name,
                0.0
            )
            + float(value)
        )

    result = pd.DataFrame(
        [
            {
                "feature":
                    name,
                "shap_value":
                    value,
                "absolute_shap":
                    abs(value)
            }
            for name, value
            in contributions.items()
        ]
    )

    return (
        result
        .sort_values(
            "absolute_shap",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )


def _xai_local_lime(
    scored,
    player_id,
    data,
    num_samples=800
):
    """
    Compute a local LIME explanation on the
    same processed feature space used by HGB.
    """

    from lime.lime_tabular import (
        LimeTabularExplainer
    )

    model = data["model"]

    X_processed, row_index, feature_names = (
        _xai_get_row(
            scored,
            player_id
        )
    )

    if not feature_names:

        feature_names = [
            f"feature_{i}"
            for i in range(
                X_processed.shape[1]
            )
        ]

    background = np.asarray(
        X_processed,
        dtype=float
    )

    explainer = (
        LimeTabularExplainer(
            background,
            mode="regression",
            feature_names=feature_names,
            discretize_continuous=True,
            random_state=42
        )
    )

    explanation = (
        explainer.explain_instance(
            background[row_index],
            lambda X:
                model.predict(
                    np.asarray(
                        X,
                        dtype=float
                    )
                ),
            num_features=min(
                20,
                len(feature_names)
            ),
            num_samples=int(
                num_samples
            )
        )
    )

    if not getattr(
        explanation,
        "local_exp",
        None
    ):

        raise ValueError(
            "LIME returned no local explanation."
        )

    # For regression, local_exp may not use
    # explanation.mode as its dictionary key.
    local_exp = next(
        iter(
            explanation.local_exp.values()
        )
    )

    raw_mapping = (
        _xai_processed_to_raw(
            feature_names,
            data
        )
    )

    weights = {}

    for feature_index, weight in local_exp:

        raw_name = (
            raw_mapping[
                int(feature_index)
            ]
        )

        weights[
            raw_name
        ] = (
            weights.get(
                raw_name,
                0.0
            )
            + float(weight)
        )

    result = pd.DataFrame(
        [
            {
                "feature":
                    name,
                "lime_weight":
                    value,
                "absolute_weight":
                    abs(value)
            }
            for name, value
            in weights.items()
        ]
    )

    return (
        result
        .sort_values(
            "absolute_weight",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )



def validate_manual_xi(
    selected_df,
    mode
):

    if len(selected_df) != 11:

        return False, (
            f"Select exactly 11 players. "
            f"Current selection: {len(selected_df)}."
        )

    role = (
        selected_df[
            "playing_role_clean"
        ]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    batters = int(
        role.isin(
            [
                "batsman",
                "wicketkeeper-batsman"
            ]
        ).sum()
    )

    bowlers = int(
        (
            role == "bowler"
        ).sum()
    )

    allrounders = int(
        (
            role == "allrounder"
        ).sum()
    )

    if "is_wicketkeeper" in selected_df.columns:

        keepers = int(
            pd.to_numeric(
                selected_df[
                    "is_wicketkeeper"
                ],
                errors="coerce"
            )
            .fillna(0)
            .gt(0)
            .sum()
        )

    else:

        keepers = int(
            role.str.contains(
                "wicketkeeper"
            ).sum()
        )

    overseas = int(
        selected_df[
            "country"
        ]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .str.lower()
        .ne("india")
        .sum()
    )

    checks = [
        (
            3 <= batters <= 7,
            f"Batters: {batters} (3–7)"
        ),
        (
            3 <= bowlers <= 6,
            f"Bowlers: {bowlers} (3–6)"
        ),
        (
            0 <= allrounders <= 4,
            f"All-rounders: {allrounders} (0–4)"
        ),
        (
            1 <= keepers <= 2,
            f"Wicketkeepers: {keepers} (1–2)"
        )
    ]

    if str(mode).upper() == "IPL":

        checks.append(
            (
                overseas <= 4,
                f"Overseas: {overseas} (0–4)"
            )
        )

    failed = [
        message
        for passed, message
        in checks
        if not passed
    ]

    if failed:

        return False, (
            "Constraint failure: "
            + "; ".join(failed)
        )

    return True, (
        f"Batters {batters} | "
        f"Bowlers {bowlers} | "
        f"All-rounders {allrounders} | "
        f"Wicketkeepers {keepers} | "
        f"Overseas {overseas}"
    )


# ============================================================
# PAGE
# ============================================================

st.title(
    "🏏 T20 Best Playing XI Predictor"
)

st.caption(
    "Eligible squad → Match context → "
    "ML suitability → Optimized XI → User confirmation"
)

try:

    data = prepare_data()

except Exception as e:

    st.error(
        f"Could not load project artifacts: {e}"
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "1. Selection"
    )

    selection_mode_label = st.radio(
        "Selection mode",
        [
            "IPL / Franchise",
            "Country"
        ]
    )

    mode = (
        "IPL"
        if selection_mode_label
        == "IPL / Franchise"
        else "COUNTRY"
    )

    # --------------------------------------------------------
    # IPL TEAM
    # --------------------------------------------------------

    if mode == "IPL":

        catalog_team_ids = set(
            data[
                "team_catalog"
            ]["team_id"]
            .astype(int)
            .tolist()
        )

        valid_team_ids = [
            team_id
            for team_id in CURRENT_IPL_TEAM_IDS
            if team_id in catalog_team_ids
        ]

        team_options = {}

        for team_id in valid_team_ids:

            team_name = (
                IPL_TEAM_NAMES.get(
                    team_id,
                    f"Team ID {team_id}"
                )
            )

            team_options[
                team_name
            ] = team_id

        selected_team_name = st.selectbox(
            "IPL team",
            sorted(
                team_options.keys()
            )
        )

        selected_team_id = (
            team_options[
                selected_team_name
            ]
        )

        selection_value = (
            selected_team_id
        )

        st.caption(
            f"Internal ID: {selected_team_id}"
        )

    # --------------------------------------------------------
    # COUNTRY
    # --------------------------------------------------------

    else:

        countries = (
            data[
                "country_catalog"
            ]["country"]
            .astype(str)
            .str.strip()
            .sort_values()
            .tolist()
        )

        selected_country = st.selectbox(
            "Country",
            countries
        )

        selection_value = (
            selected_country
        )

    # --------------------------------------------------------
    # MATCH CONTEXT
    # --------------------------------------------------------

    st.divider()

    st.header(
        "2. Match Context"
    )

    venue_options = (
        load_venue_options()
    )

    if venue_options:

        selected_venue = st.selectbox(
            "Venue",
            venue_options
        )

    else:

        selected_venue = st.text_input(
            "Venue"
        )

    pitch_type = st.selectbox(
        "Pitch type",
        [
            "Balanced",
            "Batting-friendly",
            "Bowling-friendly",
            "Pace-friendly",
            "Spin-friendly",
            "Slow / Low",
            "Dry / Turning"
        ]
    )

    toss_options = [
        selected_team_name
        if mode == "IPL"
        else selected_country,
        "Opponent"
    ]

    toss_winner = st.selectbox(
        "Toss winner",
        toss_options
    )

    toss_decision = st.selectbox(
        "Toss decision",
        [
            "Bat first",
            "Field first"
        ]
    )

    st.caption(
        "Venue is currently a trained model feature. "
        "Pitch type and toss are collected as match "
        "context but are not yet ML features."
    )

    # --------------------------------------------------------
    # SQUAD
    # --------------------------------------------------------

    st.divider()


    st.header(
        "3. Squad"
    )

    # --------------------------------------------------------
    # BASE ELIGIBLE POOL
    # --------------------------------------------------------

    base_eligible_squad = (
        get_eligible_squad(
            data,
            mode,
            selection_value
        )
        .copy()
    )

    # The existing Generate Best XI block expects
    # upload_result["player_ids"].
    upload_result = None

    # --------------------------------------------------------
    # PLAYER NAME NORMALIZATION
    # --------------------------------------------------------

    def normalize_uploaded_name(value):

        value = str(
            value
        ).strip().lower()

        value = re.sub(
            r"[^a-z0-9]+",
            " ",
            value
        )

        value = re.sub(
            r"\s+",
            " ",
            value
        )

        return value.strip()

    # --------------------------------------------------------
    # RESOLVE NUMBERED NAME LIST
    # --------------------------------------------------------

    def resolve_numbered_names(
        raw_text,
        eligible_pool
    ):

        text = str(
            raw_text
        )

        # Allow comma/semicolon separated input
        text = text.replace(
            ";",
            "\n"
        )

        text = text.replace(
            ",",
            "\n"
        )

        lines = []

        for raw_line in text.splitlines():

            line = raw_line.strip()

            if not line:
                continue

            # Remove:
            # 1.
            # 1)
            # 1 -
            # -
            # *
            # •
            line = re.sub(
                r"^\s*(?:\d+\s*[\.\)\-:]\s*|[-*•]\s*)",
                "",
                line
            )

            line = line.strip()

            if line:
                lines.append(
                    line
                )

        master = eligible_pool.copy()

        master[
            "_normalized_name"
        ] = (
            master[
                "name"
            ]
            .fillna("")
            .astype(str)
            .map(
                normalize_uploaded_name
            )
        )

        lookup = (
            master
            .groupby(
                "_normalized_name"
            )["ID"]
            .apply(list)
            .to_dict()
        )

        matched_ids = []
        matched_names = []
        unmatched = []
        ambiguous = []

        for input_name in lines:

            normalized = (
                normalize_uploaded_name(
                    input_name
                )
            )

            matches = lookup.get(
                normalized,
                []
            )

            if len(matches) == 1:

                pid = int(
                    matches[0]
                )

                if pid not in matched_ids:

                    matched_ids.append(
                        pid
                    )

                    player_row = master[
                        master[
                            "ID"
                        ]
                        == pid
                    ]

                    if len(player_row):

                        matched_names.append(
                            str(
                                player_row.iloc[0][
                                    "name"
                                ]
                            )
                        )

            elif len(matches) > 1:

                ambiguous.append(
                    input_name
                )

            else:

                unmatched.append(
                    input_name
                )

        return {
            "player_ids": matched_ids,
            "player_names": matched_names,
            "unmatched_names": unmatched,
            "ambiguous_names": ambiguous
        }

    # --------------------------------------------------------
    # INPUT METHOD
    # --------------------------------------------------------

    squad_input_mode = st.radio(
        "How do you want to add the squad?",
        [
            "Upload squad file",
            "Enter numbered names",
            "Add players one-by-one"
        ],
        key="squad_input_method"
    )

    # ========================================================
    # OPTION 1 — UPLOAD FILE
    # ========================================================

    if (
        squad_input_mode
        == "Upload squad file"
    ):

        uploaded_file = st.file_uploader(
            "Upload squad list",
            type=[
                "csv",
                "xlsx",
                "xls",
                "txt"
            ],
            help=(
                "Upload player IDs or player names. "
                "TXT may contain one player per line."
            ),
            key="squad_upload_v2"
        )

        if uploaded_file is not None:

            try:

                # Reuse existing parser
                parsed = parse_uploaded_squad(
                    uploaded_file,
                    data["players"]
                )

                uploaded_ids = [
                    int(x)
                    for x in parsed[
                        "player_ids"
                    ]
                ]

                # Strictly validate against the selected pool
                allowed_ids = set(
                    base_eligible_squad[
                        "ID"
                    ]
                    .astype(int)
                )

                invalid_ids = (
                    set(uploaded_ids)
                    - allowed_ids
                )

                if invalid_ids:

                    raise ValueError(
                        "The uploaded squad contains "
                        "players outside the selected "
                        "team/country."
                    )

                upload_result = {
                    "player_ids": list(
                        dict.fromkeys(
                            uploaded_ids
                        )
                    ),
                    "player_names": (
                        parsed.get(
                            "matched_names",
                            []
                        )
                    ),
                    "unmatched_names": (
                        parsed.get(
                            "unmatched_names",
                            []
                        )
                    ),
                    "ambiguous_names": (
                        parsed.get(
                            "ambiguous_names",
                            []
                        )
                    )
                }

                st.success(
                    f"Matched "
                    f"{len(upload_result['player_ids'])} "
                    "eligible players."
                )

                if upload_result[
                    "unmatched_names"
                ]:

                    st.warning(
                        "Unmatched: "
                        + ", ".join(
                            upload_result[
                                "unmatched_names"
                            ][:10]
                        )
                    )

                if upload_result[
                    "ambiguous_names"
                ]:

                    st.warning(
                        "Ambiguous: "
                        + ", ".join(
                            upload_result[
                                "ambiguous_names"
                            ][:10]
                        )
                    )

            except Exception as e:

                st.error(
                    f"Could not process squad: {e}"
                )

    # ========================================================
    # OPTION 2 — NUMBERED NAME LIST
    # ========================================================

    elif (
        squad_input_mode
        == "Enter numbered names"
    ):

        st.caption(
            "Enter one player per line. Numbering is optional."
        )

        st.code(
            "1. Virat Kohli\n"
            "2. Rohit Sharma\n"
            "3. Jasprit Bumrah\n"
            "4. Hardik Pandya\n"
            "5. ...",
            language="text"
        )

        numbered_text = st.text_area(
            "Player names",
            placeholder=(
                "1. Player Name\n"
                "2. Player Name\n"
                "3. Player Name"
            ),
            height=180,
            key="numbered_player_names"
        )

        if numbered_text.strip():

            resolved = (
                resolve_numbered_names(
                    numbered_text,
                    base_eligible_squad
                )
            )

            upload_result = {
                "player_ids": (
                    resolved[
                        "player_ids"
                    ]
                ),
                "player_names": (
                    resolved[
                        "player_names"
                    ]
                ),
                "unmatched_names": (
                    resolved[
                        "unmatched_names"
                    ]
                ),
                "ambiguous_names": (
                    resolved[
                        "ambiguous_names"
                    ]
                )
            }

            st.write(
                f"Matched players: "
                f"**{len(upload_result['player_ids'])}**"
            )

            if upload_result[
                "player_names"
            ]:

                with st.expander(
                    "Matched players"
                ):

                    for i, name in enumerate(
                        upload_result[
                            "player_names"
                        ],
                        start=1
                    ):

                        st.write(
                            f"{i}. {name}"
                        )

            if upload_result[
                "unmatched_names"
            ]:

                st.warning(
                    "Unmatched: "
                    + ", ".join(
                        upload_result[
                            "unmatched_names"
                        ][:10]
                    )
                )

            if upload_result[
                "ambiguous_names"
            ]:

                st.warning(
                    "Ambiguous: "
                    + ", ".join(
                        upload_result[
                            "ambiguous_names"
                        ][:10]
                    )
                )

    # ========================================================
    # OPTION 3 — ONE-BY-ONE PLAYER ADDITION
    # ========================================================

    else:

        current_context = (
            f"{mode}_{selection_value}"
        )

        previous_context = (
            st.session_state.get(
                "manual_squad_context"
            )
        )

        if (
            previous_context
            != current_context
        ):

            st.session_state[
                "manual_squad_context"
            ] = current_context

            st.session_state[
                "manual_squad_ids"
            ] = []

        if (
            "manual_squad_ids"
            not in st.session_state
        ):

            st.session_state[
                "manual_squad_ids"
            ] = []

        pool = (
            base_eligible_squad
            .copy()
            .sort_values(
                "name"
            )
        )

        option_map = {}

        for _, row in pool.iterrows():

            pid = int(
                row["ID"]
            )

            label = (
                f"{row['name']} | "
                f"{row['playing_role_clean']} | "
                f"ID {pid}"
            )

            option_map[
                label
            ] = pid

        already_selected = set(
            st.session_state[
                "manual_squad_ids"
            ]
        )

        available = {
            label: pid
            for label, pid
            in option_map.items()
            if pid not in already_selected
        }

        st.caption(
            "Add players from the verified "
            "eligible team/country pool."
        )

        selected_option = st.selectbox(
            "Select player to add",
            [
                "— Select player —"
            ]
            + list(
                available.keys()
            ),
            key="add_player_select"
        )

        if st.button(
            "➕ Add Player",
            width="stretch",
            key="add_player_button"
        ):

            if (
                selected_option
                != "— Select player —"
            ):

                pid = available[
                    selected_option
                ]

                if pid not in (
                    st.session_state[
                        "manual_squad_ids"
                    ]
                ):

                    st.session_state[
                        "manual_squad_ids"
                    ].append(
                        pid
                    )

                st.rerun()

        current_ids = (
            st.session_state[
                "manual_squad_ids"
            ]
        )

        # Convert one-by-one selection into the same
        # upload_result structure used by the rest of app.
        upload_result = {
            "player_ids": list(
                current_ids
            ),
            "player_names": [],
            "unmatched_names": [],
            "ambiguous_names": []
        }

        st.write(
            f"Selected squad: "
            f"**{len(current_ids)} players**"
        )

        if current_ids:

            selected_df = pool[
                pool[
                    "ID"
                ].isin(
                    current_ids
                )
            ].copy()

            selected_df[
                "_order"
            ] = (
                selected_df[
                    "ID"
                ]
                .map(
                    {
                        pid: i
                        for i, pid
                        in enumerate(
                            current_ids
                        )
                    }
                )
            )

            selected_df = (
                selected_df
                .sort_values(
                    "_order"
                )
            )

            for i, (_, row) in enumerate(
                selected_df.iterrows(),
                start=1
            ):

                col_remove, col_name = (
                    st.columns(
                        [
                            1,
                            8
                        ]
                    )
                )

                with col_remove:

                    if st.button(
                        "✕",
                        key=(
                            f"remove_{int(row['ID'])}"
                        )
                    ):

                        st.session_state[
                            "manual_squad_ids"
                        ].remove(
                            int(row["ID"])
                        )

                        st.rerun()

                with col_name:

                    st.write(
                        f"{i}. "
                        f"{row['name']} "
                        f"— "
                        f"{row['playing_role_clean']}"
                    )

        else:

            st.info(
                "No players added yet."
            )

    # The existing Generate Best XI code below this
    # section uses upload_result["player_ids"].


    generate = st.button(
        "🚀 Generate Best XI",
        type="primary",
        width="stretch"
    )


# ============================================================
# MAIN AREA
# ============================================================

st.subheader(
    "4. AI Best XI"
)

if generate:

    try:

        if upload_result is None:

            raise ValueError(
                "Please upload the squad list first."
            )

        explicit_ids = (
            upload_result[
                "player_ids"
            ]
        )

        if len(explicit_ids) < 11:

            raise ValueError(
                f"Uploaded squad has only "
                f"{len(explicit_ids)} matched players. "
                "At least 11 are required."
            )

        # ----------------------------------------
        # Validate squad against selected team/country
        # ----------------------------------------

        eligible_squad = (
            get_eligible_squad(
                data,
                mode,
                selection_value,
                explicit_ids
            )
        )

        if len(eligible_squad) < 11:

            raise ValueError(
                "Valid eligible squad contains fewer than 11 players."
            )

        # ----------------------------------------
        # Score
        # ----------------------------------------

        with st.spinner(
            "Running ML suitability scoring..."
        ):

            scored = (
                score_squad_with_match_context(
                    data=data,
                    squad=eligible_squad,
                    mode=mode,
                    selection_value=selection_value,
                    selected_venue=selected_venue,
                    pitch_type=pitch_type,
                    toss_winner=toss_winner,
                    toss_decision=toss_decision
                )
            )

        # ----------------------------------------
        # Optimizer
        # ----------------------------------------

        with st.spinner(
            "Optimizing the Best Playing XI..."
        ):

            xi, composition = optimize_xi(
                scored,
                mode
            )

        st.session_state[
            "final_xi"
        ] = xi

        st.session_state[
            "scored"
        ] = scored

        st.session_state[
            "eligible_squad"
        ] = eligible_squad

        st.session_state[
            "composition"
        ] = composition

        st.session_state[
            "match_context"
        ] = {
            "mode": mode,
            "selection":
                (
                    selected_team_name
                    if mode == "IPL"
                    else selected_country
                ),
            "venue": selected_venue,
            "pitch": pitch_type,
            "toss_winner": toss_winner,
            "toss_decision": toss_decision
        }

        st.success(
            "Best XI generated successfully."
        )

    except Exception as e:

        st.error(
            f"Could not generate Best XI: {e}"
        )

        st.stop()


# ============================================================
# SHOW RESULTS
# ============================================================

if (
    "final_xi"
    in st.session_state
):

    xi = st.session_state[
        "final_xi"
    ]

    scored = st.session_state[
        "scored"
    ]

    eligible_squad = st.session_state[
        "eligible_squad"
    ]

    composition = st.session_state[
        "composition"
    ]

    context = st.session_state[
        "match_context"
    ]

    # --------------------------------------------------------
    # MATCH CONTEXT
    # --------------------------------------------------------

    st.subheader(
        "Match Context"
    )

    context_cols = st.columns(5)

    context_cols[0].metric(
        "Team / Country",
        context["selection"]
    )

    context_cols[1].metric(
        "Venue",
        context["venue"]
    )

    context_cols[2].metric(
        "Pitch",
        context["pitch"]
    )

    context_cols[3].metric(
        "Toss Winner",
        context["toss_winner"]
    )

    context_cols[4].metric(
        "Decision",
        context["toss_decision"]
    )

    # --------------------------------------------------------
    # COMPOSITION
    # --------------------------------------------------------

    st.subheader(
        "XI Composition"
    )

    metric_cols = st.columns(6)

    metric_cols[0].metric(
        "Eligible",
        len(eligible_squad)
    )

    metric_cols[1].metric(
        "Batters",
        composition["batters"]
    )

    metric_cols[2].metric(
        "Bowlers",
        composition["bowlers"]
    )

    metric_cols[3].metric(
        "All-rounders",
        composition["allrounders"]
    )

    metric_cols[4].metric(
        "Wicketkeepers",
        composition["wicketkeepers"]
    )

    metric_cols[5].metric(
        "Overseas",
        composition["overseas"]
    )

    # --------------------------------------------------------
    # AI XI
    # --------------------------------------------------------

    st.subheader(
        "🤖 AI Selected Best XI"
    )

    xi_display = xi[
        [
            "xi_position",
            "ID",
            "name",
            "country",
            "playing_role_clean",
            "suitability_score"
        ]
    ].copy()

    xi_display[
        "suitability_score"
    ] = (
        xi_display[
            "suitability_score"
        ].round(4)
    )

    st.dataframe(
        xi_display,
        width="stretch",
        hide_index=True,
        column_config={
            "xi_position": "XI Position",
            "ID": "Player ID",
            "name": "Player",
            "country": "Country",
            "playing_role_clean": "Role",
            "suitability_score":
                st.column_config.NumberColumn(
                    "Suitability",
                    format="%.4f"
                )
        }
    )

    # --------------------------------------------------------
    # FINAL USER SELECTION
    # --------------------------------------------------------

    st.subheader(
        "5. Review / Confirm Final XI"
    )

    scored_options = {}

    for _, row in scored.iterrows():

        label = (
            f"{row['name']} | "
            f"{row['playing_role_clean']} | "
            f"{row['suitability_score']:.4f}"
        )

        scored_options[
            label
        ] = int(
            row["ID"]
        )

    default_labels = [
        label
        for label, pid
        in scored_options.items()
        if pid in set(
            xi["ID"].astype(int)
        )
    ]

    selected_labels = st.multiselect(
        "Choose exactly 11 players",
        options=list(
            scored_options.keys()
        ),
        default=default_labels,
        max_selections=11
    )

    selected_ids = [
        scored_options[label]
        for label in selected_labels
    ]

    selected_players = scored[
        scored["ID"]
        .isin(selected_ids)
    ].copy()

    valid_manual, validation_message = (
        validate_manual_xi(
            selected_players,
            context["mode"]
        )
    )

    if valid_manual:

        st.success(
            "✅ Final XI satisfies all constraints."
        )

        st.write(
            validation_message
        )

        final_csv = (
            selected_players[
                [
                    "ID",
                    "name",
                    "country",
                    "playing_role_clean",
                    "suitability_score"
                ]
            ]
            .to_csv(index=False)
            .encode("utf-8")
        )

        st.download_button(
            "⬇️ Download Confirmed XI",
            data=final_csv,
            file_name="confirmed_best_playing_xi.csv",
            mime="text/csv"
        )

    elif len(selected_players) == 11:

        st.error(
            validation_message
        )

    else:

        st.warning(
            f"Select exactly 11 players. "
            f"Current selection: "
            f"{len(selected_players)}."
        )

    # --------------------------------------------------------
    # RANKING
    # --------------------------------------------------------

    with st.expander(
        "View full eligible-squad ranking"
    ):

        ranking = scored[
            [
                "eligible_rank",
                "ID",
                "name",
                "country",
                "playing_role_clean",
                "suitability_score"
            ]
        ].copy()

        ranking[
            "suitability_score"
        ] = (
            ranking[
                "suitability_score"
            ].round(4)
        )

        st.dataframe(
            ranking,
            width="stretch",
            hide_index=True
        )

    # --------------------------------------------------------
    # ============================================================
    # PHASE 11 - EXPLAINABILITY + FAIRNESS UI
    # ============================================================

    st.subheader(
        "🔍 Explainability & Fairness Audit"
    )

    xai_tabs = st.tabs(
        [
            "Global SHAP",
            "Player XAI",
            "Fairness Audit",
            "Drift Check"
        ]
    )

# ============================================================
    # DRIFT CHECK
    # ============================================================
    with xai_tabs[3]:

        st.markdown("### Train vs Test Drift Monitoring")

        drift_dir = (
            Path(__file__).resolve().parent.parent
            / "results"
            / "phase10_xai_fairness"
            / "drift_analysis"
        )

        drift_summary_file = (
            drift_dir / "train_vs_test_drift_summary.json"
        )

        drift_feature_file = (
            drift_dir / "train_vs_test_drift_feature_summary.csv"
        )

        if not drift_summary_file.exists():
            st.warning(
                "Drift summary artifact was not found."
            )

        elif not drift_feature_file.exists():
            st.warning(
                "Drift feature summary artifact was not found."
            )

        else:

            with open(
                drift_summary_file,
                "r",
                encoding="utf-8"
            ) as f:
                drift_summary = json.load(f)

            drift_features = pd.read_csv(
                drift_feature_file
            )

            st.markdown(
                "The drift check compares the historical training "
                "feature distributions with the later test-period "
                "distributions."
            )

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Features Checked",
                drift_summary.get(
                    "features_checked",
                    len(drift_features)
                )
            )

            c2.metric(
                "Low Drift",
                drift_summary.get(
                    "low_drift_features",
                    0
                )
            )

            c3.metric(
                "Moderate Drift",
                drift_summary.get(
                    "moderate_drift_features",
                    0
                )
            )

            c4.metric(
                "High Drift",
                drift_summary.get(
                    "high_drift_features",
                    0
                )
            )

            st.markdown("### Evaluation Periods")

            period_df = pd.DataFrame(
                {
                    "Dataset": [
                        "Training",
                        "Test"
                    ],
                    "Rows": [
                        drift_summary.get(
                            "train_rows",
                            0
                        ),
                        drift_summary.get(
                            "test_rows",
                            0
                        )
                    ],
                    "Start": [
                        drift_summary.get(
                            "train_date_min",
                            ""
                        ),
                        drift_summary.get(
                            "test_date_min",
                            ""
                        )
                    ],
                    "End": [
                        drift_summary.get(
                            "train_date_max",
                            ""
                        ),
                        drift_summary.get(
                            "test_date_max",
                            ""
                        )
                    ]
                }
            )

            st.dataframe(
                period_df,
                use_container_width=True,
                hide_index=True
            )

            st.markdown("### Drift Summary")

            d1, d2 = st.columns(2)

            d1.metric(
                "Mean PSI",
                f"{drift_summary.get('mean_psi', 0):.4f}"
            )

            d2.metric(
                "Maximum PSI",
                f"{drift_summary.get('max_psi', 0):.4f}"
            )

            st.markdown(
                "### Top Features by Distribution Shift"
            )

            top_drift = drift_features[
                [
                    "feature",
                    "type",
                    "psi",
                    "psi_level",
                    "missing_shift_pct",
                    "ks_statistic"
                ]
            ].head(15)

            st.dataframe(
                top_drift,
                use_container_width=True,
                hide_index=True
            )

            st.info(
                "High drift indicates that a feature's distribution "
                "changed between the historical training period and "
                "the later test period. Drift is a monitoring signal "
                "and does not by itself establish model failure."
            )
        # ============================================================

    # GLOBAL SHAP
    # ============================================================
    with xai_tabs[0]:

        st.markdown(
            "### Global SHAP Feature Importance"
        )

        global_shap_file = (
            SHAP_DIR
            / "shap_global_importance.csv"
        )

        global_shap_plot = (
            SHAP_DIR
            / "shap_global_top20_bar.png"
        )

        if global_shap_file.exists():

            global_shap = pd.read_csv(
                global_shap_file
            )

            st.dataframe(
                global_shap.head(20),
                width="stretch",
                hide_index=True
            )

        else:

            st.warning(
                "Global SHAP importance artifact "
                "is unavailable."
            )

        if global_shap_plot.exists():

            st.image(
                str(global_shap_plot),
                caption=(
                    "Top global SHAP features "
                    "from the production model."
                ),
                width="stretch"
            )

        st.info(
            "SHAP values explain model behavior. "
            "They are not causal effects."
        )


    # ============================================================
    # PLAYER XAI
    # ============================================================
    with xai_tabs[1]:

        st.markdown(
            "### Player-level explanation"
        )

        xai_options = (
            scored[
                [
                    "ID",
                    "name",
                    "playing_role_clean",
                    "country",
                    "suitability_score"
                ]
            ]
            .copy()
        )

        xai_options["label"] = (
            xai_options["name"].astype(str)
            + " | "
            + xai_options[
                "playing_role_clean"
            ].astype(str)
            + " | score="
            + xai_options[
                "suitability_score"
            ].round(4).astype(str)
        )

        xai_label_to_id = dict(
            zip(
                xai_options["label"],
                xai_options["ID"].astype(int)
            )
        )

        xai_selected_label = st.selectbox(
            "Choose a player to explain",
            options=list(
                xai_label_to_id.keys()
            ),
            key="xai_player_selector"
        )

        xai_player_id = int(
            xai_label_to_id[
                xai_selected_label
            ]
        )

        selected_xai_row = scored[
            scored["ID"].astype(int)
            == xai_player_id
        ].iloc[0]

        xai_cols = st.columns(2)

        xai_cols[0].metric(
            "Player",
            str(
                selected_xai_row["name"]
            )
        )

        xai_cols[1].metric(
            "Suitability",
            f"{float(
                selected_xai_row[
                    'suitability_score'
                ]
            ):.4f}"
        )

        explain_clicked = st.button(
            "Generate SHAP + LIME explanation",
            key="generate_player_xai"
        )

        if explain_clicked:

            # ----------------------------------------------------
            # SHAP
            # ----------------------------------------------------
            try:

                with st.spinner(
                    "Computing local SHAP..."
                ):

                    shap_df = (
                        _xai_local_shap(
                            scored,
                            xai_player_id,
                            data
                        )
                    )

                st.markdown(
                    "#### Local SHAP contributions"
                )

                st.dataframe(
                    shap_df.head(15),
                    width="stretch",
                    hide_index=True
                )

                positive_shap = (
                    shap_df[
                        shap_df["shap_value"] > 0
                    ]
                    .sort_values(
                        "shap_value",
                        ascending=False
                    )
                    .head(8)
                )

                negative_shap = (
                    shap_df[
                        shap_df["shap_value"] < 0
                    ]
                    .sort_values(
                        "shap_value",
                        ascending=True
                    )
                    .head(8)
                )

                shap_cols = st.columns(2)

                with shap_cols[0]:

                    st.markdown(
                        "##### Positive contributions"
                    )

                    st.dataframe(
                        positive_shap,
                        width="stretch",
                        hide_index=True
                    )

                with shap_cols[1]:

                    st.markdown(
                        "##### Negative contributions"
                    )

                    st.dataframe(
                        negative_shap,
                        width="stretch",
                        hide_index=True
                    )

            except Exception as e:

                st.error(
                    f"SHAP explanation failed: {e}"
                )


            # ----------------------------------------------------
            # LIME
            # ----------------------------------------------------
            try:

                with st.spinner(
                    "Computing local LIME..."
                ):

                    lime_df = (
                        _xai_local_lime(
                            scored,
                            xai_player_id,
                            data
                        )
                    )

                st.markdown(
                    "#### Local LIME contributions"
                )

                st.dataframe(
                    lime_df.head(15),
                    width="stretch",
                    hide_index=True
                )

                st.info(
                    "LIME is a local surrogate explanation "
                    "and is not a causal explanation."
                )

            except Exception as e:

                st.error(
                    f"LIME explanation failed: {e}"
                )


    # ============================================================
    # FAIRNESS AUDIT
    # ============================================================
    with xai_tabs[2]:

        st.markdown(
            "### Historical Fairness Audit"
        )

        regression_summary_file = (
            FAIRNESS_DIR
            / "fairlearn_regression_summary.json"
        )

        regression_groups_file = (
            FAIRNESS_DIR
            / "fairlearn_country_group_metrics.csv"
        )

        selection_groups_file = (
            MITIGATION_DIR
            / "step121_baseline_vs_mitigation_fairness_groups.csv"
        )

        decision_file = (
            XAI_DIR
            / "phase10_final_model_decision.json"
        )

        # --------------------------------------------------------
        # Regression fairness
        # --------------------------------------------------------
        if regression_summary_file.exists():

            try:

                with open(
                    regression_summary_file,
                    "r",
                    encoding="utf-8"
                ) as f:

                    regression_summary = (
                        json.load(f)
                    )

                st.markdown(
                    "#### Regression fairness summary"
                )

                summary_rows = []

                if isinstance(
                    regression_summary,
                    dict
                ):

                    for key, value in (
                        regression_summary.items()
                    ):

                        if isinstance(
                            value,
                            (int, float, str)
                        ):

                            summary_rows.append(
                                {
                                    "Metric":
                                        str(key),
                                    "Value":
                                        str(value)
                                }
                            )

                if summary_rows:

                    st.dataframe(
                        pd.DataFrame(
                            summary_rows
                        ),
                        width="stretch",
                        hide_index=True
                    )

            except Exception as e:

                st.warning(
                    f"Could not load regression "
                    f"fairness summary: {e}"
                )


        # --------------------------------------------------------
        # Country regression metrics
        # --------------------------------------------------------
        if regression_groups_file.exists():

            st.markdown(
                "#### Country-level regression metrics"
            )

            fairness_groups = pd.read_csv(
                regression_groups_file
            )

            st.dataframe(
                fairness_groups,
                width="stretch",
                hide_index=True
            )


        # --------------------------------------------------------
        # XI selection fairness
        # --------------------------------------------------------
        if selection_groups_file.exists():

            st.markdown(
                "#### Baseline vs mitigation "
                "XI-selection audit"
            )

            selection_groups = pd.read_csv(
                selection_groups_file
            )

            st.dataframe(
                selection_groups,
                width="stretch",
                hide_index=True
            )


        # --------------------------------------------------------
        # XI selection fairness summary
        # --------------------------------------------------------
        selection_summary_file = (
            FAIRNESS_DIR
            / "xi_selection_fairness_summary.json"
        )

        if selection_summary_file.exists():

            try:

                with open(
                    selection_summary_file,
                    "r",
                    encoding="utf-8"
                ) as f:
                    selection_summary = json.load(f)

                st.markdown(
                    "#### XI-selection fairness summary"
                )

                selection_summary_rows = [
                    {
                        "Metric": "Team-match groups",
                        "Value": selection_summary.get(
                            "team_match_groups",
                            "Unavailable"
                        )
                    },
                    {
                        "Metric": "Baseline selection rate",
                        "Value": (
                            f"{selection_summary.get('baseline_selection_rate', 0) * 100:.4f}%"
                        )
                    },
                    {
                        "Metric": "Demographic parity difference",
                        "Value": selection_summary.get(
                            "demographic_parity_difference",
                            "Unavailable"
                        )
                    },
                    {
                        "Metric": "Equalized odds difference",
                        "Value": selection_summary.get(
                            "equalized_odds_difference",
                            "Unavailable"
                        )
                    }
                ]

                st.dataframe(
                    pd.DataFrame(
                        selection_summary_rows
                    ),
                    width="stretch",
                    hide_index=True
                )

                st.info(
                    "The historical XI-selection audit uses a "
                    "participation proxy because an official "
                    "historical Playing XI field was not available."
                )

            except Exception as e:

                st.warning(
                    f"Could not load XI-selection fairness summary: {e}"
                )

        # --------------------------------------------------------
        # Final model decision
        # --------------------------------------------------------
        if decision_file.exists():

            try:

                with open(
                    decision_file,
                    "r",
                    encoding="utf-8"
                ) as f:

                    decision = json.load(f)

                st.markdown(
                    "#### Production model decision"
                )

                decision_text = str(
                    decision.get(
                        "decision",
                        "Decision unavailable"
                    )
                )

                if (
                    "RETAIN_BASELINE"
                    in decision_text
                ):

                    st.success(
                        decision_text
                    )

                else:

                    st.info(
                        decision_text
                    )

            except Exception as e:

                st.warning(
                    f"Could not load final "
                    f"model decision: {e}"
                )


        st.warning(
            "Fairness results are historical audit results. "
            "The XI-selection audit uses a participation proxy "
            "rather than an official historical Playing XI record."
        )

    # MODEL NOTE
    # --------------------------------------------------------

    st.info(
        "The model uses the trained pre-match feature set. "
        "Venue is supplied through the trained venue feature. "
        "Pitch type and toss are captured in the dashboard "
        "for match context but are not currently ML features. "
        "Batting order, bowling order, captain and vice-captain "
        "are not predicted in this version."
    )

else:

    st.info(
        "Select an IPL team or country, enter the match "
        "context, upload the squad, and click "
        "**Generate Best XI**."
    )

    cards = st.columns(3)

    cards[0].metric(
        "Player Master",
        "4,827"
    )

    cards[1].metric(
        "Verified Teams",
        "10"
    )

    cards[2].metric(
        "Countries",
        "43"
    )

    st.subheader(
        "System Workflow"
    )

    st.markdown(
        """
        **1. Select team/country**

        **2. Enter venue, pitch and toss**

        **3. Upload the eligible squad**

        **4. ML scores only those players**

        **5. Optimizer selects exactly 11**

        **6. Review and confirm the final XI**
        """
    )