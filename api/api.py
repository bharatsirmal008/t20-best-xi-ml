import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from scipy.optimize import Bounds, LinearConstraint, milp

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

PLAYER_FILE = DATA_DIR / "players_performance_final.csv"

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
# IPL TEAM MAPPING
# Same team IDs used by the production Streamlit project
# ============================================================

IPL_TEAM_NAMES = {
    1: "Royal Challengers Bengaluru",
    2: "Sunrisers Hyderabad",
    3: "Mumbai Indians",
    4: "Rising Pune Supergiant",
    5: "Gujarat Lions",
    6: "Kolkata Knight Riders",
    129: "Chennai Super Kings",
    134: "Rajasthan Royals",
    252: "Delhi Capitals",
    494: "Punjab Kings",
    614: "Lucknow Super Giants",
    615: "Gujarat Titans",
    1414: "Kochi Tuskers Kerala",
    1419: "Pune Warriors"
}


# Optional common aliases
IPL_TEAM_ALIASES = {
    "rcb": "Royal Challengers Bengaluru",
    "royal challengers bangalore": "Royal Challengers Bengaluru",

    "srh": "Sunrisers Hyderabad",

    "mi": "Mumbai Indians",

    "rps": "Rising Pune Supergiant",
    "rising pune supergiants": "Rising Pune Supergiant",

    "gl": "Gujarat Lions",

    "kkr": "Kolkata Knight Riders",

    "csk": "Chennai Super Kings",

    "rr": "Rajasthan Royals",

    "dc": "Delhi Capitals",
    "delhi daredevils": "Delhi Capitals",

    "pbks": "Punjab Kings",
    "kxip": "Punjab Kings",
    "kings xi punjab": "Punjab Kings",

    "lsg": "Lucknow Super Giants",

    "gt": "Gujarat Titans",

    "ktk": "Kochi Tuskers Kerala",

    "pw": "Pune Warriors",
    "pune warriors india": "Pune Warriors"
}


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="T20/T20I Best Playing XI API",
    description=(
        "Production API for T20/T20I player suitability "
        "scoring and Best Playing XI prediction."
    ),
    version="1.0.0"
)


# ============================================================
# REQUEST SCHEMA
# ============================================================

class PredictRequest(BaseModel):

    mode: str = Field(
        ...,
        description="Prediction mode: IPL or Country"
    )

    team: str | None = Field(
        default=None,
        description="IPL team name"
    )

    country: str | None = Field(
        default=None,
        description="Country name for country mode"
    )

    venue: str = Field(
        ...,
        description="Match venue"
    )

    pitch: str = Field(
        ...,
        description="Pitch type"
    )

    toss_winner: str = Field(
        ...,
        description="Toss winner"
    )

    toss_decision: str = Field(
        ...,
        description="Bat or Bowl"
    )

    squad: list[str] = Field(
        ...,
        min_length=11,
        description="User-provided eligible squad"
    )


# ============================================================
# GLOBAL DATA CACHE
# ============================================================

_DATA = None


# ============================================================
# HELPERS
# ============================================================

def extract_features(
    definition,
    keys,
    fallback=None
):

    if isinstance(definition, dict):

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


# ============================================================
# ARTIFACT LOADING
# ============================================================

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
# PREPARE PRODUCTION DATA
# ============================================================

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

    # --------------------------------------------------------
    # Player master
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Historical data
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Training membership
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Feature definitions
    # --------------------------------------------------------

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


def get_data():

    global _DATA

    if _DATA is None:

        _DATA = prepare_data()

    return _DATA


# ============================================================
# TEAM RESOLUTION
# ============================================================

def resolve_team_id(
    team_name,
    team_catalog
):

    raw_name = str(
        team_name
    ).strip()

    # --------------------------------------------------------
    # Numeric team ID
    # --------------------------------------------------------

    try:

        numeric_id = int(
            raw_name
        )

        valid_ids = set(
            team_catalog["team_id"]
            .astype(int)
        )

        if numeric_id in valid_ids:

            return numeric_id

    except (
        ValueError,
        TypeError
    ):

        pass

    # --------------------------------------------------------
    # Normalize aliases
    # --------------------------------------------------------

    normalized = (
        raw_name
        .lower()
        .strip()
    )

    normalized = IPL_TEAM_ALIASES.get(
        normalized,
        normalized
    )

    # --------------------------------------------------------
    # Match canonical project mapping
    # --------------------------------------------------------

    for team_id, canonical_name in IPL_TEAM_NAMES.items():

        if normalized == canonical_name.lower():

            return team_id

    # --------------------------------------------------------
    # Final catalog fallback
    # --------------------------------------------------------

    if "verified_team_name" in team_catalog.columns:

        for team_id in team_catalog["team_id"]:

            if normalized == str(
                team_id
            ).strip().lower():

                return int(team_id)

    raise ValueError(
        f"Invalid IPL team: {raw_name}"
    )


# ============================================================
# PLAYER NAME → PLAYER ID
# ============================================================

def resolve_player_ids(
    squad_names,
    players
):

    resolved_ids = []
    unresolved = []

    player_ids = set(
        players["ID"].astype(int)
    )

    for raw_name in squad_names:

        name = str(
            raw_name
        ).strip()

        if not name:

            unresolved.append(
                raw_name
            )

            continue

        # ----------------------------------------------------
        # Allow player ID
        # ----------------------------------------------------

        try:

            numeric_id = int(
                name
            )

            if numeric_id in player_ids:

                resolved_ids.append(
                    numeric_id
                )

                continue

        except (
            ValueError,
            TypeError
        ):

            pass

        # ----------------------------------------------------
        # Player name
        # ----------------------------------------------------

        matches = players[
            players["name"]
            .astype(str)
            .str.strip()
            .str.lower()
            == name.lower()
        ]

        if len(matches) == 0:

            unresolved.append(
                raw_name
            )

        elif len(matches) > 1:

            raise ValueError(
                f"Multiple players found for name: {raw_name}"
            )

        else:

            resolved_ids.append(
                int(
                    matches.iloc[0]["ID"]
                )
            )

    if unresolved:

        raise ValueError(
            "Players not found: "
            + ", ".join(
                map(
                    str,
                    unresolved
                )
            )
        )

    resolved_ids = list(
        dict.fromkeys(
            resolved_ids
        )
    )

    if len(resolved_ids) < 11:

        raise ValueError(
            "Squad must contain at least 11 unique players."
        )

    return resolved_ids


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

    # --------------------------------------------------------
    # Explicit squad
    # --------------------------------------------------------

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
                ].astype(int)
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

    # --------------------------------------------------------
    # IPL
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # COUNTRY
    # --------------------------------------------------------

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
# HISTORICAL SNAPSHOTS
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
# PRODUCTION MODEL SCORING
# ============================================================

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

    # --------------------------------------------------------
    # Venue
    # --------------------------------------------------------

    if "venue_standard" in raw_features:

        model_input[
            "venue_standard"
        ] = selected_venue

    # --------------------------------------------------------
    # Categories
    # --------------------------------------------------------

    for col in categorical_features:

        if col in model_input.columns:

            model_input[col] = (
                clean_category(
                    model_input[col]
                )
            )

    # --------------------------------------------------------
    # Numeric values
    # --------------------------------------------------------

    for col in numeric_features:

        if col in model_input.columns:

            model_input[col] = pd.to_numeric(
                model_input[col],
                errors="coerce"
            )

    # --------------------------------------------------------
    # Raw model matrix
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Production preprocessing
    # --------------------------------------------------------

    X_processed = (
        preprocessor.transform(
            X_raw
        )
    )

    # --------------------------------------------------------
    # Production model prediction
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Attach predictions
    # --------------------------------------------------------

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

    return result


# ============================================================
# XI OPTIMIZER
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
# JSON SERIALIZATION
# ============================================================
def clean_json_value(value):

    if value is None:
        return None

    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False

    if isinstance(missing, (bool, np.bool_)) and missing:
        return None

    if isinstance(value, np.generic):
        return value.item()

    return value

def dataframe_records(
    df,
    columns
):

    available = [
        c
        for c in columns
        if c in df.columns
    ]

    records = []

    for row in df[
        available
    ].to_dict(
        orient="records"
    ):

        records.append(
            {
                key: clean_json_value(value)
                for key, value in row.items()
            }
        )

    return records


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():

    try:

        data = get_data()

        return {
            "status": "ok",

            "service": (
                "T20/T20I Best Playing XI API"
            ),

            "model": type(
                data["model"]
            ).__name__,

            "players_loaded": len(
                    data["players"]
                ),

            "model_features": len(
                    data["raw_features"]
                )
        }

    except Exception as exc:  # noqa: BLE001

        raise HTTPException(
            status_code=500,
            detail=(
                "Artifact loading failed: "
                f"{exc}"
            )
        )


# ============================================================
# PREDICT ENDPOINT
# ============================================================

@app.post("/predict")
def predict(
    request: PredictRequest
):

    try:

        data = get_data()

        # ----------------------------------------------------
        # Mode
        # ----------------------------------------------------

        mode = (
            str(request.mode)
            .strip()
            .upper()
        )

        if mode not in {
            "IPL",
            "COUNTRY"
        }:

            raise ValueError(
                "mode must be IPL or Country."
            )

        # ----------------------------------------------------
        # Resolve team/country
        # ----------------------------------------------------

        if mode == "IPL":

            if not request.team:

                raise ValueError(
                    "team is required for IPL mode."
                )

            team_id = resolve_team_id(
                request.team,
                data["team_catalog"]
            )

            selection_value = team_id

            selection_name = (
                IPL_TEAM_NAMES.get(
                    team_id,
                    request.team
                )
            )

        else:

            if not request.country:

                raise ValueError(
                    "country is required for Country mode."
                )

            country = (
                str(request.country)
                .strip()
            )

            selection_value = country
            selection_name = country

        # ----------------------------------------------------
        # Resolve player names
        # ----------------------------------------------------

        explicit_ids = resolve_player_ids(
            request.squad,
            data["players"]
        )

        # ----------------------------------------------------
        # Eligibility
        # ----------------------------------------------------

        eligible_squad = get_eligible_squad(
            data=data,
            mode=mode,
            selection_value=selection_value,
            explicit_ids=explicit_ids
        )

        if len(eligible_squad) < 11:

            raise ValueError(
                "Eligible squad contains fewer than 11 players."
            )

        # ----------------------------------------------------
        # Production ML scoring
        # ----------------------------------------------------

        scored = score_squad_with_match_context(
            data=data,
            squad=eligible_squad,
            mode=mode,
            selection_value=selection_value,
            selected_venue=request.venue,
            pitch_type=request.pitch,
            toss_winner=request.toss_winner,
            toss_decision=request.toss_decision
        )

        # ----------------------------------------------------
        # Production optimizer
        # ----------------------------------------------------

        xi, composition = optimize_xi(
            scored=scored,
            mode=mode
        )

        # ----------------------------------------------------
        # Response columns
        # ----------------------------------------------------

        xi_columns = [
            "xi_position",
            "ID",
            "name",
            "country",
            "playing_role_clean",
            "suitability_score",
            "eligible_rank"
        ]

        scored_columns = [
            "ID",
            "name",
            "country",
            "playing_role_clean",
            "suitability_score",
            "eligible_rank"
        ]

        # ----------------------------------------------------
        # Final response
        # ----------------------------------------------------

        return {
            "status": "success",

            "message": (
                "Best Playing XI generated using the "
                "production HistGradientBoosting model "
                "and MILP optimizer."
            ),

            "mode": request.mode,

            "selection": selection_name,

            "team_id": (
                int(selection_value)
                if mode == "IPL"
                else None
            ),

            "country": (
                selection_name
                if mode == "COUNTRY"
                else None
            ),

            "match_context": {
                "venue": request.venue,
                "pitch": request.pitch,
                "toss_winner": request.toss_winner,
                "toss_decision": request.toss_decision
            },

            "input_squad_size": len(
                request.squad
            ),

            "eligible_squad_size": len(
                eligible_squad
            ),

            "xi_size": len(xi),

            "composition": composition,

            "best_xi": dataframe_records(
                xi,
                xi_columns
            ),

            "scored_squad": dataframe_records(
                scored,
                scored_columns
            )
        }

    except HTTPException:

        raise

    except Exception as exc:  # noqa: BLE001

        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )