from fastapi.testclient import TestClient

from api.api import app

client = TestClient(app)


CSK_SQUAD = [
    "MS Dhoni",
    "Brendon McCullum",
    "Suresh Raina",
    "Robin Uthappa",
    "Ajinkya Rahane",
    "Michael Hussey",
    "Piyush Chawla",
    "Ravichandran Ashwin",
    "Ravindra Jadeja",
    "Shane Watson",
    "Harbhajan Singh",
    "Ashish Nehra",
    "Lakshmipathy Balaji",
    "Manpreet Gony",
    "Imran Tahir",
]


def test_health():

    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["model"] == "HistGradientBoostingRegressor"
    assert data["players_loaded"] == 4827
    assert data["model_features"] == 63


def test_predict_best_xi():

    payload = {
        "mode": "IPL",
        "team": "Chennai Super Kings",
        "country": None,
        "venue": "Wankhede Stadium",
        "pitch": "Balanced",
        "toss_winner": "Chennai Super Kings",
        "toss_decision": "Bat",
        "squad": CSK_SQUAD,
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "success"
    assert data["team_id"] == 129
    assert data["input_squad_size"] == 15
    assert data["eligible_squad_size"] == 15
    assert data["xi_size"] == 11

    composition = data["composition"]

    assert composition["xi_size"] == 11
    assert composition["batters"] >= 3
    assert composition["bowlers"] >= 3
    assert composition["allrounders"] <= 4
    assert composition["wicketkeepers"] >= 1
    assert composition["wicketkeepers"] <= 2
    assert composition["overseas"] <= 4

    best_xi = data["best_xi"]

    assert len(best_xi) == 11

    selected_names = {
        player["name"]
        for player in best_xi
    }

    assert selected_names.issubset(
        set(CSK_SQUAD)
    )
