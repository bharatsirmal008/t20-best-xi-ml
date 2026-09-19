# Responsible AI Report — T20/T20I Best Playing XI ML System

## 1. Project Overview

This project predicts player suitability for selection in a T20/T20I Playing XI and then uses a constraint-based optimizer to construct a valid XI.

The system is intended as a decision-support tool. It is not designed to replace cricket selectors, coaches, analysts, or other human decision-makers.

## 2. Intended Use

The system can:

- score eligible players using pre-match performance and contextual features;
- rank eligible players by predicted suitability;
- generate an XI subject to role and competition constraints;
- provide model explanations using SHAP and LIME;
- provide fairness audit information;
- report distribution drift between historical training data and later test data.

The output should be treated as an analytical recommendation generated from historical data, not as an objective statement about a player's overall ability.

## 3. Data and Leakage Controls

The prediction pipeline uses pre-match features.

The project uses a chronological data split:

- Training: 2008-04-20 to 2021-10-13
- Validation: 2021-10-15 to 2024-04-10
- Test: 2024-04-11 to 2026-05-01

The temporal separation is intended to evaluate performance on later matches rather than randomly mixing observations across time.

## 4. Explainability

The project uses:

### SHAP

SHAP is used to identify feature contributions to individual predictions and overall model behavior.

Global SHAP analysis identifies the features that contribute most strongly to model predictions.

Local SHAP analysis is used to explain individual player predictions.

### LIME

LIME is used as an additional local explanation method for individual player predictions.

SHAP and LIME explanations should be interpreted as model explanations, not causal explanations.

## 5. Fairness

The project includes fairness auditing using group-based prediction and selection analysis.

The audit considers country/eligibility-related groups and selection outcomes.

The project also evaluates feature ablations involving country and India-eligibility information.

Fairness metrics are reported as measurements of the model's behavior on the evaluated data. They should not be interpreted as proof that the system is universally fair across all players, countries, competitions, or future seasons.

## 6. Selection Constraints

The optimizer generates exactly 11 players.

For IPL mode:

- XI size = 11
- Batters: minimum 3, maximum 7
- Bowlers: minimum 3, maximum 6
- All-rounders: minimum 0, maximum 4
- Wicketkeepers: minimum 1, maximum 2
- Overseas players: maximum 4

For country mode, the overseas-player restriction is not applied.

These are explicit optimization constraints and are separate from the ML prediction model.

## 7. Data Drift Monitoring

A train-vs-test drift analysis was performed using 51 model input features.

Results:

- Low drift features: 33
- Moderate drift features: 4
- High drift features: 14
- Mean PSI: 1.2168992619957901
- Maximum PSI: 10.235632703797165

The strongest distribution changes occurred in historical match-count, team/opponent context, venue, and related features.

Examples include:

- pre_team_match_count
- pre_team_matches
- pre_opponent_match_count
- pre_team_vs_opponent_matches
- venue_standard
- opponent_team_id
- pre_venue_avg_runs
- pre_recent5_venue_avg_runs

High drift does not by itself establish model failure. Some historical features are expected to change over time because teams, venues, players, and match histories evolve.

The drift result should therefore be treated as a monitoring signal.

## 8. Recommended Drift Response

When the system is updated with substantially newer cricket data:

1. recompute the drift analysis;
2. compare new validation/test performance;
3. inspect the features with the largest distribution changes;
4. check whether feature definitions remain valid;
5. investigate possible competition, team, venue, or rule changes;
6. retrain or recalibrate the model when evidence shows that performance has materially degraded.

## 9. Human Oversight

The final Playing XI should remain subject to human review.

A selector or analyst may consider information that is unavailable to the model, including:

- current fitness or availability;
- tactical matchups;
- pitch and weather information;
- recent injuries;
- workload;
- team strategy;
- squad balance;
- competition-specific rules.

The system should therefore be used as decision support rather than autonomous selection.

## 10. Privacy and Data Protection

Only data necessary for the modeling and analytical objectives should be collected and retained.

Personally identifying information should not be included unless there is a documented reason and appropriate authorization.

Any external data source should be reviewed for its collection terms, licensing requirements, and permitted use.

## 11. Consent and Responsible Data Use

Where personal or sensitive information is introduced into future versions of the system, the project should establish:

- a legitimate purpose for collection;
- appropriate consent or other lawful basis;
- data minimization;
- controlled access;
- secure storage;
- retention and deletion procedures.

The current system is primarily based on cricket performance and match-context information.

## 12. Limitations

Important limitations include:

- historical data may not represent future playing conditions;
- distribution drift is present between the training and test periods;
- model predictions are statistical estimates rather than causal judgments;
- fairness metrics depend on the available groups and sample sizes;
- the optimizer can only enforce the constraints explicitly encoded in the system;
- explanations describe model behavior and do not establish causality;
- unseen future players, teams, venues, formats, or rule changes may reduce reliability.

## 13. Transparency

The project maintains artifacts for:

- model configuration;
- preprocessing;
- feature definitions;
- model selection;
- XAI analysis;
- fairness analysis;
- drift analysis;
- API deployment;
- CI/CD testing;
- Streamlit dashboard.

## 14. Responsible Deployment Checklist

Before a production release:

- [x] Chronological train/validation/test split
- [x] Pre-match feature design
- [x] SHAP explainability
- [x] LIME local explainability
- [x] Fairness audit
- [x] Feature ablation analysis
- [x] Drift analysis
- [x] API health testing
- [x] Automated tests
- [x] Lint checks
- [x] CI/CD workflow
- [x] Human oversight statement

Future monitoring should include model performance, feature drift, data quality, fairness metrics, and changes in cricket competition context.

## 15. Final Responsible AI Position

The T20/T20I Best XI system is designed as an explainable, auditable decision-support system.

Its outputs should be interpreted together with model limitations, drift measurements, fairness audit results, and human cricket expertise.

The system should not be treated as an autonomous authority for player selection.
