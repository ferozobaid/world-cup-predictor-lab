# World Cup Predictor Lab

A Next.js football analytics app that predicts World Cup-style matchups from historical FIFA World Cup match data and uses the OpenAI API only for short cached analyst explanations.

The app now supports three prediction modes:

- **Historical Local**: the original transparent heuristic model.
- **Experimental ML**: a lightweight Logistic Regression baseline trained offline in Python and exported to JSON for the frontend.
- **Elo + Score**: a second-generation offline score model that builds chronological Elo-style ratings, predicts expected goals, then converts score distributions into win/draw/loss probabilities.

It also includes an optional **Modern Squad Strength** adjustment layer. This toggle is off by default and applies a small curated squad-strength proxy after the selected base model has already produced its prediction.

## Local Setup

```bash
npm install
npm run generate:data
npm run dev
```

To regenerate the ML outputs:

```bash
python -m pip install -r requirements.txt
python ml/train_model.py
```

To regenerate the curated squad-strength export:

```bash
python ml/squad_strength/build_squad_strength.py
```

The predictor works without an API key. To enable AI explanations, set:

```bash
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-5-nano
```

## Data

- Historical match data: Joshua Fjelstul's World Cup Database, `matches.csv`.
- 2026 fixtures: openfootball `worldcup.json`.
- 2026 emblem asset: Wikimedia Commons `2026 FIFA World Cup emblem.svg`, CC BY-SA 4.0.

Raw source snapshots are in `public/data/`. The compact app snapshot is generated into `src/data/world-cup-data.json`.

## ML Training

The ML pipeline lives in `ml/train_model.py`, with a companion notebook stub at `notebooks/world_cup_model_training.ipynb`.

### Target Variable

Each historical match is classified as:

- `teamA_win`
- `draw`
- `teamB_win`

Score equality is treated as a draw, even in knockout matches.

### Feature Engineering

Features are generated chronologically to avoid data leakage. For every match, the model only sees information available before that match:

- historical win rate before the match
- goals scored per match before the match
- goals conceded per match before the match
- goal difference per match before the match
- recent form index
- tournament experience
- pre-match Elo-style team rating
- host/co-host proxy
- tournament stage importance
- knockout match flag
- final flag
- strength, form, goal-balance, and experience differences

### Models and Evaluation

The training script fits:

- Logistic Regression as the explainable experimental classifier.
- Random Forest Classifier as a nonlinear comparison.
- Calibrated Gradient Boosting as another classifier comparison.
- Elo + Poisson Score Model as the second-generation frontend model.
- Majority-class baseline.
- Historical-strength baseline.

Evaluation uses a chronological split: training before 2014 and testing on 2014-2022. Metrics include accuracy, macro F1, log loss, Brier score, confusion matrix, and classification report where applicable.

The current Logistic Regression model is intentionally documented as educational/experimental because it does not beat the historical-strength baseline. The Elo + Score model is a better football-shaped approach because the scoreline and probabilities come from the same expected-goals distribution, but it is still not promoted to default unless it beats the historical baseline on macro F1 or probabilistic quality. Its probabilities are model-estimated probabilities, not betting odds.

### Frontend Export

The Next.js app does not train models at runtime. `ml/train_model.py` exports static JSON files in `ml/model_outputs/`:

- `model_metrics.json`
- `feature_importance.json`
- `team_features.json`
- `prediction_examples.json`

`src/lib/ml-prediction-model.ts` reads those JSON files and reproduces both Logistic Regression probabilities and the Elo + Score expected-goals model in TypeScript. No model training runs in Next.js or Vercel.

## Modern Squad Strength Layer

Phase 2 adds a small manually curated CSV at `ml/squad_strength/modern_squad_strength.csv`. It covers a limited sample of major modern national teams and exports `src/data/squad-strength.json` for the Next.js app.

The CSV schema is:

```text
snapshot_id,snapshot_year,as_of_date,competition_source,source_type,source_url,raw_team_name,canonical_team,player_count,fifa_rank,fifa_points,squad_market_value_eur_m,top_11_market_value_eur_m,top_5_market_value_eur_m,avg_age,total_caps,avg_caps,club_strength_index,recent_competitive_form,data_quality,notes
```

The export script computes a 0-100 `squad_strength_score` from normalized proxy fields:

- FIFA points
- log-scaled squad market value
- log-scaled top-11 market value
- total caps
- club strength index
- recent competitive form

When the UI toggle is enabled, the app applies a capped display adjustment after the selected base model:

- `Historical Local`, `Experimental ML`, or `Elo + Score` runs first.
- The squad layer shifts at most 8 percentage points between the two win probabilities.
- Draw probability is reduced by at most 3 points for large squad-strength gaps.
- The probabilities are normalized back to 100.

If either team is missing from the curated CSV, no adjustment is applied and the UI shows that squad data is unavailable. The layer does not scrape websites, does not fetch live data, and does not replace the base models.

### Limitations

World Cup history is a small dataset for supervised ML. Probability calibration may be weak, team eras change dramatically, and many teams have limited samples. The ML mode is best understood as a portfolio-friendly, explainable ML layer rather than a production-grade forecasting system.

The Modern Squad Strength layer is also a proxy, not a player-level production model. Its current values are manually curated sample/proxy values for portfolio demonstration. Market values, squad age, caps, FIFA ranking points, club strength, and recent form are imperfect indicators of actual match strength. Official 2026 squads are not final in this dataset, and the current coverage is intentionally limited.

All probabilities in the app are model-estimated probabilities, not betting odds.

## Tournament Monte Carlo Simulation

The "Monte Carlo tournament outlook" panel in the app is sourced from a static offline artifact at `ml/model_outputs/worldcup_simulation.json`. It is generated by `ml/simulate_tournament.py`.

Run it manually:

```bash
python3 ml/simulate_tournament.py --runs 100000 --seed 42
```

Key points:

- **Offline-only.** The Next.js app never runs simulations at request time — it imports the JSON statically.
- **Tournament view only.** Monte Carlo outputs (champion / finalist / semifinal / quarterfinal / knockout probabilities per team) drive the tournament outlook section. They are **not** used for single-match prediction probabilities.
- **Probability source is Elo / team strength.** The simulator draws match results from an Elo-based logistic over `ml/model_outputs/team_strengths.json`, not from the Calibrated ML (CatBoost) model. Single-match predictions in the UI continue to use Calibrated ML via `ml/model_outputs/matchup_predictions.json`.
- **Default count is 100,000 runs.** Controlled by `SIM_RUNS` in `ml/simulate_tournament.py` and overridable via `--runs`. The current 100k run completes in ~38s on a modern laptop.
- **Deterministic.** Fixed `--seed 42` so re-runs are reproducible.
- The simulator depends only on the Python standard library plus the 2026 fixture in `ml/data/worldcup_2026/worldcup.json` and the team-strength artifact above. No `.joblib`, CSV, or model file is loaded at runtime by the frontend.
