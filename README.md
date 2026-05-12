# World Cup Predictor Lab

A Next.js football analytics app that predicts World Cup-style matchups from historical FIFA World Cup match data and uses the OpenAI API only for short cached analyst explanations.

The app now supports two prediction modes:

- **Historical Local**: the original transparent heuristic model.
- **Tournament ML**: a lightweight supervised ML model trained offline in Python and exported to JSON for the frontend.

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
- knockout match flag
- final flag
- strength, form, goal-balance, and experience differences

### Models and Evaluation

The training script fits:

- Logistic Regression as the explainable frontend-exported model.
- Random Forest Classifier as a nonlinear comparison.
- Majority-class baseline.
- Historical-strength baseline.

Evaluation uses a chronological split: training before 2014 and testing on 2014-2022. Metrics include accuracy, macro F1, confusion matrix, and classification report.

The current Logistic Regression model is intentionally documented as educational/experimental if it does not beat the historical-strength baseline. Its probabilities are model-estimated probabilities, not betting odds.

### Frontend Export

The Next.js app does not train models at runtime. `ml/train_model.py` exports static JSON files in `ml/model_outputs/`:

- `model_metrics.json`
- `feature_importance.json`
- `team_features.json`
- `prediction_examples.json`

`src/lib/ml-prediction-model.ts` reads those JSON files and reproduces Logistic Regression probabilities in TypeScript.

### Limitations

World Cup history is a small dataset for supervised ML. Probability calibration may be weak, team eras change dramatically, and many teams have limited samples. The ML mode is best understood as a portfolio-friendly, explainable ML layer rather than a production-grade forecasting system.
