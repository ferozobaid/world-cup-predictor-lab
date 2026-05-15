"""Optional Plotly chart generation for analytics review."""

from __future__ import annotations

import json

import pandas as pd

from src.config import CONFIG, PipelineConfig
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def generate_charts(config: PipelineConfig = CONFIG) -> dict[str, str]:
    config.ensure_directories()
    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except Exception as exc:
        status_path = config.charts_dir / "visualization_status.json"
        status_path.write_text(
            json.dumps({"status": "skipped", "reason": f"Plotly unavailable: {exc}"}, indent=2),
            encoding="utf-8",
        )
        logger.warning("Skipping chart generation: %s", exc)
        return {"visualization_status": str(status_path)}

    outputs: dict[str, str] = {}

    elo_path = config.processed_dir / "elo_history.csv"
    if elo_path.exists():
        elo = pd.read_csv(elo_path)
        if not elo.empty:
            fig = px.line(elo, x="date", y="elo", color="team", title="Elo Evolution")
            outputs["elo_evolution"] = _write_html(fig, config.charts_dir / "elo_evolution.html")

    evaluation_path = config.metrics_dir / "evaluation.json"
    if evaluation_path.exists():
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        best_model = evaluation.get("best_model")
        importances = evaluation.get("feature_importance", {}).get(best_model, [])[:25]
        if importances:
            fig = px.bar(importances, x="importance", y="feature", orientation="h", title="Feature Importance")
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            outputs["feature_importance"] = _write_html(fig, config.charts_dir / "feature_importance.html")

        curves = evaluation.get("models", {}).get(best_model, {}).get("calibration_curve", [])
        if curves:
            fig = go.Figure()
            for class_curve in curves:
                bins = [row for row in class_curve["bins"] if row["mean_predicted"] is not None]
                fig.add_trace(
                    go.Scatter(
                        x=[row["mean_predicted"] for row in bins],
                        y=[row["observed_rate"] for row in bins],
                        mode="lines+markers",
                        name=class_curve["class"],
                    )
                )
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect"))
            fig.update_layout(title="Probability Calibration", xaxis_title="Predicted", yaxis_title="Observed")
            outputs["calibration"] = _write_html(fig, config.charts_dir / "calibration_curves.html")

        examples = evaluation.get("prediction_examples", [])
        if examples:
            probability_rows = [
                {"match": f"{row['home_team']} vs {row['away_team']}", "class": label, "probability": probability}
                for row in examples
                for label, probability in row["probabilities"].items()
            ]
            fig = px.box(probability_rows, x="class", y="probability", title="Prediction Probability Distribution")
            outputs["probability_distribution"] = _write_html(
                fig, config.charts_dir / "probability_distribution.html"
            )

        best_report = evaluation.get("models", {}).get(best_model, {})
        confusion = best_report.get("confusion_matrix")
        if confusion:
            fig = px.imshow(
                confusion,
                text_auto=True,
                x=["Pred H", "Pred D", "Pred A"],
                y=["Actual H", "Actual D", "Actual A"],
                title="Confusion Matrix",
            )
            outputs["confusion_matrix"] = _write_html(fig, config.charts_dir / "confusion_matrix.html")

    simulation_path = config.predictions_dir / "worldcup_simulation.json"
    if simulation_path.exists():
        simulation = json.loads(simulation_path.read_text(encoding="utf-8")).get("data", {})
        probabilities = simulation.get("probabilities", {})
        if probabilities:
            rows = sorted(
                [
                    {"team": team, "champion_probability": values["champion_probability"]}
                    for team, values in probabilities.items()
                ],
                key=lambda row: row["champion_probability"],
                reverse=True,
            )[:20]
            fig = px.bar(rows, x="champion_probability", y="team", orientation="h", title="World Cup Win Probability")
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            outputs["worldcup_simulation"] = _write_html(fig, config.charts_dir / "worldcup_simulation.html")

    strengths_path = config.processed_dir / "team_strengths.csv"
    if strengths_path.exists():
        strengths = pd.read_csv(strengths_path).sort_values("elo", ascending=False).head(1)
        if not strengths.empty:
            row = strengths.iloc[0]
            fig = go.Figure(
                data=go.Scatterpolar(
                    r=[row["elo"], row["offensive_elo"], row["defensive_elo"]],
                    theta=["Elo", "Attack", "Defense"],
                    fill="toself",
                    name=row["team"],
                )
            )
            fig.update_layout(title=f"Team Radar: {row['team']}")
            outputs["team_radar"] = _write_html(fig, config.charts_dir / "team_radar.html")

    return outputs


def _write_html(fig, path) -> str:
    fig.write_html(path)
    return str(path)

