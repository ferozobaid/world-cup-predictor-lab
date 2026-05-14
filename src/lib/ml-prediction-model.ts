import featureImportance from "../../ml/model_outputs/feature_importance.json";
import matchupPredictions from "../../ml/model_outputs/matchup_predictions.json";
import modelMetrics from "../../ml/model_outputs/model_metrics.json";
import { getHeadToHead, getTeamStats, predictMatch, type MatchStage, type Prediction, type PredictionFactor } from "./world-cup-model";

export type MLModelMode = "calibrated" | "benchmark" | "elo";

export type MLModelDetails = {
  modelType: string;
  accuracy: number;
  macroF1: number;
  logLoss?: number;
  brierScore?: number;
  baselineAccuracy: number;
  baselineMacroF1: number;
  topFeatures: { feature: string; importance: number }[];
  probabilityNote: string;
  limitation: string;
  isExperimental: boolean;
};

type StaticPredictionLike = {
  probabilities: Prediction["probabilities"];
  likelyScore: Prediction["likelyScore"];
  expectedGoals?: {
    teamA: number;
    teamB: number;
  };
  favorite: string;
  confidence: Prediction["confidence"];
  factors: PredictionFactor[];
};

type MatchupPredictionRecord = {
  teamA: string;
  teamB: string;
  stage: MatchStage;
  pipelineTeams?: {
    teamA: string;
    teamB: string;
  };
  models: Record<MLModelMode, StaticPredictionLike>;
};

type MatchupPredictionPayload = {
  metadata: {
    schemaVersion: string;
    generatedAt: string;
    defaultModel: MLModelMode;
    sourceModels: Record<MLModelMode, string>;
  };
  data: {
    teams: string[];
    predictions: Record<string, MatchupPredictionRecord>;
  };
};

type PipelineMetric = {
  accuracy?: number;
  macro_f1?: number;
  log_loss?: number;
  brier?: number;
  brier_score?: number;
};

type PipelineMetricsPayload = {
  data?: {
    best_model?: string;
    models?: Record<string, PipelineMetric>;
  };
};

type FeatureImportancePayload = {
  data?: Record<string, { feature: string; importance: number }[]>;
};

const typedMatchups = matchupPredictions as MatchupPredictionPayload;
const typedMetrics = modelMetrics as PipelineMetricsPayload;
const typedFeatureImportance = featureImportance as FeatureImportancePayload;

const sourceModelByMode: Record<MLModelMode, string> = {
  calibrated: "catboost",
  benchmark: "logistic_regression",
  elo: "elo_poisson"
};

const modelTypeByMode: Record<MLModelMode, string> = {
  calibrated: "Calibrated ML (CatBoost)",
  benchmark: "ML Benchmark (Logistic Regression)",
  elo: "Elo + Poisson Baseline"
};

const legacyHistoricalBaseline = {
  accuracy: 0.484,
  macroF1: 0.456
};

export function getMLModelDetails(mode: MLModelMode = "calibrated"): MLModelDetails {
  const sourceModel = sourceModelByMode[mode];
  const metrics = typedMetrics.data?.models?.[sourceModel] ?? {};
  const topFeatures = (typedFeatureImportance.data?.[sourceModel] ?? []).slice(0, 5).map((feature) => ({
    feature: feature.feature,
    importance: Number(feature.importance) || 0
  }));

  return {
    modelType: modelTypeByMode[mode],
    accuracy: metricValue(metrics.accuracy),
    macroF1: metricValue(metrics.macro_f1),
    logLoss: optionalMetric(metrics.log_loss),
    brierScore: optionalMetric(metrics.brier_score ?? metrics.brier),
    baselineAccuracy: legacyHistoricalBaseline.accuracy,
    baselineMacroF1: legacyHistoricalBaseline.macroF1,
    topFeatures,
    probabilityNote: probabilityNote(mode),
    limitation: limitationNote(mode),
    isExperimental: mode !== "calibrated"
  };
}

export function predictMatchWithCalibratedML(teamA: string, teamB: string, stage: MatchStage): Prediction {
  return predictFromStaticArtifact(teamA, teamB, stage, "calibrated");
}

export function predictMatchWithML(teamA: string, teamB: string, stage: MatchStage): Prediction {
  return predictFromStaticArtifact(teamA, teamB, stage, "benchmark");
}

export function predictMatchWithEloScore(teamA: string, teamB: string, stage: MatchStage): Prediction {
  return predictFromStaticArtifact(teamA, teamB, stage, "elo");
}

function predictFromStaticArtifact(teamA: string, teamB: string, stage: MatchStage, mode: MLModelMode): Prediction {
  const fallbackPrediction = predictMatch(teamA, teamB, stage);
  const matchup = typedMatchups.data.predictions[predictionKey(teamA, teamB, stage)];
  const staticPrediction = matchup?.models?.[mode];

  if (!staticPrediction) {
    return mlFallback(
      fallbackPrediction,
      `${modelLabel(mode)} coverage was unavailable for this matchup, so Legacy Historical is shown.`
    );
  }

  return {
    teamA,
    teamB,
    stage,
    probabilities: normalizeStaticProbabilities(staticPrediction.probabilities),
    likelyScore: normalizeStaticScore(staticPrediction.likelyScore),
    confidence: validConfidence(staticPrediction.confidence),
    favorite: staticPrediction.favorite || fallbackPrediction.favorite,
    factors: normalizeStaticFactors(staticPrediction.factors, mode),
    statsA: getTeamStats(teamA),
    statsB: getTeamStats(teamB),
    headToHead: getHeadToHead(teamA, teamB)
  };
}

function mlFallback(historical: Prediction, reason: string): Prediction {
  return {
    ...historical,
    factors: [
      {
        label: "ML coverage fallback",
        value: reason,
        impact: "neutral"
      },
      ...historical.factors.slice(0, 4)
    ]
  };
}

function normalizeStaticProbabilities(probabilities: Prediction["probabilities"]): Prediction["probabilities"] {
  const teamAWin = clampPercent(Math.round(Number(probabilities.teamAWin) || 0));
  const draw = clampPercent(Math.round(Number(probabilities.draw) || 0));
  const teamBWin = clampPercent(100 - teamAWin - draw);

  if (teamAWin + draw + teamBWin === 100) {
    return { teamAWin, draw, teamBWin };
  }

  return {
    teamAWin,
    draw,
    teamBWin: clampPercent(Math.round(Number(probabilities.teamBWin) || 0))
  };
}

function normalizeStaticScore(score: Prediction["likelyScore"]): Prediction["likelyScore"] {
  return {
    teamA: Math.max(0, Math.round(Number(score.teamA) || 0)),
    teamB: Math.max(0, Math.round(Number(score.teamB) || 0))
  };
}

function normalizeStaticFactors(factors: PredictionFactor[] | undefined, mode: MLModelMode): PredictionFactor[] {
  if (!factors?.length) {
    return [
      {
        label: modelLabel(mode),
        value: "Static model output was generated offline from the Python ML pipeline.",
        impact: "neutral"
      }
    ];
  }

  return factors.slice(0, 5).map((factor) => ({
    label: String(factor.label || "Model factor"),
    value: String(factor.value || "No factor detail available."),
    impact: validImpact(factor.impact)
  }));
}

function validConfidence(confidence: string): Prediction["confidence"] {
  if (confidence === "High" || confidence === "Medium" || confidence === "Low") return confidence;
  return "Medium";
}

function validImpact(impact: string): PredictionFactor["impact"] {
  if (impact === "positive" || impact === "negative" || impact === "neutral") return impact;
  return "neutral";
}

function predictionKey(teamA: string, teamB: string, stage: MatchStage) {
  return `${teamA}|${teamB}|${stage}`;
}

function metricValue(value: unknown) {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

function optionalMetric(value: unknown) {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function clampPercent(value: number) {
  return Math.min(100, Math.max(0, value));
}

function modelLabel(mode: MLModelMode) {
  if (mode === "calibrated") return "Calibrated ML";
  if (mode === "benchmark") return "ML Benchmark";
  return "Elo + Score";
}

function probabilityNote(mode: MLModelMode) {
  if (mode === "calibrated") {
    return "Calibrated ML probabilities are static model-estimated probabilities exported from the offline Python pipeline. They are not betting odds.";
  }
  if (mode === "benchmark") {
    return "ML Benchmark probabilities come from an offline Logistic Regression comparison model. They are not betting odds.";
  }
  return "Elo + Score probabilities come from an offline Elo + Poisson baseline. They are not betting odds.";
}

function limitationNote(mode: MLModelMode) {
  if (mode === "calibrated") {
    return "Calibrated ML uses broader 2018-2026 senior international results, so it is stronger than the old World Cup-only model but not directly apples-to-apples with the legacy backtest.";
  }
  if (mode === "benchmark") {
    return "ML Benchmark is kept for transparency as an explainable Logistic Regression comparison, not as the recommended default.";
  }
  return "Elo + Score is a football-native baseline for comparison; it is useful for score intuition but trails the calibrated classifier on the new test split.";
}
