import featureImportance from "../../ml/model_outputs/feature_importance.json";
import modelMetrics from "../../ml/model_outputs/model_metrics.json";
import teamFeatures from "../../ml/model_outputs/team_features.json";
import { canonicalTeam, getHeadToHead, getTeamStats, predictMatch, type MatchStage, type Prediction, type PredictionFactor } from "./world-cup-model";

type TeamFeature = {
  win_rate: number;
  goals_for_per_match: number;
  goals_against_per_match: number;
  goal_difference_per_match: number;
  recent_form_index: number;
  tournament_experience: number;
  strength: number;
  matches_before: number;
};

type MLModelDetails = {
  modelType: string;
  accuracy: number;
  macroF1: number;
  baselineAccuracy: number;
  baselineMacroF1: number;
  topFeatures: { feature: string; importance: number }[];
  probabilityNote: string;
  limitation: string;
  isExperimental: boolean;
};

const typedTeamFeatures = teamFeatures as Record<string, TeamFeature>;
const frontendParameters = modelMetrics.frontend_parameters;
const logisticMetrics = modelMetrics.models.logistic_regression;
const historicalBaseline = modelMetrics.baselines.historical_strength;
const topLogisticFeatures = featureImportance.logistic_regression.slice(0, 5);

export function getMLModelDetails(): MLModelDetails {
  const isExperimental = logisticMetrics.macro_f1 <= historicalBaseline.macro_f1;
  return {
    modelType: modelMetrics.selected_frontend_model,
    accuracy: logisticMetrics.accuracy,
    macroF1: logisticMetrics.macro_f1,
    baselineAccuracy: historicalBaseline.accuracy,
    baselineMacroF1: historicalBaseline.macro_f1,
    topFeatures: topLogisticFeatures,
    probabilityNote: modelMetrics.probability_note,
    limitation: isExperimental
      ? "Tournament ML is educational/experimental here because it does not outperform the historical-strength baseline on the time-based test split."
      : "Tournament ML outperforms the simple historical-strength baseline on the time-based test split, but calibration can still be weak on a small dataset.",
    isExperimental
  };
}

export function predictMatchWithML(teamA: string, teamB: string, stage: MatchStage): Prediction {
  const historical = predictMatch(teamA, teamB, stage);
  const features = buildFeatureVector(teamA, teamB, stage);

  if (!features) {
    return {
      ...historical,
      factors: [
        {
          label: "ML fallback",
          value: "Team features were unavailable, so the historical local model is shown.",
          impact: "neutral"
        },
        ...historical.factors.slice(0, 4)
      ]
    };
  }

  const probabilities = predictProbabilities(features);
  const favorite =
    Math.abs(probabilities.teamAWin - probabilities.teamBWin) < 5
      ? "Toss-up"
      : probabilities.teamAWin > probabilities.teamBWin
        ? teamA
        : teamB;
  const spread = Math.abs(probabilities.teamAWin - probabilities.teamBWin);
  const confidence = spread > 22 ? "High" : spread < 8 ? "Low" : "Medium";

  return {
    teamA,
    teamB,
    stage,
    probabilities,
    likelyScore: historical.likelyScore,
    confidence,
    favorite,
    factors: buildMLFactors(teamA, teamB, features),
    statsA: getTeamStats(teamA),
    statsB: getTeamStats(teamB),
    headToHead: getHeadToHead(teamA, teamB)
  };
}

function buildFeatureVector(teamA: string, teamB: string, stage: MatchStage) {
  const featuresA = typedTeamFeatures[canonicalTeam(teamA)] ?? typedTeamFeatures[teamA];
  const featuresB = typedTeamFeatures[canonicalTeam(teamB)] ?? typedTeamFeatures[teamB];
  if (!featuresA || !featuresB) return null;

  return {
    team_a_win_rate: featuresA.win_rate,
    team_b_win_rate: featuresB.win_rate,
    team_a_goals_for_per_match: featuresA.goals_for_per_match,
    team_b_goals_for_per_match: featuresB.goals_for_per_match,
    team_a_goals_against_per_match: featuresA.goals_against_per_match,
    team_b_goals_against_per_match: featuresB.goals_against_per_match,
    team_a_goal_difference_per_match: featuresA.goal_difference_per_match,
    team_b_goal_difference_per_match: featuresB.goal_difference_per_match,
    team_a_recent_form_index: featuresA.recent_form_index,
    team_b_recent_form_index: featuresB.recent_form_index,
    team_a_tournament_experience: featuresA.tournament_experience,
    team_b_tournament_experience: featuresB.tournament_experience,
    knockout_match_flag: stage === "Group stage" ? 0 : 1,
    final_flag: stage === "Final" ? 1 : 0,
    strength_difference: featuresA.strength - featuresB.strength,
    form_difference: featuresA.recent_form_index - featuresB.recent_form_index,
    goal_balance_difference: featuresA.goal_difference_per_match - featuresB.goal_difference_per_match,
    experience_difference: featuresA.tournament_experience - featuresB.tournament_experience
  };
}

function predictProbabilities(features: Record<string, number>) {
  const featureOrder = frontendParameters.feature_order as string[];
  const scaled = featureOrder.map((feature, index) => {
    const scale = frontendParameters.scaler_scale[index] || 1;
    return ((features[feature] ?? 0) - frontendParameters.scaler_mean[index]) / scale;
  });
  const scores = frontendParameters.coefficients.map((row: number[], classIndex: number) =>
    row.reduce((total, coefficient, featureIndex) => total + coefficient * scaled[featureIndex], frontendParameters.intercepts[classIndex])
  );
  const probabilities = softmax(scores);
  const byClass = Object.fromEntries(
    (frontendParameters.class_labels as string[]).map((label, index) => [label, probabilities[index]])
  );
  const teamAWin = Math.round((byClass.teamA_win ?? 0) * 100);
  const draw = Math.round((byClass.draw ?? 0) * 100);
  return {
    teamAWin,
    draw,
    teamBWin: Math.max(0, 100 - teamAWin - draw)
  };
}

function softmax(values: number[]) {
  const max = Math.max(...values);
  const exp = values.map((value) => Math.exp(value - max));
  const total = exp.reduce((sum, value) => sum + value, 0);
  return exp.map((value) => value / total);
}

function buildMLFactors(teamA: string, teamB: string, features: Record<string, number>): PredictionFactor[] {
  return [
    {
      label: "Model-estimated probabilities",
      value: "Tournament ML probabilities are not betting odds and may be weakly calibrated on this small dataset.",
      impact: "neutral"
    },
    {
      label: "Strength difference",
      value: `${teamA} ${features.strength_difference.toFixed(2)} model-strength edge vs ${teamB}`,
      impact: features.strength_difference > 2 ? "positive" : features.strength_difference < -2 ? "negative" : "neutral"
    },
    {
      label: "Form difference",
      value: `${Math.round(features.form_difference * 100)} point recent-form gap`,
      impact: features.form_difference > 0.05 ? "positive" : features.form_difference < -0.05 ? "negative" : "neutral"
    },
    {
      label: "Goal balance difference",
      value: `${features.goal_balance_difference.toFixed(2)} goals per match net gap`,
      impact: features.goal_balance_difference > 0.15 ? "positive" : features.goal_balance_difference < -0.15 ? "negative" : "neutral"
    },
    {
      label: "Experience difference",
      value: `${features.experience_difference.toFixed(0)} tournament appearance gap`,
      impact: features.experience_difference > 1 ? "positive" : features.experience_difference < -1 ? "negative" : "neutral"
    }
  ];
}
