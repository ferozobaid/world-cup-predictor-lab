import squadStrengthData from "@/data/squad-strength.json";
import { canonicalTeam, type Prediction, type PredictionFactor } from "./world-cup-model";

type SquadStrengthTeam = {
  snapshotId: string;
  snapshotYear: number;
  asOfDate: string;
  competitionSource: string;
  sourceType: string;
  sourceUrl: string;
  rawTeamName: string;
  canonicalTeam: string;
  playerCount: number | null;
  fifaRank: number | null;
  fifaPoints: number | null;
  squadMarketValueEurM: number | null;
  top11MarketValueEurM: number | null;
  top5MarketValueEurM: number | null;
  avgAge: number | null;
  totalCaps: number | null;
  avgCaps: number | null;
  clubStrengthIndex: number | null;
  recentCompetitiveForm: number | null;
  dataQuality: string;
  notes: string;
  squadStrengthScore: number | null;
  scoreComponents: Record<string, number>;
};

type SquadStrengthPayload = {
  generatedAt: string;
  method: {
    description: string;
    weights: Record<string, number>;
    logScaledFields: string[];
  };
  teams: Record<string, SquadStrengthTeam>;
};

export type SquadAdjustmentState =
  | {
      status: "disabled";
      available: false;
      message: string;
      teamA?: SquadStrengthTeam;
      teamB?: SquadStrengthTeam;
    }
  | {
      status: "unavailable";
      available: false;
      message: string;
      teamA?: SquadStrengthTeam;
      teamB?: SquadStrengthTeam;
    }
  | {
      status: "applied";
      available: true;
      message: string;
      teamA: SquadStrengthTeam;
      teamB: SquadStrengthTeam;
      strengthDelta: number;
      shift: number;
      drawAdjustment: number;
    };

const typedSquadStrengthData = squadStrengthData as SquadStrengthPayload;

export function getSquadStrength(team: string) {
  return typedSquadStrengthData.teams[canonicalTeam(team)] ?? typedSquadStrengthData.teams[team];
}

export function getSquadStrengthMeta() {
  return {
    generatedAt: typedSquadStrengthData.generatedAt,
    description: typedSquadStrengthData.method.description
  };
}

export function applySquadStrengthAdjustment(prediction: Prediction, enabled: boolean) {
  const teamA = getSquadStrength(prediction.teamA);
  const teamB = getSquadStrength(prediction.teamB);

  if (!enabled) {
    return {
      prediction,
      adjustment: {
        status: "disabled",
        available: false,
        message: "Modern squad strength is off."
      } satisfies SquadAdjustmentState
    };
  }

  if (!teamA || !teamB || teamA.squadStrengthScore === null || teamB.squadStrengthScore === null) {
    return {
      prediction,
      adjustment: {
        status: "unavailable",
        available: false,
        message: "Squad data unavailable for one or both teams.",
        teamA,
        teamB
      } satisfies SquadAdjustmentState
    };
  }

  const strengthDelta = teamA.squadStrengthScore - teamB.squadStrengthScore;
  const shift = clamp(strengthDelta * 0.12, -8, 8);
  const drawAdjustment = -Math.min(Math.abs(shift) * 0.375, 3);
  const adjusted = normalizeProbabilities({
    teamAWin: prediction.probabilities.teamAWin + shift,
    draw: prediction.probabilities.draw + drawAdjustment,
    teamBWin: prediction.probabilities.teamBWin - shift
  });
  const favorite =
    Math.abs(adjusted.teamAWin - adjusted.teamBWin) < 5
      ? "Toss-up"
      : adjusted.teamAWin > adjusted.teamBWin
        ? prediction.teamA
        : prediction.teamB;
  const confidence =
    Math.abs(adjusted.teamAWin - adjusted.teamBWin) > 22
      ? "High"
      : Math.abs(adjusted.teamAWin - adjusted.teamBWin) < 8
        ? "Low"
        : prediction.confidence;

  const squadFactor: PredictionFactor = {
    label: "Modern squad layer",
    value: `${prediction.teamA} ${teamA.squadStrengthScore.toFixed(1)} squad score vs ${prediction.teamB} ${teamB.squadStrengthScore.toFixed(1)}; ${formatSigned(shift)} point display adjustment.`,
    impact: shift > 0.5 ? "positive" : shift < -0.5 ? "negative" : "neutral"
  };

  return {
    prediction: {
      ...prediction,
      probabilities: adjusted,
      favorite,
      confidence,
      factors: [squadFactor, ...prediction.factors]
    },
    adjustment: {
      status: "applied",
      available: true,
      message: formatSquadAdjustmentMessage(prediction.teamA, prediction.teamB, shift),
      teamA,
      teamB,
      strengthDelta,
      shift,
      drawAdjustment
    } satisfies SquadAdjustmentState
  };
}

function normalizeProbabilities(probabilities: Prediction["probabilities"]) {
  const raw = {
    teamAWin: clamp(probabilities.teamAWin, 1, 96),
    draw: clamp(probabilities.draw, 1, 60),
    teamBWin: clamp(probabilities.teamBWin, 1, 96)
  };
  const total = raw.teamAWin + raw.draw + raw.teamBWin;
  const teamAWin = Math.round((raw.teamAWin / total) * 100);
  const draw = Math.round((raw.draw / total) * 100);
  return {
    teamAWin,
    draw,
    teamBWin: 100 - teamAWin - draw
  };
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function formatSigned(value: number) {
  const rounded = Math.round(value * 10) / 10;
  return `${rounded >= 0 ? "+" : ""}${rounded}`;
}

function formatSquadAdjustmentMessage(teamA: string, teamB: string, shift: number) {
  if (Math.abs(shift) < 0.05) {
    return "Squad adjustment: 0 percentage points";
  }
  const boostedTeam = shift > 0 ? teamA : teamB;
  return `Squad adjustment: ${boostedTeam} ${formatSigned(Math.abs(shift))} percentage points`;
}
