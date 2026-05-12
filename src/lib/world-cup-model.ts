import data from "@/data/world-cup-data.json";

export type MatchStage = "Group stage" | "Round of 16" | "Quarter-final" | "Semi-final" | "Final";

export type TeamStats = {
  team: string;
  canonicalTeam: string;
  matches: number;
  wins: number;
  draws: number;
  losses: number;
  goalsFor: number;
  goalsAgainst: number;
  knockoutMatches: number;
  knockoutWins: number;
  recentForm: number;
  attack: number;
  defense: number;
  rating: number;
  titles: number;
  finals: number;
};

export type PredictionFactor = {
  label: string;
  value: string;
  impact: "positive" | "neutral" | "negative";
};

export type Prediction = {
  teamA: string;
  teamB: string;
  stage: MatchStage;
  probabilities: {
    teamAWin: number;
    draw: number;
    teamBWin: number;
  };
  likelyScore: {
    teamA: number;
    teamB: number;
  };
  confidence: "Low" | "Medium" | "High";
  favorite: string;
  factors: PredictionFactor[];
  statsA: TeamStats;
  statsB: TeamStats;
  headToHead: WorldCupMatch[];
};

export type WorldCupMatch = (typeof data.matches)[number];
export type WorldCupFixture = (typeof data.fixtures2026)[number];

const aliasMap: Record<string, string> = {
  "Bosnia & Herzegovina": "Bosnia and Herzegovina",
  "DR Congo": "Zaire",
  USA: "United States"
};

const finalWinners: Record<number, string> = {};
const finalistsByYear = new Map<number, Set<string>>();

for (const match of data.matches) {
  if (match.stage === "final") {
    const winner = match.homeWin ? match.homeTeam : match.awayTeam;
    finalWinners[match.year] = winner;
    finalistsByYear.set(match.year, new Set([match.homeTeam, match.awayTeam]));
  }
}

export const worldCupData = data;

export function canonicalTeam(team: string) {
  return aliasMap[team] ?? team;
}

export function displayTeam(team: string) {
  return team;
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function roundPercent(value: number) {
  return Math.round(value * 100);
}

function matchTeams(match: WorldCupMatch, team: string) {
  return match.homeTeam === team || match.awayTeam === team;
}

function scoreFor(match: WorldCupMatch, team: string) {
  return match.homeTeam === team ? match.homeScore : match.awayScore;
}

function scoreAgainst(match: WorldCupMatch, team: string) {
  return match.homeTeam === team ? match.awayScore : match.homeScore;
}

function resultFor(match: WorldCupMatch, team: string) {
  if (match.draw) return "draw";
  if ((match.homeTeam === team && match.homeWin) || (match.awayTeam === team && match.awayWin)) {
    return "win";
  }
  return "loss";
}

export function getTeamStats(teamInput: string): TeamStats {
  const team = canonicalTeam(teamInput);
  const matches = data.matches.filter((match) => matchTeams(match, team));
  const recentMatches = matches.slice(-10);
  const wins = matches.filter((match) => resultFor(match, team) === "win").length;
  const draws = matches.filter((match) => resultFor(match, team) === "draw").length;
  const losses = matches.length - wins - draws;
  const goalsFor = matches.reduce((total, match) => total + scoreFor(match, team), 0);
  const goalsAgainst = matches.reduce((total, match) => total + scoreAgainst(match, team), 0);
  const knockoutMatches = matches.filter((match) => match.knockoutStage).length;
  const knockoutWins = matches.filter((match) => match.knockoutStage && resultFor(match, team) === "win").length;
  const recentPoints = recentMatches.reduce((total, match) => {
    const result = resultFor(match, team);
    return total + (result === "win" ? 3 : result === "draw" ? 1 : 0);
  }, 0);
  const recentForm = recentMatches.length ? recentPoints / (recentMatches.length * 3) : 0.34;
  const titles = Object.values(finalWinners).filter((winner) => winner === team).length;
  const finals = [...finalistsByYear.values()].filter((finalists) => finalists.has(team)).length;

  const perMatch = Math.max(matches.length, 1);
  const winRate = wins / perMatch;
  const goalDiff = (goalsFor - goalsAgainst) / perMatch;
  const knockoutRate = knockoutMatches ? knockoutWins / knockoutMatches : 0.28;
  const heritageBonus = titles * 4 + finals * 1.6;
  const samplePenalty = matches.length < 8 ? -8 : matches.length < 20 ? -3 : 0;
  const rating = clamp(
    50 + winRate * 48 + goalDiff * 7 + recentForm * 18 + knockoutRate * 12 + heritageBonus + samplePenalty,
    18,
    98
  );

  return {
    team: teamInput,
    canonicalTeam: team,
    matches: matches.length,
    wins,
    draws,
    losses,
    goalsFor,
    goalsAgainst,
    knockoutMatches,
    knockoutWins,
    recentForm,
    attack: goalsFor / perMatch,
    defense: goalsAgainst / perMatch,
    rating,
    titles,
    finals
  };
}

export function getHeadToHead(teamAInput: string, teamBInput: string) {
  const teamA = canonicalTeam(teamAInput);
  const teamB = canonicalTeam(teamBInput);
  return data.matches
    .filter(
      (match) =>
        (match.homeTeam === teamA && match.awayTeam === teamB) ||
        (match.homeTeam === teamB && match.awayTeam === teamA)
    )
    .slice(-8)
    .reverse();
}

export function predictMatch(teamA: string, teamB: string, stage: MatchStage): Prediction {
  const statsA = getTeamStats(teamA);
  const statsB = getTeamStats(teamB);
  const isKnockout = stage !== "Group stage";
  const stageMultiplier = stage === "Final" ? 1.08 : isKnockout ? 1.04 : 1;
  const ratingDiff = (statsA.rating - statsB.rating) * stageMultiplier;
  const splitA = 1 / (1 + Math.exp(-ratingDiff / 16));
  const uncertainty = Math.abs(ratingDiff);
  const baseDraw = isKnockout ? 0.18 : 0.26;
  const draw = clamp(baseDraw - uncertainty / 300, isKnockout ? 0.09 : 0.13, baseDraw);
  const remaining = 1 - draw;
  const teamAWin = remaining * splitA;
  const teamBWin = remaining * (1 - splitA);

  const expectedA = clamp(1.18 + (statsA.attack - statsB.defense) * 0.42 + ratingDiff / 85, 0.2, 3.7);
  const expectedB = clamp(1.18 + (statsB.attack - statsA.defense) * 0.42 - ratingDiff / 85, 0.2, 3.7);
  let scoreA = Math.round(expectedA);
  let scoreB = Math.round(expectedB);

  if (isKnockout && scoreA === scoreB) {
    if (teamAWin > teamBWin) scoreA += 1;
    if (teamBWin > teamAWin) scoreB += 1;
    if (teamAWin === teamBWin) {
      if (statsA.rating >= statsB.rating) scoreA += 1;
      else scoreB += 1;
    }
  }

  const favorite =
    Math.abs(teamAWin - teamBWin) < 0.05 ? "Toss-up" : teamAWin > teamBWin ? teamA : teamB;
  const confidence =
    Math.min(statsA.matches, statsB.matches) < 8 || Math.abs(teamAWin - teamBWin) < 0.08
      ? "Low"
      : Math.abs(teamAWin - teamBWin) > 0.22
        ? "High"
        : "Medium";

  return {
    teamA,
    teamB,
    stage,
    probabilities: {
      teamAWin: roundPercent(teamAWin),
      draw: roundPercent(draw),
      teamBWin: roundPercent(teamBWin)
    },
    likelyScore: {
      teamA: scoreA,
      teamB: scoreB
    },
    confidence,
    favorite,
    factors: buildFactors(statsA, statsB, stage),
    statsA,
    statsB,
    headToHead: getHeadToHead(teamA, teamB)
  };
}

function buildFactors(statsA: TeamStats, statsB: TeamStats, stage: MatchStage): PredictionFactor[] {
  const winRateA = statsA.matches ? statsA.wins / statsA.matches : 0;
  const winRateB = statsB.matches ? statsB.wins / statsB.matches : 0;
  const goalDiffA = statsA.attack - statsA.defense;
  const goalDiffB = statsB.attack - statsB.defense;
  const ratingGap = statsA.rating - statsB.rating;
  const recentGap = statsA.recentForm - statsB.recentForm;
  const knockoutGap =
    (statsA.knockoutMatches ? statsA.knockoutWins / statsA.knockoutMatches : 0.28) -
    (statsB.knockoutMatches ? statsB.knockoutWins / statsB.knockoutMatches : 0.28);

  return [
    {
      label: "Historical strength",
      value: `${statsA.team} ${statsA.rating.toFixed(1)} rating vs ${statsB.team} ${statsB.rating.toFixed(1)}`,
      impact: ratingGap > 4 ? "positive" : ratingGap < -4 ? "negative" : "neutral"
    },
    {
      label: "Win profile",
      value: `${Math.round(winRateA * 100)}% World Cup win rate vs ${Math.round(winRateB * 100)}%`,
      impact: winRateA > winRateB + 0.06 ? "positive" : winRateB > winRateA + 0.06 ? "negative" : "neutral"
    },
    {
      label: "Goal balance",
      value: `${goalDiffA.toFixed(2)} goals per match net vs ${goalDiffB.toFixed(2)}`,
      impact: goalDiffA > goalDiffB + 0.18 ? "positive" : goalDiffB > goalDiffA + 0.18 ? "negative" : "neutral"
    },
    {
      label: "Recent tournament form",
      value: `${Math.round(statsA.recentForm * 100)} form index vs ${Math.round(statsB.recentForm * 100)}`,
      impact: recentGap > 0.08 ? "positive" : recentGap < -0.08 ? "negative" : "neutral"
    },
    {
      label: stage === "Group stage" ? "Group-stage variance" : "Knockout record",
      value:
        stage === "Group stage"
          ? "Draw probability remains higher in group play."
          : `${statsA.knockoutWins}/${statsA.knockoutMatches || 0} knockout wins vs ${statsB.knockoutWins}/${statsB.knockoutMatches || 0}`,
      impact: stage === "Group stage" ? "neutral" : knockoutGap > 0.08 ? "positive" : knockoutGap < -0.08 ? "negative" : "neutral"
    }
  ];
}

export function formatRecord(stats: TeamStats) {
  return `${stats.wins}W ${stats.draws}D ${stats.losses}L`;
}
