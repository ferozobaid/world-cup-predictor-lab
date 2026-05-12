"use client";

import Image from "next/image";
import { useMemo, useState } from "react";
import { getTeamFlag, isFallbackBadge } from "@/lib/team-flags";
import { getMLModelDetails, predictMatchWithML } from "@/lib/ml-prediction-model";
import {
  applySquadStrengthAdjustment,
  getSquadStrength,
  type SquadAdjustmentState
} from "@/lib/squad-strength-adjustment";
import {
  formatRecord,
  predictMatch,
  worldCupData,
  type MatchStage,
  type Prediction,
  type TeamStats,
  type WorldCupFixture
} from "@/lib/world-cup-model";

type PredictorLabProps = {
  teams: string[];
  fixtures: WorldCupFixture[];
  generatedAt: string;
  sources: { label: string; url: string }[];
};

type AnalystSections = {
  keyTakeaway: string;
  whyFavorite: string;
  riskFactor: string;
  upsetPath: string;
  modelLimitation: string;
};

type ExplainState =
  | { status: "idle"; text: ""; meta: "" }
  | { status: "loading"; text: ""; meta: "" }
  | { status: "ready"; text: string; meta: string; sections?: AnalystSections }
  | { status: "error"; text: string; meta: string; sections?: AnalystSections };

const stages: MatchStage[] = ["Group stage", "Round of 16", "Quarter-final", "Semi-final", "Final"];

type WhatIfState = {
  recentForm: number;
  attack: number;
  defense: number;
  knockoutPressure: boolean;
};

type ScenarioTab = "Group Stage" | "Round of 32" | "Round of 16" | "Quarter-final" | "Semi-final" | "Final";
type PredictionModelMode = "historical" | "ml";

const defaultWhatIf: WhatIfState = {
  recentForm: 0,
  attack: 0,
  defense: 0,
  knockoutPressure: false
};

const scenarioTabs: ScenarioTab[] = ["Group Stage", "Round of 32", "Round of 16", "Quarter-final", "Semi-final", "Final"];

export function PredictorLab({ teams, fixtures, generatedAt, sources }: PredictorLabProps) {
  const [teamA, setTeamA] = useState("Argentina");
  const [teamB, setTeamB] = useState("France");
  const [stage, setStage] = useState<MatchStage>("Final");
  const [predictionModel, setPredictionModel] = useState<PredictionModelMode>("historical");
  const [modernSquadEnabled, setModernSquadEnabled] = useState(false);
  const [activeView, setActiveView] = useState<"prediction" | "data" | "fixtures">("prediction");
  const [explanation, setExplanation] = useState<ExplainState>({ status: "idle", text: "", meta: "" });
  const [whatIf, setWhatIf] = useState<WhatIfState>(defaultWhatIf);

  const prediction = useMemo(
    () => (predictionModel === "ml" ? predictMatchWithML(teamA, teamB, stage) : predictMatch(teamA, teamB, stage)),
    [predictionModel, teamA, teamB, stage]
  );
  const squadAdjustedResult = useMemo(
    () => applySquadStrengthAdjustment(prediction, modernSquadEnabled),
    [prediction, modernSquadEnabled]
  );
  const squadPrediction = squadAdjustedResult.prediction;
  const displayPrediction = useMemo(() => alignScoreline(applyWhatIf(squadPrediction, whatIf)), [squadPrediction, whatIf]);
  const mlDetails = useMemo(() => getMLModelDetails(), []);
  const sortedTeams = useMemo(() => teams.filter(Boolean), [teams]);

  async function explainPrediction() {
    setExplanation({ status: "loading", text: "", meta: "" });
    try {
      const whatIfActive =
        whatIf.recentForm > 0 || whatIf.attack > 0 || whatIf.defense > 0 || whatIf.knockoutPressure;
      const response = await fetch("/api/explain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          selectedModel: predictionModel === "ml" ? "Tournament ML" : "Historical Local",
          prediction: displayPrediction,
          modelDetails: predictionModel === "ml" ? mlDetails : null,
          squadProxy: {
            enabled: modernSquadEnabled,
            status: squadAdjustedResult.adjustment.status,
            message: squadAdjustedResult.adjustment.message,
            teamA:
              squadAdjustedResult.adjustment.status === "applied"
                ? {
                    team: squadAdjustedResult.adjustment.teamA.canonicalTeam,
                    score: squadAdjustedResult.adjustment.teamA.squadStrengthScore
                  }
                : null,
            teamB:
              squadAdjustedResult.adjustment.status === "applied"
                ? {
                    team: squadAdjustedResult.adjustment.teamB.canonicalTeam,
                    score: squadAdjustedResult.adjustment.teamB.squadStrengthScore
                  }
                : null
          },
          whatIf: {
            active: whatIfActive,
            recentForm: whatIf.recentForm,
            attack: whatIf.attack,
            defense: whatIf.defense,
            knockoutPressure: whatIf.knockoutPressure
          }
        })
      });
      const payload = await response.json();
      setExplanation({
        status: "ready",
        text: payload.text,
        sections: payload.sections,
        meta: payload.missingKey
          ? "OpenAI key missing"
          : `${payload.cached ? "Cached" : "Generated"} with ${payload.model}`
      });
    } catch {
      setExplanation({
        status: "error",
        text: "The explanation endpoint did not respond. The local prediction remains available.",
        meta: "API error"
      });
    }
  }

  function loadFixture(fixture: WorldCupFixture) {
    setTeamA(fixture.team1);
    setTeamB(fixture.team2);
    setStage("Group stage");
    setActiveView("prediction");
    setExplanation({ status: "idle", text: "", meta: "" });
  }

  function clearExplanation() {
    setExplanation({ status: "idle", text: "", meta: "" });
  }

  function applyScenario(scenario: string) {
    const scenarioTeams = sortedTeams.length ? sortedTeams : teams;
    if (scenario === "random") {
      const first = scenarioTeams[Math.floor(Math.random() * scenarioTeams.length)] || "Argentina";
      const pool = scenarioTeams.filter((team) => team !== first);
      const second = pool[Math.floor(Math.random() * pool.length)] || "France";
      setTeamA(first);
      setTeamB(second);
      setStage("Group stage");
    }
    if (scenario === "final") {
      setTeamA("Argentina");
      setTeamB("France");
      setStage("Final");
    }
    if (scenario === "upset") {
      setTeamA("South Africa");
      setTeamB("Brazil");
      setStage("Round of 16");
    }
    if (scenario === "heavyweight") {
      setTeamA("Brazil");
      setTeamB("Germany");
      setStage("Semi-final");
    }
    if (scenario === "group") {
      setTeamA("Mexico");
      setTeamB("South Africa");
      setStage("Group stage");
    }
    setActiveView("prediction");
    clearExplanation();
  }

  return (
    <div className="app-shell">
      <section className="workspace">
        <aside className="control-panel" aria-label="Prediction controls">
          <BrandHeader />
          <Controls
            sortedTeams={sortedTeams}
            teamA={teamA}
            teamB={teamB}
            stage={stage}
            setTeamA={(value) => {
              setTeamA(value);
              clearExplanation();
            }}
            setTeamB={(value) => {
              setTeamB(value);
              clearExplanation();
            }}
            setStage={(value) => {
              setStage(value);
              clearExplanation();
            }}
            swapTeams={() => {
              setTeamA(teamB);
              setTeamB(teamA);
              clearExplanation();
            }}
          />
          <ModelSelector
            value={predictionModel}
            modernSquadEnabled={modernSquadEnabled}
            onChange={(value) => {
              setPredictionModel(value);
              clearExplanation();
            }}
            onModernSquadChange={(value) => {
              setModernSquadEnabled(value);
              clearExplanation();
            }}
          />
          <QuickScenarios onApply={applyScenario} />
          <WhatIfLab
            whatIf={whatIf}
            setWhatIf={(value) => {
              setWhatIf(value);
              clearExplanation();
            }}
          />
          <div className="cost-box">
            <span>Budget guard</span>
            <strong>$0 prediction engine</strong>
            <p>On-demand analyst notes stay capped and cached.</p>
          </div>
        </aside>

        <section className="main-panel">
          <div className="topbar">
            <div>
              <p className="eyebrow">2026 Scenario</p>
              <h2>
                <TeamBadge team={teamA} /> {teamA} vs <TeamBadge team={teamB} /> {teamB}
              </h2>
            </div>
            <div className="tabs" role="tablist" aria-label="Dashboard views">
              <button
                className={activeView === "prediction" ? "active" : ""}
                onClick={() => setActiveView("prediction")}
                type="button"
              >
                Prediction
              </button>
              <button
                className={activeView === "data" ? "active" : ""}
                onClick={() => setActiveView("data")}
                type="button"
              >
                Team Data
              </button>
              <button
                className={activeView === "fixtures" ? "active" : ""}
                onClick={() => setActiveView("fixtures")}
                type="button"
              >
                2026 Scenario
              </button>
            </div>
          </div>

          {activeView === "prediction" && (
            <PredictionView
              prediction={prediction}
              displayPrediction={displayPrediction}
              predictionModel={predictionModel}
              mlDetails={mlDetails}
              squadAdjustment={squadAdjustedResult.adjustment}
              explanation={explanation}
              onExplain={explainPrediction}
            />
          )}

          {activeView === "data" && <DataView prediction={prediction} generatedAt={generatedAt} sources={sources} />}

          {activeView === "fixtures" && (
            <ScenarioView
              fixtures={fixtures}
              prediction={displayPrediction}
              onLoadFixture={loadFixture}
            />
          )}
        </section>
      </section>
    </div>
  );
}

function BrandHeader() {
  return (
    <div className="brand-block">
      <div className="brand-row">
        <div>
          <p className="eyebrow">World Cup Predictor Lab</p>
          <h1>Matchup model</h1>
        </div>
        <Image
          className="fifa-logo"
          src="/fifa-2026-logo.svg"
          alt="FIFA World Cup 2026 emblem"
          width={455}
          height={701}
          loading="eager"
          priority
          unoptimized
        />
      </div>
      <p className="intro">Build a matchup, test the stage, then stress the forecast with quick what-if scenarios.</p>
    </div>
  );
}

function Controls({
  sortedTeams,
  teamA,
  teamB,
  stage,
  setTeamA,
  setTeamB,
  setStage,
  swapTeams
}: {
  sortedTeams: string[];
  teamA: string;
  teamB: string;
  stage: MatchStage;
  setTeamA: (team: string) => void;
  setTeamB: (team: string) => void;
  setStage: (stage: MatchStage) => void;
  swapTeams: () => void;
}) {
  return (
    <div className="control-stack">
      <TeamSelect label="Team A" value={teamA} teams={sortedTeams} onChange={setTeamA} />
      <button className="swap-button" type="button" onClick={swapTeams} aria-label="Swap teams">
        ⇄
        <span>Swap</span>
      </button>
      <TeamSelect label="Team B" value={teamB} teams={sortedTeams} onChange={setTeamB} />
      <label className="field">
        <span>Match context</span>
        <select value={stage} onChange={(event) => setStage(event.target.value as MatchStage)}>
          {stages.map((stageName) => (
            <option key={stageName} value={stageName}>
              {stageName}
            </option>
          ))}
        </select>
      </label>
    </div>
  );
}

function TeamSelect({
  label,
  value,
  teams,
  onChange
}: {
  label: string;
  value: string;
  teams: string[];
  onChange: (team: string) => void;
}) {
  return (
    <label className="field team-field">
      <span>{label}</span>
      <div className="select-shell">
        <TeamBadge team={value} />
        <select value={value} onChange={(event) => onChange(event.target.value)}>
          {teams.map((team) => (
            <option key={team} value={team}>
              {team}
            </option>
          ))}
        </select>
      </div>
    </label>
  );
}

function ModelSelector({
  value,
  modernSquadEnabled,
  onChange,
  onModernSquadChange
}: {
  value: PredictionModelMode;
  modernSquadEnabled: boolean;
  onChange: (value: PredictionModelMode) => void;
  onModernSquadChange: (value: boolean) => void;
}) {
  return (
    <section className="mini-module model-selector">
      <div className="mini-heading">
        <span>Model</span>
      </div>
      <div className="model-toggle" role="group" aria-label="Prediction model">
        <button className={value === "historical" ? "active" : ""} type="button" onClick={() => onChange("historical")}>
          Historical Local
        </button>
        <button className={value === "ml" ? "active" : ""} type="button" onClick={() => onChange("ml")}>
          Tournament ML
        </button>
      </div>
      <div className="switch-control squad-switch">
        <button
          className={modernSquadEnabled ? "switch-button active" : "switch-button"}
          type="button"
          role="switch"
          aria-checked={modernSquadEnabled}
          onClick={() => onModernSquadChange(!modernSquadEnabled)}
        >
          <span>Modern squad layer: {modernSquadEnabled ? "On" : "Off"}</span>
          <i aria-hidden="true" />
        </button>
        <small>Optional curated proxy adjustment. Not betting odds.</small>
      </div>
    </section>
  );
}

function QuickScenarios({ onApply }: { onApply: (scenario: string) => void }) {
  const scenarios = [
    ["random", "Random matchup"],
    ["final", "2026 Final"],
    ["upset", "Underdog upset"],
    ["heavyweight", "Classic heavyweight clash"],
    ["group", "Group stage test"]
  ];

  return (
    <section className="mini-module">
      <div className="mini-heading">
        <span>Quick scenarios</span>
      </div>
      <div className="scenario-strip">
        {scenarios.map(([id, label]) => (
          <button key={id} type="button" onClick={() => onApply(id)}>
            {label}
          </button>
        ))}
      </div>
    </section>
  );
}

function WhatIfLab({
  whatIf,
  setWhatIf
}: {
  whatIf: WhatIfState;
  setWhatIf: (value: WhatIfState) => void;
}) {
  return (
    <section className="mini-module what-if-module">
      <div className="mini-heading">
        <span>What-if lab</span>
        <small>Team A boost</small>
      </div>
      <SliderControl
        label="Recent form"
        value={whatIf.recentForm}
        onChange={(recentForm) => setWhatIf({ ...whatIf, recentForm })}
      />
      <SliderControl
        label="Attack"
        value={whatIf.attack}
        onChange={(attack) => setWhatIf({ ...whatIf, attack })}
      />
      <SliderControl
        label="Defensive stability"
        value={whatIf.defense}
        onChange={(defense) => setWhatIf({ ...whatIf, defense })}
      />
      <div className="switch-control">
        <button
          className={whatIf.knockoutPressure ? "switch-button active" : "switch-button"}
          type="button"
          role="switch"
          aria-checked={whatIf.knockoutPressure}
          onClick={() => setWhatIf({ ...whatIf, knockoutPressure: !whatIf.knockoutPressure })}
        >
          <span>Knockout pressure: {whatIf.knockoutPressure ? "On" : "Off"}</span>
          <i aria-hidden="true" />
        </button>
      </div>
    </section>
  );
}

function SliderControl({
  label,
  value,
  onChange
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="slider-field">
      <span>
        {label}
        <strong>+{value}</strong>
      </span>
      <input type="range" min="0" max="10" value={value} onChange={(event) => onChange(Number(event.target.value))} />
    </label>
  );
}

function PredictionView({
  prediction,
  displayPrediction,
  predictionModel,
  mlDetails,
  squadAdjustment,
  explanation,
  onExplain
}: {
  prediction: Prediction;
  displayPrediction: Prediction;
  predictionModel: PredictionModelMode;
  mlDetails: ReturnType<typeof getMLModelDetails>;
  squadAdjustment: SquadAdjustmentState;
  explanation: ExplainState;
  onExplain: () => void;
}) {
  const [activeProbability, setActiveProbability] = useState("");
  const probabilities = [
    {
      label: `${displayPrediction.teamA} win`,
      team: displayPrediction.teamA,
      value: displayPrediction.probabilities.teamAWin,
      className: "home",
      detail: `${displayPrediction.teamA}'s path depends on form, goal balance, and stage pressure.`
    },
    {
      label: "Draw",
      team: "",
      value: displayPrediction.probabilities.draw,
      className: "draw",
      detail: displayPrediction.stage === "Group stage" ? "Draws carry more weight in group play." : "Knockout draws represent regulation-time deadlocks before extra time or penalties."
    },
    {
      label: `${displayPrediction.teamB} win`,
      team: displayPrediction.teamB,
      value: displayPrediction.probabilities.teamBWin,
      className: "away",
      detail: `${displayPrediction.teamB}'s chance rises when the matchup suppresses ${displayPrediction.teamA}'s attacking edge.`
    }
  ];
  const activeProbabilityDetail = probabilities.find((probability) => probability.label === activeProbability)?.detail;

  return (
    <div className="grid prediction-grid">
      <section className="panel match-card">
        <Image
          className="match-watermark"
          src="/fifa-2026-logo.svg"
          alt=""
          aria-hidden="true"
          width={455}
          height={701}
          loading="eager"
          unoptimized
        />
        <div className="match-card-top">
          <span>{displayPrediction.stage}</span>
          <strong>{displayPrediction.favorite === "Toss-up" ? "Toss-up" : `${displayPrediction.favorite} edge`}</strong>
        </div>
        <div className="scoreboard">
          <TeamIdentity team={displayPrediction.teamA} align="left" />
          <div className="score-core" aria-label={`${displayPrediction.teamA} ${displayPrediction.likelyScore.teamA} to ${displayPrediction.likelyScore.teamB} ${displayPrediction.teamB}`}>
            <strong>{displayPrediction.likelyScore.teamA}</strong>
            <span>-</span>
            <strong>{displayPrediction.likelyScore.teamB}</strong>
          </div>
          <TeamIdentity team={displayPrediction.teamB} align="right" />
        </div>
        <p className="match-story">{matchStory(displayPrediction, prediction)}</p>
        <div className="metric-strip">
          <Metric label="Favorite" value={displayPrediction.favorite} />
          <Metric label="Confidence" value={displayPrediction.confidence} />
          <Metric label="Model" value={predictionModel === "ml" ? "Tournament ML" : "Historical local"} />
          <Metric label="Match Type" value={displayPrediction.stage} />
          {squadAdjustment.status !== "disabled" && (
            <Metric
              label="Squad layer"
              value={squadAdjustment.status === "applied" ? squadAdjustment.message : "Unavailable"}
            />
          )}
        </div>
      </section>

      <section className="panel probability-panel">
        <div className="section-heading">
          <h3>Win probability</h3>
          <span>Normalized forecast</span>
        </div>
        <div className="bars">
          {probabilities.map((probability) => (
            <button
              key={probability.label}
              className={`bar-row ${activeProbability === probability.label ? "active" : ""}`}
              type="button"
              title={probability.detail}
              onClick={() => setActiveProbability(activeProbability === probability.label ? "" : probability.label)}
            >
              <div className="bar-label">
                <span>
                  {probability.team && <TeamBadge team={probability.team} />} {probability.label}
                </span>
                <strong>{probability.value}%</strong>
              </div>
              <div className="bar-track">
                <div className={`bar-fill ${probability.className}`} style={{ width: `${probability.value}%` }} />
              </div>
            </button>
          ))}
        </div>
        <p className="probability-note">
          {activeProbabilityDetail ||
            (squadAdjustment.status === "applied"
              ? "Why this matters: the selected base model is adjusted by a small curated squad-strength proxy. These are not betting odds."
              : predictionModel === "ml"
                ? "Why this matters: these are model-estimated probabilities from exported ML features, not betting odds."
                : "Why this matters: the bars translate the historical model into a quick match-read before the analyst brief.")}
        </p>
      </section>

      <section className="panel factors-panel">
        <div className="section-heading">
          <h3>Key factors</h3>
          <span>{prediction.teamA} lens</span>
        </div>
        <div className="factor-list">
          {prediction.factors.map((factor) => (
            <div className={`factor ${factor.impact}`} key={factor.label}>
              <span>{factor.label}</span>
              <p>{factor.value}</p>
            </div>
          ))}
        </div>
      </section>

      <section className={`panel ai-panel ${explanation.status}`}>
        <div className="section-heading">
          <h3>AI analyst brief</h3>
          <span>{explanation.meta || "Standby"}</span>
        </div>
        <AnalystBrief explanation={explanation} prediction={displayPrediction} />
        <button className="primary-button" type="button" onClick={onExplain} disabled={explanation.status === "loading"}>
          {explanation.status === "loading" ? "Generating..." : "Generate analyst brief"}
        </button>
      </section>

      {predictionModel === "ml" && <ModelDetailsPanel details={mlDetails} squadAdjustment={squadAdjustment} />}
    </div>
  );
}

function ModelDetailsPanel({
  details,
  squadAdjustment
}: {
  details: ReturnType<typeof getMLModelDetails>;
  squadAdjustment: SquadAdjustmentState;
}) {
  return (
    <section className="panel model-details-panel">
      <div className="section-heading">
        <h3>Model details</h3>
        <span>{details.isExperimental ? "Experimental" : "Validated"}</span>
      </div>
      <div className="model-metrics">
        <Metric label="Model type" value={details.modelType} />
        <Metric label="Accuracy" value={`${Math.round(details.accuracy * 100)}%`} />
        <Metric label="Macro F1" value={details.macroF1.toFixed(3)} />
        <Metric label="Historical baseline" value={`${Math.round(details.baselineAccuracy * 100)}% / ${details.baselineMacroF1.toFixed(3)} F1`} />
      </div>
      <div className="top-features">
        <span>Top features</span>
        <div>
          {details.topFeatures.map((feature) => (
            <strong key={feature.feature}>{formatFeatureName(feature.feature)}</strong>
          ))}
        </div>
      </div>
      {squadAdjustment.status !== "disabled" && (
        <div className="squad-detail-row">
          <span>Modern squad layer</span>
          <strong>
            {squadAdjustment.status === "applied"
              ? `${squadAdjustment.teamA.canonicalTeam} ${squadAdjustment.teamA.squadStrengthScore?.toFixed(1)} vs ${squadAdjustment.teamB.canonicalTeam} ${squadAdjustment.teamB.squadStrengthScore?.toFixed(1)}`
              : squadAdjustment.message}
          </strong>
        </div>
      )}
      <p>{details.probabilityNote}</p>
      <p>{details.limitation}</p>
    </section>
  );
}

function AnalystBrief({ explanation, prediction }: { explanation: ExplainState; prediction: Prediction }) {
  if (explanation.status === "idle") {
    return (
      <p className="ai-copy">
        Generate an OpenAI-powered analyst brief explaining the selected model output. OpenAI explains the forecast but does not make the prediction.
      </p>
    );
  }

  if (explanation.status === "loading") {
    return <p className="ai-copy">Generating brief...</p>;
  }

  const favorite = prediction.favorite === "Toss-up" ? prediction.teamA : prediction.favorite;
  const riskTeam = favorite === prediction.teamA ? prediction.teamB : prediction.teamA;
  const structured = explanation.sections ?? {
    keyTakeaway: explanation.text,
    whyFavorite:
      favorite === "Toss-up"
        ? "The model sees a narrow spread, so the stronger match moments matter more than the headline rating."
        : `${favorite} owns the current edge through the selected model, score projection, and matchup factors.`,
    riskFactor: `${riskTeam} can keep this close if the game state lowers tempo or turns the forecast into a one-chance match.`,
    upsetPath: "The underdog route is early defensive control, set-piece pressure, and forcing the favorite away from its normal scoring rhythm.",
    modelLimitation: "OpenAI explains the selected forecast but does not make the prediction. Probabilities are model-estimated, not betting odds."
  };
  const sections = [
    ["Key takeaway", structured.keyTakeaway],
    ["Why the favorite is favored", structured.whyFavorite],
    ["Risk factor", structured.riskFactor],
    ["Upset path", structured.upsetPath],
    ["Model limitation", structured.modelLimitation]
  ];

  return (
    <div className="brief-grid">
      {sections.map(([label, value]) => (
        <div className="brief-item" key={label}>
          <span>{label}</span>
          <p>{value}</p>
        </div>
      ))}
    </div>
  );
}

function TeamIdentity({ team, align }: { team: string; align: "left" | "right" }) {
  return (
    <div className={`team-identity ${align}`}>
      <TeamBadge team={team} size="large" />
      <span>{team}</span>
    </div>
  );
}

function DataView({
  prediction,
  generatedAt,
  sources
}: {
  prediction: Prediction;
  generatedAt: string;
  sources: { label: string; url: string }[];
}) {
  const teams = [prediction.statsA, prediction.statsB];

  return (
    <div className="grid data-grid">
      {teams.map((stats) => (
        <section className="panel team-data-panel" key={stats.team}>
          <div className="section-heading">
            <h3>
              <TeamBadge team={stats.team} /> {stats.team}
            </h3>
            <span>{stats.matches} matches</span>
          </div>
          <div className="stat-table">
            <Metric label="Record" value={formatRecord(stats)} />
            <Metric label="World Cup win rate" value={`${Math.round((stats.wins / Math.max(stats.matches, 1)) * 100)}%`} />
            <Metric label="Goals for" value={String(stats.goalsFor)} />
            <Metric label="Goals against" value={String(stats.goalsAgainst)} />
            <Metric label="Recent form index" value={`${Math.round(stats.recentForm * 100)}`} />
            <Metric label="Tournaments played" value={String(teamInsight(stats).tournaments)} />
            <Metric label="Best stage" value={teamInsight(stats).bestStage} />
            <SquadStrengthMetric team={stats.team} />
          </div>
        </section>
      ))}

      <section className="panel wide-panel">
        <div className="section-heading">
          <h3>Head-to-head</h3>
          <span>{prediction.headToHead.length || "No"} World Cup meetings</span>
        </div>
        {prediction.headToHead.length ? (
          <div className="match-list">
            {prediction.headToHead.map((match) => (
              <div className="match-row" key={match.id}>
                <span>{match.year}</span>
                <strong>
                  <TeamBadge team={match.homeTeam} /> {match.homeTeam} {match.homeScore}-{match.awayScore}{" "}
                  <TeamBadge team={match.awayTeam} /> {match.awayTeam}
                </strong>
                <span>{match.stage}</span>
              </div>
            ))}
          </div>
        ) : (
          <p className="muted">No direct World Cup meetings in the historical dataset.</p>
        )}
      </section>

      <section className="panel wide-panel source-panel">
        <div className="section-heading">
          <h3>Dataset</h3>
          <span>Generated {new Date(generatedAt).toLocaleDateString()}</span>
        </div>
        <p>
          Local snapshot built from public World Cup match records and 2026 fixture data. The model does
          not call OpenAI or any third-party API to calculate predictions.
        </p>
        <div className="source-links">
          {sources.map((source) => (
            <a key={source.url} href={source.url} target="_blank" rel="noreferrer">
              {source.label}
            </a>
          ))}
        </div>
      </section>
    </div>
  );
}

function SquadStrengthMetric({ team }: { team: string }) {
  const squad = getSquadStrength(team);
  if (!squad || squad.squadStrengthScore === null) {
    return <Metric label="Modern squad proxy" value="Unavailable" />;
  }

  return (
    <Metric
      label="Modern squad proxy"
      value={`${squad.squadStrengthScore.toFixed(1)} · ${squad.dataQuality.replaceAll("_", " ")}`}
    />
  );
}

function ScenarioView({
  fixtures,
  prediction,
  onLoadFixture
}: {
  fixtures: WorldCupFixture[];
  prediction: Prediction;
  onLoadFixture: (fixture: WorldCupFixture) => void;
}) {
  const [scenarioTab, setScenarioTab] = useState<ScenarioTab>("Group Stage");
  const fixtureGroups = useMemo(() => groupFixtures(fixtures), [fixtures]);
  const totalFixtures = fixtureGroups.reduce((total, group) => total + group.fixtures.length, 0);

  return (
    <div className="grid scenario-grid">
      <section className="panel wide-panel scenario-panel">
        <div className="section-heading">
          <h3>Scenario simulator</h3>
          <span>
            <TeamBadge team={prediction.teamA} /> {prediction.teamA} vs <TeamBadge team={prediction.teamB} /> {prediction.teamB}
          </span>
        </div>
        <div className="stage-buttons" role="group" aria-label="Match stage simulator">
          {scenarioTabs.map((stageName) => (
            <button
              key={stageName}
              className={scenarioTab === stageName ? "active" : ""}
              type="button"
              onClick={() => setScenarioTab(stageName)}
            >
              {stageName}
            </button>
          ))}
        </div>
        <p className="scenario-copy">
          {scenarioTab === "Group Stage"
            ? "Group fixtures are separated by section so the 72-match schedule is easier to scan."
            : "Bracket slots are placeholders until group standings are simulated or finalized."}
        </p>
      </section>
      {scenarioTab === "Group Stage" ? (
        <section className="panel wide-panel fixture-panel">
          <div className="section-heading">
            <h3>2026 group fixtures</h3>
            <span>{totalFixtures} group fixtures</span>
          </div>
          <div className="group-fixture-stack">
            {fixtureGroups.map((group) => (
              <section className="group-section" key={group.name}>
                <div className="group-header">
                  <div>
                    <span>{group.name}</span>
                    <strong>
                      {group.teams.map((team) => (
                        <span className="group-team" key={team}>
                          <TeamBadge team={team} /> {team}
                        </span>
                      ))}
                    </strong>
                  </div>
                  <small>{group.fixtures.length} fixtures</small>
                </div>
                <div className="fixture-grid">
                  {group.fixtures.map((fixture) => (
                    <button className="fixture" key={`${fixture.date}-${fixture.team1}-${fixture.team2}`} onClick={() => onLoadFixture(fixture)}>
                      <span className="fixture-meta">
                        {fixture.round} · {fixture.date}
                      </span>
                      <strong>
                        <TeamBadge team={fixture.team1} /> {fixture.team1}
                        <em>vs</em>
                        <TeamBadge team={fixture.team2} /> {fixture.team2}
                      </strong>
                      <small>{fixture.ground}</small>
                    </button>
                  ))}
                </div>
              </section>
            ))}
          </div>
        </section>
      ) : (
        <BracketView scenarioTab={scenarioTab} />
      )}
    </div>
  );
}

function BracketView({ scenarioTab }: { scenarioTab: Exclude<ScenarioTab, "Group Stage"> }) {
  const matchups = bracketMatchups(scenarioTab);

  return (
    <section className={`panel wide-panel bracket-panel ${scenarioTab === "Final" ? "final-bracket" : ""}`}>
      <div className="section-heading">
        <h3>{scenarioTab} bracket</h3>
        <span>{matchups.length} {matchups.length === 1 ? "matchup" : "matchups"}</span>
      </div>
      <div className="bracket-grid">
        {matchups.map((matchup) => (
          <article className="bracket-card" key={matchup.id}>
            <span className="bracket-meta">{matchup.id}</span>
            <div className="bracket-side">
              <span>{matchup.home}</span>
            </div>
            <em>vs</em>
            <div className="bracket-side">
              <span>{matchup.away}</span>
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}

function groupFixtures(fixtures: WorldCupFixture[]) {
  const groups = new Map<string, WorldCupFixture[]>();
  for (const fixture of fixtures) {
    const group = groups.get(fixture.group) ?? [];
    group.push(fixture);
    groups.set(fixture.group, group);
  }

  return [...groups.entries()]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([name, groupFixtures]) => ({
      name,
      teams: [...new Set(groupFixtures.flatMap((fixture) => [fixture.team1, fixture.team2]))],
      fixtures: groupFixtures
    }));
}

function bracketMatchups(scenarioTab: Exclude<ScenarioTab, "Group Stage">) {
  if (scenarioTab === "Round of 32") {
    return [
      ["R32-1", "Winner Group A", "Runner-up Group B"],
      ["R32-2", "Winner Group C", "Runner-up Group D"],
      ["R32-3", "Winner Group E", "Runner-up Group F"],
      ["R32-4", "Winner Group G", "Runner-up Group H"],
      ["R32-5", "Winner Group I", "Runner-up Group J"],
      ["R32-6", "Winner Group K", "Runner-up Group L"],
      ["R32-7", "Winner Group B", "Best 3rd-place team"],
      ["R32-8", "Winner Group D", "Best 3rd-place team"],
      ["R32-9", "Winner Group F", "Best 3rd-place team"],
      ["R32-10", "Winner Group H", "Best 3rd-place team"],
      ["R32-11", "Winner Group J", "Best 3rd-place team"],
      ["R32-12", "Winner Group L", "Best 3rd-place team"],
      ["R32-13", "Runner-up Group A", "Runner-up Group C"],
      ["R32-14", "Runner-up Group E", "Runner-up Group G"],
      ["R32-15", "Runner-up Group I", "Runner-up Group K"],
      ["R32-16", "Best 3rd-place team", "Best 3rd-place team"]
    ].map(toBracketMatchup);
  }

  if (scenarioTab === "Round of 16") {
    return Array.from({ length: 8 }, (_, index) =>
      toBracketMatchup([`R16-${index + 1}`, `Winner R32-${index * 2 + 1}`, `Winner R32-${index * 2 + 2}`])
    );
  }

  if (scenarioTab === "Quarter-final") {
    return Array.from({ length: 4 }, (_, index) =>
      toBracketMatchup([`QF-${index + 1}`, `Winner R16-${index * 2 + 1}`, `Winner R16-${index * 2 + 2}`])
    );
  }

  if (scenarioTab === "Semi-final") {
    return Array.from({ length: 2 }, (_, index) =>
      toBracketMatchup([`SF-${index + 1}`, `Winner QF-${index * 2 + 1}`, `Winner QF-${index * 2 + 2}`])
    );
  }

  return [toBracketMatchup(["Final", "Winner SF-1", "Winner SF-2"])];
}

function toBracketMatchup(matchup: string[]) {
  const [id, home, away] = matchup;
  return { id, home, away };
}

function applyWhatIf(prediction: Prediction, whatIf: WhatIfState): Prediction {
  const boost = whatIf.recentForm * 0.7 + whatIf.attack * 0.8 + whatIf.defense * 0.5 + (whatIf.knockoutPressure && prediction.stage !== "Group stage" ? 4 : 0);
  if (boost === 0) return prediction;

  const teamAWin = clampPercent(prediction.probabilities.teamAWin + boost, 5, 88);
  const drawDrop = Math.min(prediction.probabilities.draw - 8, boost * 0.28);
  const draw = clampPercent(prediction.probabilities.draw - Math.max(0, drawDrop), prediction.stage === "Group stage" ? 10 : 6, 42);
  const teamBWin = clampPercent(100 - teamAWin - draw, 4, 88);
  const normalizedTotal = teamAWin + draw + teamBWin;
  const probabilities = {
    teamAWin: Math.round((teamAWin / normalizedTotal) * 100),
    draw: Math.round((draw / normalizedTotal) * 100),
    teamBWin: 0
  };
  probabilities.teamBWin = 100 - probabilities.teamAWin - probabilities.draw;
  const favorite =
    Math.abs(probabilities.teamAWin - probabilities.teamBWin) < 5
      ? "Toss-up"
      : probabilities.teamAWin > probabilities.teamBWin
        ? prediction.teamA
        : prediction.teamB;
  const confidence =
    Math.abs(probabilities.teamAWin - probabilities.teamBWin) > 22
      ? "High"
      : Math.abs(probabilities.teamAWin - probabilities.teamBWin) < 8
        ? "Low"
        : prediction.confidence;

  return {
    ...prediction,
    probabilities,
    favorite,
    confidence
  };
}

function alignScoreline(prediction: Prediction): Prediction {
  const { teamA, teamB, favorite, likelyScore, probabilities, stage } = prediction;
  const isKnockout = stage !== "Group stage";
  const highScore = Math.max(likelyScore.teamA, likelyScore.teamB, isKnockout ? 1 : 0);
  const lowScore = Math.max(0, Math.min(likelyScore.teamA, likelyScore.teamB));

  if (favorite === "Toss-up") {
    if (!isKnockout) {
      const drawScore = Math.max(0, Math.min(highScore, 2));
      return {
        ...prediction,
        likelyScore: {
          teamA: drawScore,
          teamB: drawScore
        }
      };
    }

    const edgeTeam =
      probabilities.teamAWin === probabilities.teamBWin
        ? likelyScore.teamA >= likelyScore.teamB
          ? teamA
          : teamB
        : probabilities.teamAWin > probabilities.teamBWin
          ? teamA
          : teamB;
    return setOneGoalScore(prediction, edgeTeam, highScore, lowScore);
  }

  if (favorite === teamA && likelyScore.teamA <= likelyScore.teamB) {
    return setOneGoalScore(prediction, teamA, highScore, lowScore);
  }

  if (favorite === teamB && likelyScore.teamB <= likelyScore.teamA) {
    return setOneGoalScore(prediction, teamB, highScore, lowScore);
  }

  return prediction;
}

function setOneGoalScore(prediction: Prediction, winningTeam: string, highScore: number, lowScore: number): Prediction {
  const winnerScore = Math.max(highScore, lowScore + 1, 1);
  const loserScore = Math.max(0, Math.min(lowScore, winnerScore - 1));

  return {
    ...prediction,
    likelyScore:
      winningTeam === prediction.teamA
        ? {
            teamA: winnerScore,
            teamB: loserScore
          }
        : {
            teamA: loserScore,
            teamB: winnerScore
          }
  };
}

function matchStory(displayPrediction: Prediction, basePrediction: Prediction) {
  const adjusted =
    displayPrediction.probabilities.teamAWin !== basePrediction.probabilities.teamAWin ||
    displayPrediction.probabilities.teamBWin !== basePrediction.probabilities.teamBWin;
  if (displayPrediction.favorite === "Toss-up") {
    return `${displayPrediction.teamA} and ${displayPrediction.teamB} project as a tight ${displayPrediction.stage.toLowerCase()} with little room between them.`;
  }
  return `${displayPrediction.favorite} carries the edge${adjusted ? " after active scenario adjustments" : ""}, but the model still leaves a live route for the other side.`;
}

function teamInsight(stats: TeamStats) {
  const matches = worldCupData.matches.filter(
    (match) => match.homeTeam === stats.canonicalTeam || match.awayTeam === stats.canonicalTeam
  );
  const tournaments = new Set(matches.map((match) => match.year)).size;
  if (stats.titles > 0) return { tournaments, bestStage: "Champion" };
  if (stats.finals > 0) return { tournaments, bestStage: "Final" };

  const stageRank: Record<string, number> = {
    final: 6,
    "semi-finals": 5,
    "quarter-finals": 4,
    "round of 16": 3,
    "second group stage": 2,
    "final round": 2,
    "group stage": 1
  };
  const best = matches.reduce(
    (winner, match) => (stageRank[match.stage] > stageRank[winner] ? match.stage : winner),
    "group stage"
  );
  return { tournaments, bestStage: toTitleCase(best) };
}

function toTitleCase(value: string) {
  return value
    .split(" ")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function formatFeatureName(value: string) {
  return toTitleCase(value.replaceAll("_", " "));
}

function clampPercent(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function TeamBadge({ team, size = "default" }: { team: string; size?: "default" | "large" }) {
  if (team === "England" || team === "Scotland") {
    return (
      <span className={`team-badge ${size} subdivision ${team.toLowerCase()}`} aria-label={`${team} flag`}>
        <span />
      </span>
    );
  }

  const badge = getTeamFlag(team);
  return (
    <span className={`team-badge ${size} ${isFallbackBadge(team) ? "fallback" : ""}`} aria-label={`${team} flag`}>
      {badge}
    </span>
  );
}
