"use client";

import Image from "next/image";
import { useMemo, useRef, useState } from "react";
import worldCupSimulation from "../../ml/model_outputs/worldcup_simulation.json";
import { getTeamFlag, isFallbackBadge } from "@/lib/team-flags";
import {
  getMLModelDetails,
  predictMatchWithCalibratedML,
  predictMatchWithEloScore,
  predictMatchWithML,
  type MLModelDetails
} from "@/lib/ml-prediction-model";
import {
  applySquadStrengthAdjustment,
  getSquadStrength,
  type SquadAdjustmentState
} from "@/lib/squad-strength-adjustment";
import {
  predictMatch,
  worldCupData,
  type MatchStage,
  type Prediction,
  type PredictionFactor,
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

const knockoutStages: MatchStage[] = ["Round of 32", "Round of 16", "Quarter-final", "Semi-final", "Final"];

type WhatIfTarget = "teamA" | "teamB";

type WhatIfValues = {
  recentForm: number;
  attack: number;
  defense: number;
  knockoutPressure: boolean;
};

type WhatIfState = {
  activeTarget: WhatIfTarget;
  teamA: WhatIfValues;
  teamB: WhatIfValues;
};

type ScenarioTab = "Group Stage" | "Round of 32" | "Round of 16" | "Quarter-final" | "Semi-final" | "Final";
type PredictionModelMode = "calibrated" | "legacy" | "benchmark" | "elo";

type SimulationProbability = {
  knockout_probability: number;
  quarterfinal_probability: number;
  semifinal_probability: number;
  finalist_probability: number;
  champion_probability: number;
};

type WorldCupSimulationPayload = {
  data?: {
    runs?: number;
    probabilities?: Record<string, SimulationProbability>;
  };
};

const defaultWhatIfValues: WhatIfValues = {
  recentForm: 0,
  attack: 0,
  defense: 0,
  knockoutPressure: false
};

const defaultWhatIf: WhatIfState = {
  activeTarget: "teamA",
  teamA: { ...defaultWhatIfValues },
  teamB: { ...defaultWhatIfValues }
};

const scenarioTabs: ScenarioTab[] = ["Group Stage", "Round of 32", "Round of 16", "Quarter-final", "Semi-final", "Final"];
const typedWorldCupSimulation = worldCupSimulation as WorldCupSimulationPayload;
const simulationTeamAliases: Record<string, string> = {
  "Bosnia & Herzegovina": "Bosnia and Herzegovina",
  "DR Congo": "Congo DR",
  USA: "United States"
};
const simulationToAppTeamAliases: Record<string, string> = Object.fromEntries(
  Object.entries(simulationTeamAliases).map(([appTeam, simulationTeam]) => [simulationTeam, appTeam])
);
type HostCityPoster = {
  id: string;
  city: string;
  fixtureGround: string;
  hostCountry: string;
  stadiumName: string;
  caption: string;
  label: string;
  imageSrc?: string;
  width?: number;
  height?: number;
};

const hostCityPosters: HostCityPoster[] = [
  {
    id: "atlanta",
    city: "Atlanta",
    fixtureGround: "Atlanta",
    hostCountry: "USA",
    stadiumName: "Mercedes-Benz Stadium",
    caption: "peach sky, skyline, supporter color",
    label: "Atlanta World Cup 2026 poster with peach sky, fans, skyline, and football energy",
    imageSrc: "/world-cup-posters/atlanta-2026-poster.jpeg",
    width: 768,
    height: 1154
  },
  {
    id: "boston",
    city: "Boston",
    fixtureGround: "Boston (Foxborough)",
    hostCountry: "USA",
    stadiumName: "Gillette Stadium",
    caption: "harbor character, hand-drawn match joy",
    label: "Boston World Cup 2026 poster with harbor artwork, football, and coastal character",
    imageSrc: "/world-cup-posters/boston-2026-poster.jpeg",
    width: 754,
    height: 1116
  },
  {
    id: "dallas",
    city: "Dallas",
    fixtureGround: "Dallas (Arlington)",
    hostCountry: "USA",
    stadiumName: "AT&T Stadium",
    caption: "deep teal, red skyline, overhead kick",
    label: "Dallas World Cup 2026 poster with red skyline and overhead kick artwork",
    imageSrc: "/world-cup-posters/dallas-2026-poster.jpeg",
    width: 736,
    height: 1122
  },
  {
    id: "guadalajara",
    city: "Guadalajara",
    fixtureGround: "Guadalajara (Zapopan)",
    hostCountry: "Mexico",
    stadiumName: "Estadio Akron",
    caption: "pink festival field, bright folk pattern",
    label: "Guadalajara World Cup 2026 poster with pink festival color and stadium motifs",
    imageSrc: "/world-cup-posters/guadalajara-2026-poster.jpeg",
    width: 760,
    height: 1148
  },
  {
    id: "houston",
    city: "Houston",
    fixtureGround: "Houston",
    hostCountry: "USA",
    stadiumName: "NRG Stadium",
    caption: "space city, night blue, astronaut football",
    label: "Houston World Cup 2026 poster with space city night colors and astronaut football",
    imageSrc: "/world-cup-posters/houston-2026-poster.jpeg",
    width: 854,
    height: 1320
  },
  {
    id: "kansas-city",
    city: "Kansas City",
    fixtureGround: "Kansas City",
    hostCountry: "USA",
    stadiumName: "GEHA Field at Arrowhead Stadium",
    caption: "music strips, match murals, midnight color",
    label: "Kansas City World Cup 2026 poster with layered mural ribbons and football scenes",
    imageSrc: "/world-cup-posters/kansas-city-2026-poster.jpeg",
    width: 764,
    height: 1170
  },
  {
    id: "los-angeles",
    city: "Los Angeles",
    fixtureGround: "Los Angeles (Inglewood)",
    hostCountry: "USA",
    stadiumName: "SoFi Stadium",
    caption: "sunset skyline, palm silhouettes, match glow",
    label: "Los Angeles World Cup 2026 poster with sunset skyline and footballer silhouette",
    imageSrc: "/world-cup-posters/los-angeles-2026-poster.jpeg",
    width: 714,
    height: 1014
  },
  {
    id: "toronto",
    city: "Toronto",
    fixtureGround: "Toronto",
    hostCountry: "Canada",
    stadiumName: "BMO Field",
    caption: "motion, blue blocks, match night red",
    label: "Toronto World Cup 2026 poster with blue blocks and red football motion",
    imageSrc: "/world-cup-posters/toronto-2026-poster.png",
    width: 960,
    height: 1232
  },
  {
    id: "miami",
    city: "Miami",
    fixtureGround: "Miami (Miami Gardens)",
    hostCountry: "USA",
    stadiumName: "Hard Rock Stadium",
    caption: "pink waterfront, teal skyline, festival heat",
    label: "Miami World Cup 2026 poster with pink waterfront colors and a flamingo",
    imageSrc: "/world-cup-posters/miami-2026-poster.png",
    width: 888,
    height: 1348
  },
  {
    id: "vancouver",
    city: "Vancouver",
    fixtureGround: "Vancouver",
    hostCountry: "Canada",
    stadiumName: "BC Place",
    caption: "coast green, sky blue, Pacific linework",
    label: "Vancouver World Cup 2026 poster with coast green, blue, and Pacific artwork",
    imageSrc: "/world-cup-posters/vancouver-2026-poster.png",
    width: 832,
    height: 1284
  },
  {
    id: "mexico",
    city: "Mexico City",
    fixtureGround: "Mexico City",
    hostCountry: "Mexico",
    stadiumName: "Estadio Azteca",
    caption: "orange festival color, stadium geometry",
    label: "Mexico City World Cup 2026 poster with orange festival colors",
    imageSrc: "/world-cup-posters/mexico-city-2026-poster.png",
    width: 288,
    height: 432
  },
  {
    id: "monterrey",
    city: "Monterrey",
    fixtureGround: "Monterrey (Guadalupe)",
    hostCountry: "Mexico",
    stadiumName: "Estadio BBVA",
    caption: "royal blue, green pattern, northern rhythm",
    label: "Monterrey World Cup 2026 poster with royal blue and green stadium pattern",
    imageSrc: "/world-cup-posters/monterrey-2026-poster.jpeg",
    width: 790,
    height: 1192
  },
  {
    id: "new-york",
    city: "New York/New Jersey",
    fixtureGround: "New York/New Jersey (East Rutherford)",
    hostCountry: "USA",
    stadiumName: "MetLife Stadium",
    caption: "torch energy, electric blue, orange flame",
    label: "New York New Jersey World Cup 2026 poster with electric blue and orange torch energy",
    imageSrc: "/world-cup-posters/new-york-new-jersey-2026-poster.png",
    width: 288,
    height: 440
  },
  {
    id: "philadelphia",
    city: "Philadelphia",
    fixtureGround: "Philadelphia",
    hostCountry: "USA",
    stadiumName: "Lincoln Financial Field",
    caption: "spotlight football, blue city collage",
    label: "Philadelphia World Cup 2026 poster with blue city collage and football spotlights",
    imageSrc: "/world-cup-posters/philadelphia-2026-poster.jpeg",
    width: 910,
    height: 1100
  },
  {
    id: "san-francisco",
    city: "San Francisco Bay Area",
    fixtureGround: "San Francisco Bay Area (Santa Clara)",
    hostCountry: "USA",
    stadiumName: "Levi's Stadium",
    caption: "bridge angle, orange football, bay light",
    label: "San Francisco Bay Area World Cup 2026 poster with bridge artwork and floating football",
    imageSrc: "/world-cup-posters/san-francisco-bay-area-2026-poster.jpeg",
    width: 822,
    height: 1208
  },
  {
    id: "seattle",
    city: "Seattle",
    fixtureGround: "Seattle",
    hostCountry: "USA",
    stadiumName: "Lumen Field",
    caption: "coast green, whale mark, mountain energy",
    label: "Seattle World Cup 2026 poster with whale artwork, water, and mountain silhouette",
    imageSrc: "/world-cup-posters/seattle-2026-poster.jpeg",
    width: 806,
    height: 1218
  }
];

type FixtureGroup = {
  name: string;
  teams: string[];
  fixtures: WorldCupFixture[];
};

function modelLabel(mode: PredictionModelMode) {
  if (mode === "calibrated") return "Calibrated ML";
  if (mode === "benchmark") return "ML Benchmark";
  if (mode === "elo") return "Elo + Score";
  return "Legacy Historical";
}

export function PredictorLab({ teams, fixtures, generatedAt, sources }: PredictorLabProps) {
  const [teamA, setTeamA] = useState("Argentina");
  const [teamB, setTeamB] = useState("France");
  const [stage, setStage] = useState<MatchStage>("Final");
  const [predictionModel, setPredictionModel] = useState<PredictionModelMode>("calibrated");
  const [modernSquadEnabled, setModernSquadEnabled] = useState(false);
  const [activeView, setActiveView] = useState<"prediction" | "data" | "fixtures">("prediction");
  const [explanation, setExplanation] = useState<ExplainState>({ status: "idle", text: "", meta: "" });
  const [whatIf, setWhatIf] = useState<WhatIfState>(defaultWhatIf);
  const fixtureGroups = useMemo(() => groupFixtures(fixtures), [fixtures]);
  const activeTeams = useMemo(
    () => [...new Set(fixtureGroups.flatMap((group) => group.teams))].sort((a, b) => a.localeCompare(b)),
    [fixtureGroups]
  );
  const availableStages = useMemo(
    () => validStagesForMatch(teamA, teamB, fixtureGroups),
    [fixtureGroups, teamA, teamB]
  );

  const prediction = useMemo(() => {
    if (predictionModel === "calibrated") return predictMatchWithCalibratedML(teamA, teamB, stage);
    if (predictionModel === "benchmark") return predictMatchWithML(teamA, teamB, stage);
    if (predictionModel === "elo") return predictMatchWithEloScore(teamA, teamB, stage);
    return predictMatch(teamA, teamB, stage);
  }, [predictionModel, teamA, teamB, stage]);
  const squadAdjustedResult = useMemo(
    () => applySquadStrengthAdjustment(prediction, modernSquadEnabled),
    [prediction, modernSquadEnabled]
  );
  const squadPrediction = squadAdjustedResult.prediction;
  const displayPrediction = useMemo(() => alignScoreline(applyWhatIf(squadPrediction, whatIf)), [squadPrediction, whatIf]);
  const modelDetails = useMemo(
    () => (predictionModel === "legacy" ? null : getMLModelDetails(predictionModel)),
    [predictionModel]
  );
  const sortedTeams = activeTeams.length ? activeTeams : teams.filter(Boolean);

  async function explainPrediction() {
    setExplanation({ status: "loading", text: "", meta: "" });
    try {
      const whatIfActive =
        isWhatIfValuesActive(whatIf.teamA) ||
        isWhatIfValuesActive(whatIf.teamB);
      const response = await fetch("/api/explain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          selectedModel: modelLabel(predictionModel),
          prediction: displayPrediction,
          modelDetails,
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
            activeTarget: whatIf.activeTarget,
            teamA: whatIf.teamA,
            teamB: whatIf.teamB
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
          : payload.providerError
            ? "OpenAI unavailable"
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

  function loadBracketMatchup(nextTeamA: string, nextTeamB: string, scenarioTab: ScenarioTab) {
    setTeamA(nextTeamA);
    setTeamB(nextTeamB);
    setStage(scenarioTabToMatchStage(scenarioTab));
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
      setStage(defaultStageForMatch(first, second, fixtureGroups));
    }
    if (scenario === "final") {
      setTeamA("Argentina");
      setTeamB("France");
      setStage("Final");
    }
    if (scenario === "upset") {
      setTeamA("South Africa");
      setTeamB("Brazil");
      setStage("Round of 32");
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
            availableStages={availableStages}
            sortedTeams={sortedTeams}
            teamA={teamA}
            teamB={teamB}
            stage={stage}
            setTeamA={(value) => {
              setTeamA(value);
              if (!validStagesForMatch(value, teamB, fixtureGroups).includes(stage)) {
                setStage(defaultStageForMatch(value, teamB, fixtureGroups));
              }
              clearExplanation();
            }}
            setTeamB={(value) => {
              setTeamB(value);
              if (!validStagesForMatch(teamA, value, fixtureGroups).includes(stage)) {
                setStage(defaultStageForMatch(teamA, value, fixtureGroups));
              }
              clearExplanation();
            }}
            setStage={(value) => {
              setStage(value);
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

          {activeView === "prediction" && <QuickScenarios onApply={applyScenario} />}

          {activeView === "prediction" && (
            <PredictionView
              prediction={prediction}
              displayPrediction={displayPrediction}
              predictionModel={predictionModel}
              modelDetails={modelDetails}
              squadAdjustment={squadAdjustedResult.adjustment}
              explanation={explanation}
              onExplain={explainPrediction}
              whatIf={whatIf}
              setWhatIf={(value) => {
                setWhatIf(value);
                clearExplanation();
              }}
            />
          )}

          {activeView === "data" && <DataView prediction={prediction} generatedAt={generatedAt} sources={sources} />}

          {activeView === "fixtures" && (
            <ScenarioView
              fixtures={fixtures}
              prediction={displayPrediction}
              onLoadFixture={loadFixture}
              onLoadBracketMatchup={loadBracketMatchup}
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
        <div className="brand-visual" aria-label="World Cup 2026 visual identity">
          <Image
            className="brand-logo-image"
            src="/world-cup-posters/can-mex-usa-2026-logo.png"
            alt="FIFA World Cup 2026 CAN MEX USA logo artwork"
            width={588}
            height={886}
            loading="eager"
          />
        </div>
      </div>
      <p className="intro">Build a matchup, test the stage, then stress the forecast with quick what-if scenarios.</p>
    </div>
  );
}

function Controls({
  availableStages,
  sortedTeams,
  teamA,
  teamB,
  stage,
  setTeamA,
  setTeamB,
  setStage
}: {
  availableStages: MatchStage[];
  sortedTeams: string[];
  teamA: string;
  teamB: string;
  stage: MatchStage;
  setTeamA: (team: string) => void;
  setTeamB: (team: string) => void;
  setStage: (stage: MatchStage) => void;
}) {
  return (
    <div className="control-stack">
      <TeamSelect label="Team A" value={teamA} teams={sortedTeams} onChange={setTeamA} />
      <TeamSelect label="Team B" value={teamB} teams={sortedTeams} onChange={setTeamB} />
      <label className="field">
        <span>Match context</span>
        <select value={stage} onChange={(event) => setStage(event.target.value as MatchStage)}>
          {availableStages.map((stageName) => (
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
        <button className={value === "calibrated" ? "active" : ""} type="button" onClick={() => onChange("calibrated")}>
          Calibrated ML
        </button>
        <button className={value === "legacy" ? "active" : ""} type="button" onClick={() => onChange("legacy")}>
          Legacy Historical
        </button>
        <button className={value === "benchmark" ? "active" : ""} type="button" onClick={() => onChange("benchmark")}>
          ML Benchmark
        </button>
        <button className={value === "elo" ? "active" : ""} type="button" onClick={() => onChange("elo")}>
          Elo + Score
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
    ["upset", "David vs Goliath"],
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
  const activeValues = whatIf[whatIf.activeTarget];
  const activeLabel = whatIf.activeTarget === "teamA" ? "Team A" : "Team B";

  function setActiveTarget(activeTarget: WhatIfTarget) {
    setWhatIf({ ...whatIf, activeTarget });
  }

  function updateActiveValues(values: Partial<WhatIfValues>) {
    setWhatIf({
      ...whatIf,
      [whatIf.activeTarget]: {
        ...activeValues,
        ...values
      }
    });
  }

  return (
    <section className="mini-module what-if-module">
      <div className="mini-heading">
        <span>What-if lab</span>
        <small>{activeLabel} boost</small>
      </div>
      <div className="what-if-team-toggle" role="group" aria-label="What-if adjustment team">
        <button
          className={whatIf.activeTarget === "teamA" ? "active" : ""}
          type="button"
          onClick={() => setActiveTarget("teamA")}
        >
          Team A
        </button>
        <button
          className={whatIf.activeTarget === "teamB" ? "active" : ""}
          type="button"
          onClick={() => setActiveTarget("teamB")}
        >
          Team B
        </button>
      </div>
      <SliderControl
        label="Recent form"
        value={activeValues.recentForm}
        onChange={(recentForm) => updateActiveValues({ recentForm })}
      />
      <SliderControl
        label="Attack"
        value={activeValues.attack}
        onChange={(attack) => updateActiveValues({ attack })}
      />
      <SliderControl
        label="Defensive stability"
        value={activeValues.defense}
        onChange={(defense) => updateActiveValues({ defense })}
      />
      <div className="switch-control">
        <button
          className={activeValues.knockoutPressure ? "switch-button active" : "switch-button"}
          type="button"
          role="switch"
          aria-checked={activeValues.knockoutPressure}
          onClick={() => updateActiveValues({ knockoutPressure: !activeValues.knockoutPressure })}
        >
          <span>Knockout pressure: {activeValues.knockoutPressure ? "On" : "Off"}</span>
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
  modelDetails,
  squadAdjustment,
  explanation,
  onExplain,
  whatIf,
  setWhatIf
}: {
  prediction: Prediction;
  displayPrediction: Prediction;
  predictionModel: PredictionModelMode;
  modelDetails: MLModelDetails | null;
  squadAdjustment: SquadAdjustmentState;
  explanation: ExplainState;
  onExplain: () => void;
  whatIf: WhatIfState;
  setWhatIf: (value: WhatIfState) => void;
}) {
  const [activeProbability, setActiveProbability] = useState("");
  const [factorLens, setFactorLens] = useState<WhatIfTarget>("teamA");
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
  const lensTeam = factorLens === "teamA" ? prediction.teamA : prediction.teamB;
  const factorTranslations = useMemo(
    () => buildFactorTranslations(prediction.factors, predictionModel, displayPrediction),
    [prediction.factors, predictionModel, displayPrediction]
  );
  const displayedFactors =
    factorLens === "teamA"
      ? prediction.factors
      : mirrorPredictionFactors(prediction.factors, prediction.teamA, prediction.teamB);

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
          <Metric label="Model" value={modelLabel(predictionModel)} />
          <Metric label="Match Type" value={displayPrediction.stage} />
          {squadAdjustment.status !== "disabled" && (
            <Metric
              label="Squad layer"
              value={squadAdjustment.status === "applied" ? squadAdjustment.message : "Unavailable"}
            />
          )}
        </div>
      </section>

      <section className={`panel ai-panel ${explanation.status}`}>
        <div className="section-heading">
          <h3>AI Analyst Brief</h3>
          {explanation.status === "loading" && <span>Building brief</span>}
          {explanation.status === "error" && <span>Unavailable</span>}
        </div>
        <AnalystBrief explanation={explanation} prediction={displayPrediction} />
        <button className="primary-button" type="button" onClick={onExplain} disabled={explanation.status === "loading"}>
          {explanation.status === "loading" ? "Generating..." : "Generate"}
        </button>
      </section>

      <div className="forecast-tools">
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
                : predictionModel === "calibrated"
                  ? "Why this matters: Calibrated ML uses static probabilities exported from the offline Python pipeline, not betting odds."
                : predictionModel === "benchmark"
                  ? "Why this matters: ML Benchmark is an explainable Logistic Regression comparison, not the recommended default."
                  : predictionModel === "elo"
                    ? "Why this matters: Elo + Score converts expected goals into model-estimated probabilities, not betting odds."
                  : "Why this matters: Legacy Historical translates World Cup records into a quick match-read before the analyst brief.")}
          </p>
        </section>
        <WhatIfLab whatIf={whatIf} setWhatIf={setWhatIf} />
      </div>

      <section className="panel factors-panel">
        <div className="section-heading">
          <h3>Key factors</h3>
          <div className="factor-lens-toggle" role="group" aria-label="Key factors lens">
            <button
              className={factorLens === "teamA" ? "active" : ""}
              type="button"
              onClick={() => setFactorLens("teamA")}
            >
              <TeamBadge team={prediction.teamA} /> {prediction.teamA}
            </button>
            <button
              className={factorLens === "teamB" ? "active" : ""}
              type="button"
              onClick={() => setFactorLens("teamB")}
            >
              <TeamBadge team={prediction.teamB} /> {prediction.teamB}
            </button>
          </div>
        </div>
        <p className="lens-note">{lensTeam} lens</p>
        <div className="factors-layout">
          <div className="factor-list">
            {displayedFactors.map((factor) => (
              <div className={`factor ${factor.impact}`} key={factor.label}>
                <span>{factor.label}</span>
                <p>{factor.value}</p>
              </div>
            ))}
          </div>
          <aside className="factor-translation" aria-label="Football translation">
            <h4>Football translation</h4>
            <ul>
              {factorTranslations.map(({ label, copy }) => (
                <li key={label}>
                  <span>{label}</span>
                  <p>{copy}</p>
                </li>
              ))}
            </ul>
          </aside>
        </div>
      </section>

      {modelDetails && <ModelDetailsPanel details={modelDetails} squadAdjustment={squadAdjustment} />}
    </div>
  );
}

function ModelDetailsPanel({
  details,
  squadAdjustment
}: {
  details: MLModelDetails;
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
        {details.brierScore !== undefined && <Metric label="Brier score" value={details.brierScore.toFixed(3)} />}
        {details.logLoss !== undefined && <Metric label="Log loss" value={details.logLoss.toFixed(3)} />}
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
        Unlock the tactical read behind this forecast: the edge, the danger, and the upset route in one compact match brief.
      </p>
    );
  }

  if (explanation.status === "loading") {
    return <p className="ai-copy">Generating brief...</p>;
  }

  const favorite = prediction.favorite === "Toss-up" ? prediction.teamA : prediction.favorite;
  const riskTeam = favorite === prediction.teamA ? prediction.teamB : prediction.teamA;
  const fallbackSections = {
    keyTakeaway: explanation.text,
    whyFavorite:
      favorite === "Toss-up"
        ? "The model sees a narrow spread, so the stronger match moments matter more than the headline rating."
        : `${favorite} owns the current edge through the selected model, score projection, and matchup factors.`,
    riskFactor: `${riskTeam} can keep this close if the game state lowers tempo or turns the forecast into a one-chance match.`,
    upsetPath: "The underdog route is early defensive control, set-piece pressure, and forcing the favorite away from its normal scoring rhythm.",
    modelLimitation: "OpenAI explains the selected forecast but does not make the prediction. Probabilities are model-estimated, not betting odds."
  };
  const structured = {
    keyTakeaway: nonEmptyBriefValue(explanation.sections?.keyTakeaway) ?? nonEmptyBriefValue(fallbackSections.keyTakeaway) ?? fallbackSections.whyFavorite,
    whyFavorite: nonEmptyBriefValue(explanation.sections?.whyFavorite) ?? fallbackSections.whyFavorite,
    riskFactor: nonEmptyBriefValue(explanation.sections?.riskFactor) ?? fallbackSections.riskFactor,
    upsetPath: nonEmptyBriefValue(explanation.sections?.upsetPath) ?? fallbackSections.upsetPath,
    modelLimitation: nonEmptyBriefValue(explanation.sections?.modelLimitation) ?? fallbackSections.modelLimitation
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

function nonEmptyBriefValue(value: unknown) {
  return typeof value === "string" && value.trim() ? value.trim() : null;
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
            <RecordMetric stats={stats} />
            <Metric label="World Cup win rate" value={`${Math.round((stats.wins / Math.max(stats.matches, 1)) * 100)}%`} />
            <Metric label="Goals for" value={String(stats.goalsFor)} />
            <Metric label="Goals against" value={String(stats.goalsAgainst)} />
            <Metric label="Recent form index" value={`${Math.round(stats.recentForm * 100)}`} />
            <Metric label="Tournaments played" value={String(teamInsight(stats).tournaments)} />
            <Metric label="World Cup trophies" value={String(stats.titles)} />
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

function RecordMetric({ stats }: { stats: TeamStats }) {
  return (
    <div className="metric record-metric">
      <span>Record</span>
      <div className="record-chips" aria-label={`${stats.wins} wins, ${stats.draws} draws, ${stats.losses} losses`}>
        <strong className="record-chip wins">{stats.wins}W</strong>
        <strong className="record-chip draws">{stats.draws}D</strong>
        <strong className="record-chip losses">{stats.losses}L</strong>
      </div>
    </div>
  );
}

function ScenarioView({
  fixtures,
  prediction,
  onLoadFixture,
  onLoadBracketMatchup
}: {
  fixtures: WorldCupFixture[];
  prediction: Prediction;
  onLoadFixture: (fixture: WorldCupFixture) => void;
  onLoadBracketMatchup: (teamA: string, teamB: string, scenarioTab: ScenarioTab) => void;
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
      <TournamentSimulationPanel prediction={prediction} />
      <HostCityEnergy fixtures={fixtures} onLoadFixture={onLoadFixture} />
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
        <BracketView fixtureGroups={fixtureGroups} onLoadMatchup={onLoadBracketMatchup} scenarioTab={scenarioTab} />
      )}
    </div>
  );
}

function TournamentSimulationPanel({ prediction }: { prediction: Prediction }) {
  const probabilities = typedWorldCupSimulation.data?.probabilities ?? {};
  const runs = typedWorldCupSimulation.data?.runs ?? 0;
  const topTeams = Object.entries(probabilities)
    .sort(([, teamA], [, teamB]) => teamB.champion_probability - teamA.champion_probability)
    .slice(0, 5);
  const selectedTeams = [prediction.teamA, prediction.teamB]
    .map((team) => [team, probabilities[simulationTeamAliases[team] ?? team]] as const)
    .filter((entry): entry is readonly [string, SimulationProbability] => Boolean(entry[1]));

  if (!topTeams.length) return null;

  return (
    <section className="panel wide-panel simulation-panel">
      <header className="simulation-header">
        <div className="simulation-header-text">
          <p className="eyebrow">Monte Carlo tournament outlook</p>
          <h3>The 2026 World Cup Simulated</h3>
          <p className="simulation-subtitle">
            After {runs.toLocaleString()} offline tournament simulations, here is the projected outlook for the strongest contenders.
          </p>
        </div>
        <span className="simulation-runs-pill">{runs.toLocaleString()} offline runs</span>
      </header>
      <div className="simulation-grid">
        <div className="simulation-card featured">
          <span>Current matchup</span>
          <div className="simulation-team-pair">
            {selectedTeams.map(([team, outlook]) => (
              <strong key={team}>
                <TeamBadge team={team} /> {team}
                <small>{formatSimulationPercent(outlook.champion_probability)} champion path</small>
              </strong>
            ))}
          </div>
          <p>Scenario-only tournament simulation; single-match probabilities still come from the selected model.</p>
        </div>
        {topTeams.map(([simulationTeam, outlook]) => {
          const team = simulationToAppTeamAliases[simulationTeam] ?? simulationTeam;
          return (
            <div className="simulation-card" key={simulationTeam}>
              <span>Champion path</span>
              <strong>
                <TeamBadge team={team} /> {team}
              </strong>
              <div className="simulation-bar" aria-label={`${team} champion probability ${formatSimulationPercent(outlook.champion_probability)}`}>
                <i style={{ width: `${Math.max(4, Math.round(outlook.champion_probability * 100))}%` }} />
              </div>
              <small>
                {formatSimulationPercent(outlook.champion_probability)} win tournament · {formatSimulationPercent(outlook.finalist_probability)} final
              </small>
            </div>
          );
        })}
      </div>
    </section>
  );
}

function HostCityEnergy({
  fixtures,
  onLoadFixture
}: {
  fixtures: WorldCupFixture[];
  onLoadFixture: (fixture: WorldCupFixture) => void;
}) {
  const [selectedCity, setSelectedCity] = useState(hostCityPosters[0]?.fixtureGround ?? "");
  const cityRailRef = useRef<HTMLDivElement>(null);
  const cityFixtures = useMemo(() => fixtures.filter((fixture) => fixture.ground === selectedCity), [fixtures, selectedCity]);
  const selectedPoster = hostCityPosters.find((poster) => poster.fixtureGround === selectedCity) ?? hostCityPosters[0];

  function scrollCities(direction: -1 | 1) {
    const rail = cityRailRef.current;
    if (!rail) return;
    rail.scrollBy({
      left: direction * Math.max(rail.clientWidth * 0.85, 280),
      behavior: "smooth"
    });
  }

  return (
    <section className="panel wide-panel city-energy-panel">
      <div className="section-heading">
        <h3>Host City Fixtures</h3>
        <span>{hostCityPosters.length} host cities</span>
      </div>
      <div className="city-selector-shell">
        <button
          aria-label="Scroll host cities left"
          className="city-scroll-button"
          onClick={() => scrollCities(-1)}
          type="button"
        >
          ‹
        </button>
        <div className="city-scroll-window">
          <div className="city-poster-grid" ref={cityRailRef} aria-label="World Cup 2026 host city selector">
            {hostCityPosters.map((poster) => {
              const cityMatchCount = fixtures.filter((fixture) => fixture.ground === poster.fixtureGround).length;
              const isSelected = selectedCity === poster.fixtureGround;

              return (
                <button
                  aria-label={`Show ${poster.city} fixtures at ${poster.stadiumName}`}
                  aria-pressed={isSelected}
                  className={`city-poster ${poster.id} ${isSelected ? "active" : ""}`}
                  key={poster.id}
                  onClick={() => setSelectedCity(poster.fixtureGround)}
                  type="button"
                >
                  <div className="city-poster-art">
                    {poster.imageSrc ? (
                      <Image
                        src={poster.imageSrc}
                        alt={poster.label}
                        width={poster.width ?? 800}
                        height={poster.height ?? 1100}
                        sizes="(max-width: 680px) 160px, (max-width: 1040px) 190px, 220px"
                      />
                    ) : (
                      <span className="city-fallback-art" aria-hidden="true">
                        {poster.city.slice(0, 3).toUpperCase()}
                      </span>
                    )}
                  </div>
                  <span className="city-poster-meta">
                    <strong>{poster.city}</strong>
                    <small className="city-poster-kicker">{poster.hostCountry}</small>
                    <small className="city-poster-stadium">{poster.stadiumName}</small>
                    <small className="city-poster-count">{cityMatchCount} matches</small>
                  </span>
                </button>
              );
            })}
          </div>
        </div>
        <button
          aria-label="Scroll host cities right"
          className="city-scroll-button"
          onClick={() => scrollCities(1)}
          type="button"
        >
          ›
        </button>
      </div>
      {selectedPoster && (
        <div className="city-fixture-panel">
          <div className="selected-city-header">
            <div>
              <span>{selectedPoster.hostCountry}</span>
              <h4>{selectedPoster.city}</h4>
              <p>{selectedPoster.stadiumName}</p>
            </div>
            <strong>{cityFixtures.length} matches hosted</strong>
          </div>
          <div className="city-fixture-list">
            {cityFixtures.map((fixture) => (
              <button
                className="city-fixture"
                key={`${fixture.date}-${fixture.team1}-${fixture.team2}-${fixture.ground}`}
                onClick={() => onLoadFixture(fixture)}
                type="button"
              >
                <span className="city-fixture-date">{fixture.date}</span>
                <strong>
                  <TeamBadge team={fixture.team1} /> {fixture.team1}
                  <em>vs</em>
                  <TeamBadge team={fixture.team2} /> {fixture.team2}
                </strong>
                <small>
                  {fixture.group} · {fixture.round}
                </small>
              </button>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}

function BracketView({
  fixtureGroups,
  onLoadMatchup,
  scenarioTab
}: {
  fixtureGroups: FixtureGroup[];
  onLoadMatchup: (teamA: string, teamB: string, scenarioTab: ScenarioTab) => void;
  scenarioTab: Exclude<ScenarioTab, "Group Stage">;
}) {
  const matchups = bracketMatchups(scenarioTab);
  const [selectedSlots, setSelectedSlots] = useState<Record<string, string>>({});
  const allGroupTeams = useMemo(
    () => [...new Set(fixtureGroups.flatMap((group) => group.teams))].sort((a, b) => a.localeCompare(b)),
    [fixtureGroups]
  );

  function optionsForSlot(slot: string) {
    const groupName = extractGroupName(slot);
    if (!groupName) return allGroupTeams;
    return fixtureGroups.find((group) => group.name === groupName)?.teams ?? allGroupTeams;
  }

  function selectedValue(matchupId: string, side: "home" | "away", slot: string) {
    const key = `${scenarioTab}-${matchupId}-${side}`;
    const options = optionsForSlot(slot);
    return selectedSlots[key] && options.includes(selectedSlots[key]) ? selectedSlots[key] : options[0] ?? "";
  }

  function updateSlot(matchupId: string, side: "home" | "away", value: string) {
    setSelectedSlots((current) => ({
      ...current,
      [`${scenarioTab}-${matchupId}-${side}`]: value
    }));
  }

  return (
    <section
      className={`panel wide-panel bracket-panel${scenarioTab === "Final" ? " final-bracket" : ""}${scenarioTab === "Semi-final" ? " semi-final-bracket" : ""}`}
    >
      <div className="section-heading">
        <h3>{scenarioTab === "Final" ? "The Final" : `${scenarioTab} bracket`}</h3>
        {scenarioTab !== "Final" && <span>{matchups.length} {matchups.length === 1 ? "matchup" : "matchups"}</span>}
      </div>
      <div className="bracket-grid">
        {matchups.map((matchup) => (
          <BracketCard
            awayValue={selectedValue(matchup.id, "away", matchup.away)}
            homeValue={selectedValue(matchup.id, "home", matchup.home)}
            key={matchup.id}
            matchup={matchup}
            onChangeAway={(value) => updateSlot(matchup.id, "away", value)}
            onChangeHome={(value) => updateSlot(matchup.id, "home", value)}
            onLoad={() =>
              onLoadMatchup(
                selectedValue(matchup.id, "home", matchup.home),
                selectedValue(matchup.id, "away", matchup.away),
                scenarioTab
              )
            }
            optionsForSlot={optionsForSlot}
          />
        ))}
      </div>
    </section>
  );
}

function BracketCard({
  awayValue,
  homeValue,
  matchup,
  onChangeAway,
  onChangeHome,
  onLoad,
  optionsForSlot
}: {
  awayValue: string;
  homeValue: string;
  matchup: ReturnType<typeof toBracketMatchup>;
  onChangeAway: (value: string) => void;
  onChangeHome: (value: string) => void;
  onLoad: () => void;
  optionsForSlot: (slot: string) => string[];
}) {
  const canLoad = Boolean(homeValue && awayValue && homeValue !== awayValue);

  return (
    <article className="bracket-card">
      <span className="bracket-meta">{matchup.id}</span>
      <BracketSlotSelect
        label={matchup.home}
        onChange={onChangeHome}
        options={optionsForSlot(matchup.home)}
        value={homeValue}
      />
      <em>vs</em>
      <BracketSlotSelect
        label={matchup.away}
        onChange={onChangeAway}
        options={optionsForSlot(matchup.away)}
        value={awayValue}
      />
      <button className="bracket-load-button" disabled={!canLoad} onClick={onLoad} type="button">
        {canLoad ? "Use matchup" : "Pick two teams"}
      </button>
    </article>
  );
}

function BracketSlotSelect({
  label,
  onChange,
  options,
  value
}: {
  label: string;
  onChange: (value: string) => void;
  options: string[];
  value: string;
}) {
  return (
    <label className="bracket-slot-select">
      <span>{label}</span>
      <div className="select-shell bracket-select-shell">
        {value && <TeamBadge team={value} />}
        <select aria-label={label} value={value} onChange={(event) => onChange(event.target.value)}>
          {options.map((team) => (
            <option key={`${label}-${team}`} value={team}>
              {team}
            </option>
          ))}
        </select>
      </div>
    </label>
  );
}

function groupFixtures(fixtures: WorldCupFixture[]): FixtureGroup[] {
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

function extractGroupName(slot: string) {
  const match = slot.match(/Group [A-L]/);
  return match?.[0] ?? "";
}

function scenarioTabToMatchStage(scenarioTab: ScenarioTab): MatchStage {
  if (scenarioTab === "Group Stage") return "Group stage";
  if (scenarioTab === "Round of 32") return "Round of 32";
  if (scenarioTab === "Quarter-final") return "Quarter-final";
  if (scenarioTab === "Semi-final") return "Semi-final";
  if (scenarioTab === "Final") return "Final";
  return "Round of 16";
}

function validStagesForMatch(teamA: string, teamB: string, fixtureGroups: FixtureGroup[]): MatchStage[] {
  const stages = [...knockoutStages];
  if (hasGroupFixture(teamA, teamB, fixtureGroups)) {
    return ["Group stage", ...stages];
  }
  return stages;
}

function defaultStageForMatch(teamA: string, teamB: string, fixtureGroups: FixtureGroup[]): MatchStage {
  return hasGroupFixture(teamA, teamB, fixtureGroups) ? "Group stage" : "Round of 32";
}

function hasGroupFixture(teamA: string, teamB: string, fixtureGroups: FixtureGroup[]) {
  return fixtureGroups.some((group) =>
    group.fixtures.some(
      (fixture) =>
        (fixture.team1 === teamA && fixture.team2 === teamB) ||
        (fixture.team1 === teamB && fixture.team2 === teamA)
    )
  );
}

function isWhatIfValuesActive(values: WhatIfValues) {
  return values.recentForm > 0 || values.attack > 0 || values.defense > 0 || values.knockoutPressure;
}

function whatIfBoost(values: WhatIfValues, stage: MatchStage) {
  return (
    values.recentForm * 0.7 +
    values.attack * 0.8 +
    values.defense * 0.5 +
    (values.knockoutPressure && stage !== "Group stage" ? 4 : 0)
  );
}

function applyWhatIf(prediction: Prediction, whatIf: WhatIfState): Prediction {
  const teamABoost = whatIfBoost(whatIf.teamA, prediction.stage);
  const teamBBoost = whatIfBoost(whatIf.teamB, prediction.stage);
  const netBoost = teamABoost - teamBBoost;
  if (netBoost === 0) return prediction;

  const shift = Math.abs(netBoost);
  const drawFloor = prediction.stage === "Group stage" ? 10 : 6;
  const drawDrop = Math.min(Math.max(0, prediction.probabilities.draw - drawFloor), shift * 0.28);
  const draw = clampPercent(prediction.probabilities.draw - drawDrop, drawFloor, 42);
  let teamAWin = prediction.probabilities.teamAWin;
  let teamBWin = prediction.probabilities.teamBWin;

  if (netBoost > 0) {
    teamAWin = clampPercent(teamAWin + shift, 5, 88);
    teamBWin = clampPercent(100 - teamAWin - draw, 4, 88);
  } else {
    teamBWin = clampPercent(teamBWin + shift, 5, 88);
    teamAWin = clampPercent(100 - teamBWin - draw, 4, 88);
  }

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

type FactorTranslation = { label: string; copy: string };

function buildFactorTranslations(
  factors: PredictionFactor[],
  predictionModel: PredictionModelMode,
  prediction: Prediction
): FactorTranslation[] {
  const out: FactorTranslation[] = [];
  const fav = prediction.favorite;
  const favouriteName = fav && fav !== "Toss-up" ? fav : null;
  const find = (needle: string) =>
    factors.find((f) => f.label.toLowerCase().includes(needle));

  const strength =
    find("elo") ?? find("strength") ?? find("rating") ?? find("historical") ?? find("difference");
  if (strength) {
    out.push({
      label: "Underlying strength",
      copy:
        strength.impact === "neutral"
          ? "The model sees very little separation between the two sides on long-term form."
          : favouriteName
            ? `${favouriteName} carries a noticeable underlying-strength edge, though football can still swing on moments.`
            : "There is a real underlying-strength edge in this matchup, but football can still swing on moments.",
    });
  }

  const goals = find("goal") ?? find("attack") ?? find("profile");
  if (goals) {
    out.push({
      label: "Chance quality",
      copy:
        goals.impact === "neutral"
          ? "Both sides project to create chances at a similar quality — expect a tight game in front of goal."
          : favouriteName
            ? `${favouriteName} is expected to produce the slightly better looks; the underdog will need a more clinical night to bridge that gap.`
            : "One side is projected to create slightly better chances, but neither is dominant.",
    });
  }

  const form = find("form") ?? find("momentum");
  if (form) {
    out.push({
      label: "Momentum",
      copy:
        form.impact === "neutral"
          ? "No real momentum edge — recent form is roughly level."
          : favouriteName
            ? `Recent form points to ${favouriteName} trending a bit better coming in.`
            : "Recent form gives one side a slight lift, but it's marginal.",
    });
  }

  out.push({
    label: "Stage context",
    copy:
      prediction.stage === "Group stage"
        ? "Group stage allows more variance — a single result can still come from a low-volume chance."
        : "Knockout football tightens up — single moments matter more than over a group phase.",
  });

  const modelCopy: Record<PredictionModelMode, string> = {
    calibrated:
      "Calibrated ML — trained on recent senior international results, weighted toward modern form.",
    legacy:
      "Legacy Historical — leans on older World Cup patterns; treat tight gaps with more skepticism.",
    benchmark:
      "ML Benchmark — a simpler comparison read; useful for sanity-checking the headline model.",
    elo:
      "Elo + Score — team-strength rating combined with an expected-chance-quality scoreline.",
  };
  out.push({ label: "Model lens", copy: modelCopy[predictionModel] });
  out.push({
    label: "Venue",
    copy: "Neutral venue — no team gets a home-field boost in this match.",
  });

  return out;
}

function mirrorPredictionFactors(factors: PredictionFactor[], teamA: string, teamB: string): PredictionFactor[] {
  return factors.map((factor) => ({
    ...factor,
    value: swapTeamNames(factor.value, teamA, teamB),
    impact: flipFactorImpact(factor.impact)
  }));
}

function swapTeamNames(value: string, teamA: string, teamB: string) {
  return value
    .replaceAll(teamA, "__TEAM_A__")
    .replaceAll(teamB, teamA)
    .replaceAll("__TEAM_A__", teamB);
}

function flipFactorImpact(impact: PredictionFactor["impact"]): PredictionFactor["impact"] {
  if (impact === "positive") return "negative";
  if (impact === "negative") return "positive";
  return "neutral";
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

function formatSimulationPercent(value: number) {
  return `${Math.round(value * 100)}%`;
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
