import fs from "node:fs";
import path from "node:path";

const root = process.cwd();
const matchesPath = path.join(root, "public/data/matches.csv");
const fixturesPath = path.join(root, "public/data/worldcup-2026.json");
const outputPath = path.join(root, "src/data/world-cup-data.json");

function parseCsv(text) {
  const rows = [];
  let row = [];
  let value = "";
  let quoted = false;

  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    const next = text[index + 1];

    if (char === '"' && quoted && next === '"') {
      value += '"';
      index += 1;
      continue;
    }

    if (char === '"') {
      quoted = !quoted;
      continue;
    }

    if (char === "," && !quoted) {
      row.push(value);
      value = "";
      continue;
    }

    if ((char === "\n" || char === "\r") && !quoted) {
      if (char === "\r" && next === "\n") index += 1;
      row.push(value);
      if (row.some((cell) => cell.length > 0)) rows.push(row);
      row = [];
      value = "";
      continue;
    }

    value += char;
  }

  if (value.length || row.length) {
    row.push(value);
    rows.push(row);
  }

  const [headers, ...records] = rows;
  return records.map((record) =>
    Object.fromEntries(headers.map((header, index) => [header, record[index] ?? ""]))
  );
}

const rows = parseCsv(fs.readFileSync(matchesPath, "utf8"));
const matches = rows
  .filter((row) => row.tournament_name.includes("Men's World Cup"))
  .map((row) => ({
    id: row.match_id,
    year: Number(row.tournament_id.replace("WC-", "")),
    date: row.match_date,
    stage: row.stage_name,
    group: row.group_name || "Knockout",
    groupStage: row.group_stage === "1",
    knockoutStage: row.knockout_stage === "1",
    hostCountry: row.country_name,
    homeTeam: row.home_team_name,
    awayTeam: row.away_team_name,
    homeCode: row.home_team_code,
    awayCode: row.away_team_code,
    homeScore: Number(row.home_team_score),
    awayScore: Number(row.away_team_score),
    homeWin: row.home_team_win === "1",
    awayWin: row.away_team_win === "1",
    draw: row.draw === "1"
  }))
  .filter((match) => match.year <= 2022)
  .sort((a, b) => a.date.localeCompare(b.date));

const fixtures = JSON.parse(fs.readFileSync(fixturesPath, "utf8")).matches.slice(0, 72);

const teamNames = new Set();
for (const match of matches) {
  teamNames.add(match.homeTeam);
  teamNames.add(match.awayTeam);
}
for (const fixture of fixtures) {
  teamNames.add(fixture.team1);
  teamNames.add(fixture.team2);
}

const payload = {
  generatedAt: new Date().toISOString(),
  sources: [
    {
      label: "Fjelstul World Cup Database",
      url: "https://github.com/jfjelstul/worldcup"
    },
    {
      label: "openfootball World Cup JSON",
      url: "https://github.com/openfootball/worldcup.json"
    }
  ],
  teams: [...teamNames].sort(),
  matches,
  fixtures2026: fixtures
};

fs.writeFileSync(outputPath, `${JSON.stringify(payload)}\n`);
console.log(`Generated ${matches.length} men's World Cup matches for ${payload.teams.length} teams.`);
