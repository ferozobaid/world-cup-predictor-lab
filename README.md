# World Cup Predictor Lab

A Next.js football analytics app that predicts World Cup-style matchups from historical FIFA World Cup match data and uses the OpenAI API only for short cached analyst explanations.

## Local Setup

```bash
npm install
npm run generate:data
npm run dev
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
