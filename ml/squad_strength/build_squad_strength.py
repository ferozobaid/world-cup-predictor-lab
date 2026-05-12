import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INPUT = Path(__file__).with_name("modern_squad_strength.csv")
OUTPUT = ROOT / "src" / "data" / "squad-strength.json"

WEIGHTS = {
    "fifa_points": 0.35,
    "squad_market_value_eur_m": 0.30,
    "top_11_market_value_eur_m": 0.15,
    "total_caps": 0.08,
    "club_strength_index": 0.07,
    "recent_competitive_form": 0.05,
}

LOG_FIELDS = {"squad_market_value_eur_m", "top_11_market_value_eur_m"}


def parse_number(value):
    if value is None or value == "":
        return None
    return float(value)


def normalize(value, low, high):
    if value is None or high == low:
        return None
    return (value - low) / (high - low)


def main():
    rows = []
    with INPUT.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)

    transformed = {}
    ranges = {}
    for field in WEIGHTS:
        values = []
        for row in rows:
            value = parse_number(row.get(field))
            if value is None:
                continue
            values.append(math.log1p(value) if field in LOG_FIELDS else value)
        ranges[field] = {
            "min": min(values) if values else 0,
            "max": max(values) if values else 0,
        }

    for row in rows:
        score_parts = []
        used_weight = 0
        components = {}

        for field, weight in WEIGHTS.items():
            raw_value = parse_number(row.get(field))
            value = math.log1p(raw_value) if raw_value is not None and field in LOG_FIELDS else raw_value
            normalized = normalize(value, ranges[field]["min"], ranges[field]["max"])
            if normalized is None:
                continue
            score_parts.append(normalized * weight)
            used_weight += weight
            components[field] = round(normalized * 100, 1)

        score = round((sum(score_parts) / used_weight) * 100, 1) if used_weight else None
        canonical_team = row["canonical_team"]
        transformed[canonical_team] = {
            "snapshotId": row["snapshot_id"],
            "snapshotYear": int(row["snapshot_year"]),
            "asOfDate": row["as_of_date"],
            "competitionSource": row["competition_source"],
            "sourceType": row["source_type"],
            "sourceUrl": row["source_url"],
            "rawTeamName": row["raw_team_name"],
            "canonicalTeam": canonical_team,
            "playerCount": int(row["player_count"]) if row["player_count"] else None,
            "fifaRank": int(row["fifa_rank"]) if row["fifa_rank"] else None,
            "fifaPoints": parse_number(row["fifa_points"]),
            "squadMarketValueEurM": parse_number(row["squad_market_value_eur_m"]),
            "top11MarketValueEurM": parse_number(row["top_11_market_value_eur_m"]),
            "top5MarketValueEurM": parse_number(row["top_5_market_value_eur_m"]),
            "avgAge": parse_number(row["avg_age"]),
            "totalCaps": parse_number(row["total_caps"]),
            "avgCaps": parse_number(row["avg_caps"]),
            "clubStrengthIndex": parse_number(row["club_strength_index"]),
            "recentCompetitiveForm": parse_number(row["recent_competitive_form"]),
            "dataQuality": row["data_quality"],
            "notes": row["notes"],
            "squadStrengthScore": score,
            "scoreComponents": components,
        }

    payload = {
        "generatedAt": "2026-05-12",
        "method": {
            "description": "Manual curated modern squad-strength proxy layer. Numeric fields are normalized within this limited sample; squad values are log-scaled.",
            "weights": WEIGHTS,
            "logScaledFields": sorted(LOG_FIELDS),
        },
        "teams": transformed,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Wrote {OUTPUT.relative_to(ROOT)} with {len(transformed)} teams")


if __name__ == "__main__":
    main()
