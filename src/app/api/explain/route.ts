import OpenAI from "openai";
import { NextResponse } from "next/server";

export const runtime = "nodejs";

const model = process.env.OPENAI_MODEL ?? "gpt-5-nano";
const analystBriefVersion = "2026-05-analyst-brief-v4";
const maxOutputTokens = 700;
type AnalystSections = {
  keyTakeaway: string;
  whyFavorite: string;
  riskFactor: string;
  upsetPath: string;
  modelLimitation: string;
};

type ExplainPrediction = {
  teamA?: string;
  teamB?: string;
  stage?: string;
  probabilities?: unknown;
  likelyScore?: unknown;
  confidence?: string;
  favorite?: string;
  factors?: unknown;
};

const explanationCache = new Map<string, { text: string; sections: AnalystSections; createdAt: string }>();

function getOpenAIClient() {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) return null;
  return new OpenAI({ apiKey });
}

function stableKey(value: unknown) {
  return `${model}:${JSON.stringify(value)}`;
}

export async function POST(request: Request) {
  const body = await request.json();
  const prediction = (body?.prediction ?? body) as ExplainPrediction;
  const key = stableKey({
    analystBriefVersion,
    teamA: prediction?.teamA,
    teamB: prediction?.teamB,
    stage: prediction?.stage,
    selectedModel: body?.selectedModel,
    probabilities: prediction?.probabilities,
    likelyScore: prediction?.likelyScore,
    favorite: prediction?.favorite,
    confidence: prediction?.confidence,
    squadProxy: body?.squadProxy,
    whatIf: body?.whatIf
  });

  const cached = explanationCache.get(key);
  if (cached) {
    return NextResponse.json({ text: cached.text, sections: cached.sections, cached: true, model, usage: null });
  }

  const client = getOpenAIClient();
  if (!client) {
    const sections = fallbackSections({
      keyTakeaway:
        "AI analysis is ready to enable. Add OPENAI_API_KEY in your local .env.local or Vercel project settings; the prediction model itself is already running locally without API spend.",
      prediction,
      selectedModel: body?.selectedModel,
      squadProxy: body?.squadProxy
    });
    return NextResponse.json(
      { text: sections.keyTakeaway, sections, cached: false, model, usage: null, missingKey: true },
      { status: 200 }
    );
  }

  const analystContext = {
    selectedModel: body?.selectedModel,
    prediction: {
      teamA: prediction.teamA,
      teamB: prediction.teamB,
      stage: prediction.stage,
      probabilities: prediction.probabilities,
      likelyScore: prediction.likelyScore,
      confidence: prediction.confidence,
      favorite: prediction.favorite,
      factors: prediction.factors
    },
    modelDetails: body?.modelDetails ?? null,
    squadProxy: body?.squadProxy ?? null,
    whatIf: body?.whatIf ?? null
  };

  let rawText = "";
  let usage = null;
  try {
    const response = await client.responses.create({
      model,
      reasoning: { effort: "minimal" },
      max_output_tokens: maxOutputTokens,
      input: [
        {
          role: "system",
          content:
            [
              "You are a concise football analyst. OpenAI explains the forecast but does not make or override the prediction.",
              "Use only the supplied JSON context. Do not add external facts.",
              "Return only valid JSON with keys: keyTakeaway, whyFavorite, riskFactor, upsetPath, modelLimitation.",
              "Return the JSON object immediately and do not leave any value blank.",
              "Keep each value to one concise sentence. State probabilities are model-estimated, not betting odds.",
              "If selectedModel is Experimental ML, say it is an experimental offline Logistic Regression model trained on historical World Cup features.",
              "If selectedModel is Elo + Score, say it is an offline Elo-style expected-goals model that derives probabilities from a Poisson score grid.",
              "If squadProxy is enabled/applied, say it is a manually curated sample/proxy layer, not official squad data."
            ].join(" ")
        },
        {
          role: "user",
          content: JSON.stringify(analystContext)
        }
      ]
    });
    rawText = response.output_text.trim();
    usage = response.usage ?? null;
  } catch {
    const sections = fallbackSections({
      keyTakeaway:
        "OpenAI could not generate the analyst brief with the current API key or billing setup. The local prediction remains available.",
      prediction,
      selectedModel: body?.selectedModel,
      squadProxy: body?.squadProxy
    });
    return NextResponse.json({
      text: sections.keyTakeaway,
      sections,
      cached: false,
      model,
      usage: null,
      providerError: true
    });
  }

  const fallback = fallbackSections({
    keyTakeaway:
      rawText ||
      "The selected forecast is ready, but OpenAI returned an empty analyst note. The local prediction remains available.",
    prediction,
    selectedModel: body?.selectedModel,
    squadProxy: body?.squadProxy
  });
  const sections = mergeSections(parseSections(rawText), fallback);
  const text = sections.keyTakeaway;
  explanationCache.set(key, { text, sections, createdAt: new Date().toISOString() });

  return NextResponse.json({
    text,
    sections,
    cached: false,
    model,
    usage
  });
}

function parseSections(value: string): Partial<AnalystSections> | null {
  if (!value.trim()) return null;
  try {
    const cleaned = value.replace(/^```json\s*/i, "").replace(/^```\s*/i, "").replace(/```$/i, "").trim();
    const parsed = JSON.parse(cleaned) as Partial<AnalystSections>;
    return parsed;
  } catch {
    return null;
  }
}

function mergeSections(parsed: Partial<AnalystSections> | null, fallback: AnalystSections): AnalystSections {
  return {
    keyTakeaway: nonEmpty(parsed?.keyTakeaway) ?? fallback.keyTakeaway,
    whyFavorite: nonEmpty(parsed?.whyFavorite) ?? fallback.whyFavorite,
    riskFactor: nonEmpty(parsed?.riskFactor) ?? fallback.riskFactor,
    upsetPath: nonEmpty(parsed?.upsetPath) ?? fallback.upsetPath,
    modelLimitation: nonEmpty(parsed?.modelLimitation) ?? fallback.modelLimitation
  };
}

function nonEmpty(value: unknown) {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function fallbackSections({
  keyTakeaway,
  prediction,
  selectedModel,
  squadProxy
}: {
  keyTakeaway: string;
  prediction: ExplainPrediction;
  selectedModel?: string;
  squadProxy?: { enabled?: boolean; status?: string; message?: string };
}): AnalystSections {
  const favorite = prediction?.favorite === "Toss-up" ? prediction?.teamA : prediction?.favorite;
  const riskTeam = favorite === prediction?.teamA ? prediction?.teamB : prediction?.teamA;
  return {
    keyTakeaway,
    whyFavorite:
      prediction?.favorite === "Toss-up"
        ? "The forecast is narrow, so match state and finishing variance carry extra weight."
        : `${favorite} is favored by the selected model's probabilities, scoreline, and key factors.`,
    riskFactor: `${riskTeam ?? "The opponent"} can keep this close if tempo drops or set-piece pressure changes the game state.`,
    upsetPath: "The underdog path is defensive control, transition chances, and forcing the favorite away from its normal scoring rhythm.",
    modelLimitation:
      selectedModel === "Experimental ML"
        ? "Experimental ML is an offline Logistic Regression baseline trained on historical World Cup features; probabilities are model-estimated, not betting odds."
        : selectedModel === "Elo + Score"
          ? "Elo + Score is an offline Elo-style expected-goals model using a Poisson score grid; probabilities are model-estimated, not betting odds."
        : squadProxy?.enabled || squadProxy?.status === "applied"
          ? "The modern squad proxy is manually curated sample data, not official squad data; probabilities are model-estimated, not betting odds."
          : "This is an explanatory brief for a local forecast; probabilities are model-estimated, not betting odds."
  };
}
