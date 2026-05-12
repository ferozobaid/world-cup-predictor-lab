import OpenAI from "openai";
import { NextResponse } from "next/server";

export const runtime = "nodejs";

const model = process.env.OPENAI_MODEL ?? "gpt-5-nano";
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

  const response = await client.responses.create({
    model,
    max_output_tokens: 320,
    input: [
      {
        role: "system",
        content:
          [
            "You are a concise football analyst. OpenAI explains the forecast but does not make or override the prediction.",
            "Use only the supplied JSON context. Do not add external facts.",
            "Return only valid JSON with keys: keyTakeaway, whyFavorite, riskFactor, upsetPath, modelLimitation.",
            "Keep each value to one concise sentence. State probabilities are model-estimated, not betting odds.",
            "If selectedModel is Tournament ML, say it is an experimental offline Logistic Regression model trained on historical World Cup features.",
            "If squadProxy is enabled/applied, say it is a manually curated sample/proxy layer, not official squad data."
          ].join(" ")
      },
      {
        role: "user",
        content: JSON.stringify(analystContext)
      }
    ]
  });

  const rawText = response.output_text.trim();
  const sections = parseSections(rawText) ?? fallbackSections({
    keyTakeaway: rawText,
    prediction,
    selectedModel: body?.selectedModel,
    squadProxy: body?.squadProxy
  });
  const text = sections.keyTakeaway;
  explanationCache.set(key, { text, sections, createdAt: new Date().toISOString() });

  return NextResponse.json({
    text,
    sections,
    cached: false,
    model,
    usage: response.usage ?? null
  });
}

function parseSections(value: string): AnalystSections | null {
  try {
    const cleaned = value.replace(/^```json\s*/i, "").replace(/^```\s*/i, "").replace(/```$/i, "").trim();
    const parsed = JSON.parse(cleaned) as Partial<AnalystSections>;
    if (
      parsed.keyTakeaway &&
      parsed.whyFavorite &&
      parsed.riskFactor &&
      parsed.upsetPath &&
      parsed.modelLimitation
    ) {
      return {
        keyTakeaway: parsed.keyTakeaway,
        whyFavorite: parsed.whyFavorite,
        riskFactor: parsed.riskFactor,
        upsetPath: parsed.upsetPath,
        modelLimitation: parsed.modelLimitation
      };
    }
  } catch {
    return null;
  }
  return null;
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
      selectedModel === "Tournament ML"
        ? "Tournament ML is an experimental offline Logistic Regression model trained on historical World Cup features; probabilities are model-estimated, not betting odds."
        : squadProxy?.enabled || squadProxy?.status === "applied"
          ? "The modern squad proxy is manually curated sample data, not official squad data; probabilities are model-estimated, not betting odds."
          : "This is an explanatory brief for a local forecast; probabilities are model-estimated, not betting odds."
  };
}
