import OpenAI from "openai";
import { NextResponse } from "next/server";

export const runtime = "nodejs";

const model = process.env.OPENAI_MODEL ?? "gpt-5-nano";
const explanationCache = new Map<string, { text: string; createdAt: string }>();

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
  const key = stableKey({
    teamA: body?.teamA,
    teamB: body?.teamB,
    stage: body?.stage,
    probabilities: body?.probabilities,
    likelyScore: body?.likelyScore
  });

  const cached = explanationCache.get(key);
  if (cached) {
    return NextResponse.json({ text: cached.text, cached: true, model, usage: null });
  }

  const client = getOpenAIClient();
  if (!client) {
    return NextResponse.json(
      {
        text:
          "AI analysis is ready to enable. Add OPENAI_API_KEY in your local .env.local or Vercel project settings; the prediction model itself is already running locally without API spend.",
        cached: false,
        model,
        usage: null,
        missingKey: true
      },
      { status: 200 }
    );
  }

  const response = await client.responses.create({
    model,
    max_output_tokens: 260,
    input: [
      {
        role: "system",
        content:
          "You are a concise football analyst. Explain the prediction in 150-220 words, grounded only in the provided statistics. Do not mention betting advice."
      },
      {
        role: "user",
        content: JSON.stringify({
          teamA: body.teamA,
          teamB: body.teamB,
          stage: body.stage,
          probabilities: body.probabilities,
          likelyScore: body.likelyScore,
          confidence: body.confidence,
          favorite: body.favorite,
          factors: body.factors
        })
      }
    ]
  });

  const text = response.output_text.trim();
  explanationCache.set(key, { text, createdAt: new Date().toISOString() });

  return NextResponse.json({
    text,
    cached: false,
    model,
    usage: response.usage ?? null
  });
}
