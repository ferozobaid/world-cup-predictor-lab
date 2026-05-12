import { PredictorLab } from "@/components/predictor-lab";
import { worldCupData } from "@/lib/world-cup-model";

export default function Home() {
  return (
    <main>
      <PredictorLab
        teams={worldCupData.teams}
        fixtures={worldCupData.fixtures2026}
        generatedAt={worldCupData.generatedAt}
        sources={worldCupData.sources}
      />
    </main>
  );
}
