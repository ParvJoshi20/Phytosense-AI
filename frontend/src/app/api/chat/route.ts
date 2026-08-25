import { NextRequest, NextResponse } from 'next/server';
import { GoogleGenAI } from '@google/genai';

export async function POST(req: NextRequest) {
  try {
    const { messages, diagnosisContext } = await req.json();

    const apiKey = process.env.GEMINI_API_KEY;
    const contextPrompt = diagnosisContext
      ? `Current Patient/Specimen Context:
Disease: ${diagnosisContext.primaryDisease} (${diagnosisContext.scientificName})
Severity: ${diagnosisContext.severity} (${diagnosisContext.confidenceScore}% confidence)
Pathogen: ${diagnosisContext.pathogen?.commonName || 'N/A'} (Optimal temp: ${diagnosisContext.pathogen?.optimalTempRange || 'N/A'})
Canopy estimate: ${diagnosisContext.affectedCanopyEstimate || 'N/A'}
Key treatments: ${diagnosisContext.keyTreatments || 'Standard fungicides / cultural practices'}`
      : 'No specific leaf scanned yet.';

    if (apiKey) {
      try {
        const ai = new GoogleGenAI({
          apiKey,
          httpOptions: {
            headers: {
              'User-Agent': 'aistudio-build',
            },
          },
        });

        const systemInstruction = `You are the Phytosense AI Virtual Agronomist & Plant Pathology Specialist.
You provide precise, evidence-backed guidance on tomato crop protection, organic & chemical interventions, fungicide resistance management (FRAC rotations), dosage calculations, and weather adaptation.
Always explain the biological reasoning ("why") behind recommendations.
Maintain a professional, encouraging, and authoritative tone suitable for agronomists, commercial greenhouse managers, and smallholder farmers.

${contextPrompt}`;

        const lastMessage = messages[messages.length - 1]?.content || 'Hello';

        const response = await ai.models.generateContent({
          model: 'gemini-3.7-flash',
          contents: lastMessage,
          config: {
            systemInstruction,
          },
        });

        return NextResponse.json({
          reply: response.text || 'I could not generate an agronomist response. Please try again.',
        });
      } catch (err) {
        console.warn('Gemini chat error:', err);
      }
    }

    // Fallback rule-based agronomist responses
    const lastUserText = (messages[messages.length - 1]?.content || '').toLowerCase();
    let reply = `As the Phytosense Agronomist, based on the **${diagnosisContext?.primaryDisease || 'Tomato'}** diagnosis:

1. **Immediate Cultural Action**: Remove any symptomatic leaves showing >20% necrosis and sanitize your cutting shears with 70% alcohol.
2. **Moisture Control**: Ensure drip irrigation is active and foliage remains dry overnight to reduce fungal spore germination.
3. **Fungicide Rotation**: Alternate between multi-site protectants (like Copper or Mancozeb) and systemic options to prevent pathogen resistance.

Feel free to ask about specific dosages per acre, organic bio-pesticides, or spray schedules!`;

    if (lastUserText.includes('dosage') || lastUserText.includes('rate') || lastUserText.includes('how much')) {
      reply = `**Dosage & Application Protocol for ${diagnosisContext?.primaryDisease || 'Tomato Foliar Disease'}:**
- **Copper Octanoate (Organic)**: Mix 1.0 to 1.5 fl oz per gallon of water (approx 8-12 mL/L). Spray to the point of wetness on both upper and lower leaf surfaces.
- **Chlorothalonil (Protectant)**: 1.5 to 2.0 pints per acre (or 15 mL per 10 L water in knapsack sprayers).
- **Spray Timing**: Apply early in the morning (6-8 AM) when wind is low (<5 mph) and before high solar radiation to prevent leaf burn.`;
    } else if (lastUserText.includes('organic') || lastUserText.includes('natural') || lastUserText.includes('bio')) {
      reply = `**Organic Interventions (OMRI-Listed):**
1. **Bacillus subtilis (Serenade ASO)**: 2-4 quarts/acre. Inoculates the leaf surface with beneficial bacteria that produce lipopeptide toxins against fungal hyphae.
2. **Potassium Bicarbonate (MilStop)**: 2.5–5.0 lbs/acre. Rapidly shifts surface pH to inhibit spore germination without leaving toxic chemical residues.
3. **Pure Cold-Pressed Neem Oil**: 0.5% solution (1 tbsp/gal + 1 tsp castile soap) to deter insect vectors like whiteflies and suppress mild surface mildew.`;
    }

    return NextResponse.json({ reply });
  } catch (error: any) {
    console.error('Chat API error:', error);
    return NextResponse.json({ error: 'Failed to process chat message' }, { status: 500 });
  }
}
