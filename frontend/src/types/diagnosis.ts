export type SeverityTier = 'Healthy' | 'Mild' | 'Moderate' | 'Severe';

export interface PathogenInfo {
  scientificName: string;
  commonName: string;
  type: 'Fungal' | 'Bacterial' | 'Viral' | 'Pest' | 'Physiological' | 'None';
  family?: string;
  optimalTempRange: string;
  humidityRiskThreshold: string;
}

export interface DiseaseProbability {
  disease: string;
  scientificName: string;
  probability: number; // 0 - 100
  isTopPrediction?: boolean;
}

export interface MorphologicalMarker {
  id: string;
  name: string;
  description: string;
  significance: 'High' | 'Medium' | 'Low';
  x: number; // percentage 0-100
  y: number; // percentage 0-100
  radius: number; // percentage
  featureType: 'concentric_rings' | 'chlorotic_halo' | 'necrotic_lesion' | 'marginal_curling' | 'vein_clearing' | 'pycnidia_specks' | 'water_soaked';
}

export interface HeatmapConfig {
  colormap: 'jet' | 'turbo' | 'inferno' | 'viridis';
  opacity: number; // 0 to 1
  showMarkers: boolean;
  showContours: boolean;
  threshold: number; // 0 to 1
}

export interface CounterfactualComparison {
  alternativeDisease: string;
  scientificName: string;
  probability: number;
  whyNotReason: string;
  missingMarker: string;
  differentiatingFeature: string;
}

export interface TreatmentAction {
  id: string;
  title: string;
  description: string;
  timing: string;
  dosage?: string;
  safetyInterval?: string; // Pre-Harvest Interval (PHI)
  costLevel: 'Low' | 'Medium' | 'High';
  urgency: 'Critical' | 'High' | 'Moderate' | 'Preventative';
  completed?: boolean;
}

export interface DecisionPlan {
  immediateContainment: TreatmentAction[];
  organicBiological: TreatmentAction[];
  targetedChemical: TreatmentAction[];
  environmentalAdjustments: string[];
  monitoringSchedule: string;
  prognosis: string;
}

export interface Layer1DetectionData {
  primaryDisease: string;
  scientificName: string;
  severity: SeverityTier;
  confidenceScore: number; // 0 to 100
  probabilities: DiseaseProbability[];
  pathogen: PathogenInfo;
  affectedCanopyEstimate: string; // e.g. "15-25% lower foliage"
}

export interface Layer2VisualExplainabilityData {
  modelArchitecture: string; // e.g. "EfficientNetV2-M + Grad-CAM++"
  targetLayer: string; // e.g. "features.7.2.conv2"
  dominantFocusArea: string;
  peakActivationScore: number; // 0.0 - 1.0
  markers: MorphologicalMarker[];
  gradCamPoints: { x: number; y: number; intensity: number; radius: number }[];
  heatmapDescription: string;
}

export interface Layer3ReasoningData {
  whyThisPrediction: {
    primarySymptom: string;
    latentFeatureMapping: string;
    morphologicalEvidence: string[];
    canopyPattern: string;
  };
  counterfactuals: CounterfactualComparison[];
  confidenceAssessment: string;
}

export interface Layer4DecisionSupportData {
  actionPlan: DecisionPlan;
  economicRisk: 'Minimal' | 'Moderate' | 'Severe Yield Loss Warning';
  recommendedReapplicationDays: number;
  weatherRiskWarning?: string;
}

export interface DiagnosticResult {
  id: string;
  timestamp: number;
  imageUrl: string;
  imageName?: string;
  layer1: Layer1DetectionData;
  layer2: Layer2VisualExplainabilityData;
  layer3: Layer3ReasoningData;
  layer4: Layer4DecisionSupportData;
  source: 'ai_inference' | 'benchmark_sample';
  processingTimeMs: number;
}

export interface BenchmarkSample {
  id: string;
  title: string;
  diseaseCategory: string;
  severity: SeverityTier;
  thumbnail: string;
  leafSvgPath?: string;
  description: string;
  precomputedResult: DiagnosticResult;
}
