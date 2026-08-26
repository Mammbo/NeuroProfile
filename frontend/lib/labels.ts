// Friendly labels for the 6 ICA systems (raw payload names are snake_case).
const NAMES: Record<string, string> = {
  audiovisual_integration: "Audiovisual integration",
  social_sts_tpj: "Social · STS/TPJ",
  visual_motion: "Visual motion",
  auditory: "Auditory",
  dmn_scene_medial_parietal: "Default mode",
  affect_reward: "Affect / reward",
};

export function prettyName(raw: string): string {
  if (NAMES[raw]) return NAMES[raw];
  const s = raw.replace(/_/g, " ");
  return s.charAt(0).toUpperCase() + s.slice(1);
}

export function tierAbbrev(t?: string): string {
  const map: Record<string, string> = { high: "hi", medium: "md", low: "lo" };
  return map[t || ""] || (t || "").slice(0, 2);
}
