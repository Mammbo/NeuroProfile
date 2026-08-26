// Muted two-tone diverging scale: below-avg = soft slate, above-avg = soft clay,
// midpoint = warm paper. The only colour anywhere in the UI lives here.
export function divColor(v: number): string {
  const t = Math.min(1, Math.abs(v));
  const paper = [251, 251, 249];
  const warm = [178, 120, 90]; // above average (muted clay)
  const cool = [108, 131, 145]; // below average (muted slate)
  const a = v >= 0 ? warm : cool;
  const m = (i: number) => Math.round(paper[i] + (a[i] - paper[i]) * t);
  return `rgb(${m(0)},${m(1)},${m(2)})`;
}

export function fmtTime(t: number): string {
  const s = Math.max(0, t | 0);
  return `${(s / 60) | 0}:${String(s % 60).padStart(2, "0")}`;
}
