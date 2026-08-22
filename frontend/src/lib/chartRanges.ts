export type QuickRange = "1M" | "3M" | "6M" | "1Y";

export const QUICK_RANGES: { label: QuickRange; days: number }[] = [
  { label: "1M", days: 30 },
  { label: "3M", days: 91 },
  { label: "6M", days: 182 },
  { label: "1Y", days: 365 },
];
