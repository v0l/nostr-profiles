export function timeAgo(dateStr?: string | null): string {
  if (!dateStr) return "";
  const diff = Date.now() - new Date(dateStr).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return mins + "m ago";
  const hours = Math.floor(mins / 60);
  if (hours < 24) return hours + "h ago";
  return Math.floor(hours / 24) + "d ago";
}

export function scoreColor(score: number): string {
  if (score >= 0.8) return "#66bb6a";
  if (score >= 0.6) return "#64b5f6";
  if (score >= 0.4) return "#ffa726";
  return "#ef5350";
}

export interface Classification {
  scores: Record<string, number>;
  bio: string;
  confidence: number;
  analyzed_at?: string | null;
  analyzed_event_count: number;
  kind_breakdown: KindBreakdown[];
}

export interface KindBreakdown {
  kind: number;
  name: string;
  count: number;
}

export interface Profile {
  pubkey: string;
  name?: string | null;
  about?: string | null;
  picture?: string | null;
  nip05?: string | null;
  event_count: number;
  classification_status:
    | "none"
    | "current"
    | { stale: { epoch: number } };
  classification?: Classification | null;
}

export interface RecentItem {
  pubkey: string;
  name?: string | null;
  picture?: string | null;
  scores: Record<string, number>;
  bio: string;
  confidence: number;
  analyzed_at?: string | null;
  metadata_json?: string | null;
}

export interface Stats {
  total_profiles: number;
  classified_profiles: number;
  total_events: number;
  images_classified: number;
  queue_size: number;
  labels: {
    total_unique_labels: number;
    label_counts: { label: string; count: number }[];
  };
}
