import { useState } from "preact/hooks";
import type { Stats } from "../types";
import { LabelPill } from "./ProfileCard";
import { timeAgo } from "../types";

interface StatsBarProps {
  stats: Stats | null;
  onSearch?: (query: string) => void;
}

export function StatsBar({ stats }: StatsBarProps) {
  if (!stats) return null;
  return (
    <div class="stats-row">
      <div class="stat-card">
        <div class="stat-value">{stats.total_profiles.toLocaleString()}</div>
        <div class="stat-label">Profiles</div>
      </div>
      <div class="stat-card">
        <div class="stat-value green">{stats.classified_profiles.toLocaleString()}</div>
        <div class="stat-label">Classified</div>
      </div>
      <div class="stat-card">
        <div class="stat-value blue">{stats.total_events.toLocaleString()}</div>
        <div class="stat-label">Events</div>
      </div>
      <div class="stat-card">
        <div class="stat-value orange">{stats.images_classified.toLocaleString()}</div>
        <div class="stat-label">Images</div>
      </div>
      <div class="stat-card">
        <div class="stat-value orange">{stats.queue_size.toLocaleString()}</div>
        <div class="stat-label">Queue</div>
      </div>
      <div class="stat-card">
        <div class="stat-value blue">{stats.labels.total_unique_labels.toLocaleString()}</div>
        <div class="stat-label">Labels</div>
      </div>
    </div>
  );
}

export function LabelStatsPanel({ stats, onSearch }: StatsBarProps) {
  const [expanded, setExpanded] = useState(false);
  if (!stats || !stats.labels || stats.labels.label_counts.length === 0) return null;

  const maxCount = stats.labels.label_counts[0]?.count || 1;
  const showCount = expanded ? stats.labels.label_counts.length : 10;
  const visible = stats.labels.label_counts.slice(0, showCount);
  const hasMore = stats.labels.label_counts.length > 10;

  return (
    <div class="label-stats">
      <h3>Top Labels</h3>
      {visible.map(({ label, count }) => (
        <div
          class="label-bar-row label-bar-row-clickable"
          key={label}
          onClick={() => onSearch?.(label)}
          title={`Search for ${label}`}>
          <span class="label-bar-name">{label}</span>
          <div class="label-bar-track">
            <div
              class="label-bar-fill"
              style={{
                width: (count / maxCount) * 100 + "%",
                background: "#64b5f6",
              }}
            />
          </div>
          <span class="label-bar-count">{count}</span>
        </div>
      ))}
      {hasMore && (
        <button class="labels-toggle" onClick={() => setExpanded(!expanded)}>
          {expanded ? "Show less" : `Show all ${stats.labels.label_counts.length} labels`}
        </button>
      )}
    </div>
  );
}

interface RecentItemProps {
  item: {
    pubkey: string;
    name?: string | null;
    display_name?: string | null;
    picture?: string | null;
    scores: Record<string, number>;
    analyzed_at?: string | null;
  };
  onSearch: (pubkey: string) => void;
}

export function RecentItemCard({ item, onSearch }: RecentItemProps) {
  const displayName = item.display_name || item.name;
  const name = displayName || item.pubkey.slice(0, 12) + "...";
  return (
    <div class="card recent-item" onClick={(e) => { if (e.target === e.currentTarget) onSearch(item.pubkey); }}>
      <div class="recent-header">
        {item.picture ? (
          <img
            class="avatar avatar-small"
            src={item.picture}
            alt=""
            onError={(e) => ((e.target as HTMLImageElement).style.display = "none")}
          />
        ) : (
          <div class="avatar avatar-small" />
        )}
        <span class="recent-name">{name}</span>
        <span class="recent-time">{timeAgo(item.analyzed_at)}</span>
      </div>
      <div class="labels">
        {Object.entries(item.scores)
          .sort((a, b) => b[1] - a[1])
          .filter(([_, s]) => s >= 0.3)
          .slice(0, 5)
          .map(([l, score]) => (
            <LabelPill key={l} label={l} score={score} onClick={() => onSearch(l)} />
          ))}
      </div>
    </div>
  );
}
