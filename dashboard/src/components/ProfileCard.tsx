import type { Classification, KindBreakdown } from "../types";

export function LabelPill({ label, score }: { label: string; score: number }) {
  const color =
    score >= 0.8 ? "#66bb6a" : score >= 0.6 ? "#64b5f6" : score >= 0.4 ? "#ffa726" : "#ef5350";

  return (
    <span class="label">
      <span class="label-bg" style={{ background: color }} />
      <span class="label-text">{label}</span>
      <span class="label-score">{(score * 100).toFixed(0)}%</span>
    </span>
  );
}

export function KindBreakdownView({
  breakdown,
  total,
}: {
  breakdown: KindBreakdown[];
  total: number;
}) {
  if (!breakdown || breakdown.length === 0) return null;
  const maxCount = breakdown[0]?.count || 1;
  return (
    <div class="kind-breakdown">
      <h3
        style={{
          color: "#fff",
          fontSize: "14px",
          marginBottom: "10px",
          textTransform: "uppercase",
          letterSpacing: "0.5px",
        }}
      >
        Event Kinds
      </h3>
      {breakdown.map(({ kind, name, count }) => (
        <div class="kind-bar-row" key={kind}>
          <span class="kind-bar-name" title={`Kind ${kind}`}>
            {name}
          </span>
          <div class="kind-bar-track">
            <div
              class="kind-bar-fill"
              style={{ width: (count / maxCount) * 100 + "%" }}
            />
          </div>
          <span class="kind-bar-count">{count}</span>
          <span class="kind-bar-pct">{((count / total) * 100).toFixed(0)}%</span>
        </div>
      ))}
    </div>
  );
}

interface ProfileCardProps {
  profile: {
    name?: string | null;
    picture?: string | null;
    nip05?: string | null;
    pubkey: string;
    about?: string | null;
    event_count: number;
    classification_status:
      | "none"
      | "current"
      | { stale: { epoch: number } };
    classification?: Classification | null;
  };
  onSearch: (pubkey: string) => void;
}

export function ProfileCard({ profile }: ProfileCardProps) {
  const { name, picture, nip05, pubkey, about, event_count, classification_status, classification } =
    profile;

  const isStale =
    typeof classification_status === "object" && classification_status !== null && "stale" in classification_status;
  const statusLabel =
    classification_status === "current"
      ? "Classified"
      : isStale
        ? `Stale (epoch ${(classification_status as { stale: { epoch: number } }).stale.epoch})`
        : "Not classified";
  const statusColor =
    classification_status === "current" ? "#66bb6a" : isStale ? "#ffa726" : "#ef5350";
  const hasClassification = classification_status !== "none";

  return (
    <div class="card">
      <div class="card-header">
        {picture && (
          <img
            class="avatar"
            src={picture}
            alt=""
            onError={(e) => ((e.target as HTMLImageElement).style.display = "none")}
          />
        )}
        <div class="author-info">
          <span class="author-name">{name || "Unknown"}</span>
          {nip05 && <span class="nip05">{nip05}</span>}
          <span class="pubkey">{pubkey}</span>
        </div>
      </div>
      {about && <div class="about">{about}</div>}
      <div class="meta">
        <span>{event_count} events</span>
        <span style={{ color: statusColor }}>{statusLabel}</span>
      </div>
      {hasClassification &&
        classification &&
        classification.kind_breakdown &&
        classification.kind_breakdown.length > 0 && (
          <KindBreakdownView
            breakdown={classification.kind_breakdown}
            total={classification.analyzed_event_count}
          />
        )}
      <div>
        <h3
          style={{
            color: "#fff",
            fontSize: "14px",
            marginBottom: "10px",
            textTransform: "uppercase",
            letterSpacing: "0.5px",
          }}
        >
          Classification
        </h3>
        {hasClassification && classification ? (
          <>
            <div class="labels">
              {Object.entries(classification.scores)
                .sort((a, b) => b[1] - a[1])
                .filter(([_, s]) => s >= 0.3)
                .map(([l, score]) => (
                  <LabelPill key={l} label={l} score={score} />
                ))}
            </div>
            <p class="bio">{classification.bio}</p>
            <div class="confidence">
              Confidence: {(classification.confidence * 100).toFixed(0)}%
              <div class="confidence-bar">
                <div
                  class="confidence-fill"
                  style={{
                    width: classification.confidence * 100 + "%",
                    background:
                      classification.confidence >= 0.8
                        ? "#66bb6a"
                        : classification.confidence >= 0.6
                          ? "#ffa726"
                          : "#ef5350",
                  }}
                />
              </div>
            </div>
          </>
        ) : (
          <p class="no-classification">No classification available yet</p>
        )}
      </div>
    </div>
  );
}
