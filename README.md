# nostr-classify

**LLM-powered Nostr profile classification.** Subscribes to relays, collects user activity, and uses a tool-calling LLM to analyze profiles — producing classification labels, a generated bio, and a confidence score. Results are stored in SQLite with FTS5 search and exposed via a web dashboard, REST API, and Nostr search relay.

## How It Works

### Pipeline

```
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│ Collect  │ ──▶ │  Filter  │ ──▶ │ Classify │ ──▶ │ Cleanup  │
└──────────┘     └──────────┘     └──────────┘     └──────────┘
```

1. **Collect** — Subscribes to Nostr relays via websocket and caches classifiable events (posts, reactions, reposts, zaps, comments, long-form content, pictures, videos, highlights) in a local SQLite database.
2. **Filter** — Profiles are only queued for classification when they meet two thresholds: enough classifiable events **and** a minimum follower count (filters out bots and test accounts). Follower counts are fetched via kind-3 contact list queries.
3. **Classify** — An LLM (any OpenAI-compatible API) analyzes the profile using **tool calls** to gather rich context:
   - Fetches additional events by kind on demand
   - Resolves NIP-21 `nostr:` URIs to profile/event details
   - Downloads and describes images and videos (frame collage → vision model)
   - Fetches OpenGraph metadata from shared URLs
   - Looks up referenced profiles

   The model outputs a structured JSON response with labels, scores, a bio, and a confidence score.

4. **Cleanup** — Old events are pruned after `cache_days` to keep the database small while preserving classification results.

### Classification Epoch

A `CLASSIFICATION_EPOCH` constant (currently `5`) is incremented whenever the system prompt, taxonomy, or classification logic changes enough to warrant re-processing. All profiles with a stale epoch are automatically re-enqueued on startup.

### Classified Event Kinds

| Kind  | Description               | NIP |
| ----- | ------------------------- | --- |
| 30580 | Metadata                  | 01  |
| 1     | Short Text Note           | 10  |
| 6     | Repost                    | 18  |
| 7     | Reaction                  | 25  |
| 16    | Generic Repost            | 18  |
| 17    | Reaction to a website     | 25  |
| 20    | Picture                   | 68  |
| 21    | Video Event               | 71  |
| 22    | Short-form Portrait Video | 71  |
| 1111  | Comment                   | 22  |
| 9735  | Zap Receipt               | 57  |
| 9802  | Highlights                | 84  |
| 30023 | Long-form Content         | 23  |
| 34235 | Addressable Normal Video  | 71  |
| 34236 | Addressable Short Video   | 71  |

Zap receipts are dual-indexed: credited to both the recipient and the verified sender, so zappers also get event count credit.

## Architecture

```
nostr-classify/
├── src/
│   ├── main.rs              # Entry point — orchestrates collector, processor, server
│   ├── config.rs            # YAML config parsing + built-in label taxonomy (~120 labels)
│   ├── classifier.rs        # LLM classification with tool-calling loop (up to 15 iterations)
│   ├── nostr_client.rs      # Nostr SDK wrapper — relay connections, event/follower fetching
│   ├── nostr_collector.rs   # Batched event ingestion from relay subscriptions
│   ├── db.rs                # SQLite via sqlx — profiles, events, classifications, FTS5
│   ├── job_queue.rs         # Bounded async job queue with retry backoff + rate limiting
│   ├── http_server.rs       # Axum HTTP server — dashboard, REST API, WebSocket upgrade
│   ├── search_relay.rs      # Nostr Discovery relay backed by FTS5 (NIP-50 search on kind 0)
│   ├── image_cache.rs       # Downloads/describes images and videos with dedup + caching
│   ├── video.rs             # Video frame extraction → collage for vision model analysis
│   ├── opengraph.rs         # OpenGraph metadata scraping from shared URLs
│   ├── profile_cache.rs     # TTL cache for profile lookups (7-day expiry)
│   ├── count_cache.rs       # In-memory event count + follower count caches
│   └── format.rs            # Human-readable event/profile formatting for LLM context
├── dashboard/               # Preact + Vite SPA dashboard
├── migrations/              # SQLite schema migrations
├── config.yaml              # Configuration file
└── Dockerfile               # Multi-stage build (Bun + Rust + Debian slim)
```

### LLM Tool Calls

The LLM is given these tools to gather context before classifying:

| Tool                 | Purpose                                                                      |
| -------------------- | ---------------------------------------------------------------------------- |
| `get_event`          | Fetch a specific Nostr event by ID (from local cache or relay fallback)      |
| `get_profile`        | Fetch profile metadata for any pubkey (cache with relay fallback)            |
| `get_profile_events` | Fetch additional events filtered by kind + time range                        |
| `describe_image`     | Download and describe image/video content via vision model                   |
| `resolve_nip21`      | Resolve `nostr:npub1...`, `nostr:nevent1...`, `nostr:naddr1...` URIs         |
| `get_opengraph`      | Scrape OpenGraph metadata (title, description, site, image) from shared URLs |

### Search Relay

The built-in Nostr search relay (NIP-50) runs on the same port as the HTTP server via WebSocket upgrade. It exposes FTS5 search over classifications — Nostr clients can query by keyword and get kind-0 metadata events for matching profiles.

## Setup

### Prerequisites

- **Rust** (stable 1.80+)
- **Bun** (for dashboard build)
- **ffmpeg** runtime libraries (for video frame extraction)
- An **OpenAI-compatible LLM API** (Ollama, vLLM, LiteLLM, OpenAI, etc.) with vision support

### Configuration

Copy and edit `config.yaml`:

```yaml
llm:
  api_base_url: "http://localhost:8001/v1"
  model: "qwen3.5:122b"
  api_key: ""
  timeout_secs: 120 # per-request HTTP timeout
  classify_timeout_secs: 300 # overall timeout for a single classification

nostr:
  nsec: "" # optional for signed requests
  relays:
    - "wss://relay.damus.io"
    - "wss://nos.lol"
    - "wss://relay.primal.net"

processing:
  event_threshold: 20 # min events before classification
  classification_event_limit: 50 # max events fed to the LLM
  min_followers: 3 # min followers to filter bots
  cache_days: 7 # prune events older than this
  max_workers: 4 # concurrent classification workers
  max_retries: 3 # retry on server errors
  image_download_timeout_secs: 30
  job_timeout_secs: 600 # overall timeout per classification job
  tool_call_timeout_secs: 30 # timeout for individual LLM tool calls

database:
  path: "nostr_classify.db"

image_cache:
  dir: "/tmp/nostr-classify-images"
  cleanup_days: 1

logging:
  level: "info"

labels:
  taxonomy_file: null # path to custom label list, or null for built-in (~120 labels)
  min_score: 0.4 # minimum score for a label to be included
```

### Run

```bash
cargo run
```

The dashboard is available at **`http://localhost:3000`**. The Nostr search relay is at **`ws://localhost:3000/`** on the same port.

### Docker

```bash
docker build -t nostr-classify .
docker run -v $(pwd)/config.yaml:/app/config.yaml -p 3000:3000 nostr-classify
```

## API

| Endpoint                                       | Description                                                                  |
| ---------------------------------------------- | ---------------------------------------------------------------------------- |
| `GET /`                                        | Web dashboard (SPA)                                                          |
| `GET /api/profile/{pubkey}`                    | Profile details + classification status/scores                               |
| `GET /api/recent?limit=20`                     | Recently classified profiles                                                 |
| `GET /api/search?q=bitcoin&limit=20`           | FTS5 search across labels, bios, names, NIP-05                               |
| `GET /api/search/label?label=bitcoin&limit=20` | Exact label match search                                                     |
| `GET /api/stats`                               | Totals: profiles, classifications, events, images, queue, label distribution |

### Profile Response Example

```json
{
  "pubkey": "abc123...",
  "name": "alice",
  "display_name": "Alice",
  "about": "Bitcoin dev, Rust enthusiast",
  "picture": "https://...",
  "nip05": "alice@nostrplebs.com",
  "event_count": 142,
  "classification_status": "current",
  "classification": {
    "scores": { "bitcoin": 0.95, "rust": 0.88, "software-developer": 0.82 },
    "bio": "Alice is a Bitcoin developer...",
    "confidence": 0.85,
    "analyzed_at": "2025-05-11T10:00:00Z",
    "analyzed_event_count": 50,
    "kind_breakdown": [
      { "kind": 1, "name": "Short Text Note", "count": 30 },
      { "kind": 9735, "name": "Zap Receipt", "count": 15 },
      { "kind": 7, "name": "Reaction", "count": 5 }
    ]
  }
}
```

## Built-in Label Taxonomy

~120 labels covering technology, Bitcoin/Nostr, privacy, content creation, professions, lifestyle, politics, languages, and content quality:

**Technology:** `rust`, `python`, `javascript`, `golang`, `linux`, `self-hosting`, `ai-ml`, `cybersecurity`, `embedded-systems`, `open-source`, `game-development`…

**Bitcoin & Nostr:** `bitcoin`, `bitcoin-mining`, `lightning-network`, `nostr-developer`, `nostr-enthusiast`, `altcoin`, `defi`, `trading`, `nft`…

**Content Creation:** `writer`, `podcaster`, `musician`, `artist`, `photographer`, `video-creator`, `memer`…

**Lifestyle:** `gaming`, `fitness`, `food`, `coffee`, `travel`, `hiking`, `yoga`, `meditation`, `homesteading`, `gardening`…

**Politics:** `libertarian`, `anarchist`, `politics`, `activism`, `anti-authoritarian`…

**Content Quality:** `nsfw`, `bot`, `spam`, `troll`, `scam`

**Languages:** `english`, `japanese`, `german`, `spanish`, `portuguese`, `russian`…

Custom taxonomies can be provided via `labels.taxonomy_file` (one label per line, `#` comments supported).

## License

MIT
