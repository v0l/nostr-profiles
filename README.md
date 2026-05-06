# nostr-classify

Classifies Nostr profiles based on their activity using LLMs.

Subscribes to Nostr relays, collects user events, and generates classification labels and bios using an OpenAI-compatible LLM API. Results are stored in SQLite and exposed via a web dashboard and REST API.

## How it works

1. **Collect** — Subscribes to relays and caches classifiable events (posts, reactions, reposts, zaps, comments, long-form content, etc.)
2. **Filter** — Profiles are only queued for classification when they have enough events *and* meet a minimum follower threshold (filters out bots and test accounts)
3. **Classify** — An LLM analyzes the profile's metadata, events, and image descriptions to produce labels, a bio, and a confidence score
4. **Cleanup** — Old events are pruned after classification to keep the database small

### Classified event kinds

Only user-generated event kinds are used for classification:

| Kind | Description | NIP |
|------|-------------|-----|
| 0 | Metadata | 01 |
| 1 | Short Text Note | 10 |
| 6 | Repost | 18 |
| 7 | Reaction | 25 |
| 16 | Generic Repost | 18 |
| 17 | Reaction to a website | 25 |
| 20 | Picture | 68 |
| 21 | Video Event | 71 |
| 22 | Short-form Portrait Video | 71 |
| 1111 | Comment | 22 |
| 9735 | Zap Receipt | 57 |
| 9802 | Highlights | 84 |
| 30023 | Long-form Content | 23 |

## Setup

### Configuration

Copy `config.yaml` and edit:

```yaml
llm:
  api_base_url: "http://localhost:8001/v1"
  model: "qwen3.5:122b"
  api_key: ""

nostr:
  nsec: ""  # optional
  relays:
    - "wss://relay.damus.io"
    - "wss://nos.lol"
    - "wss://relay.primal.net"

processing:
  event_threshold: 20     # min events before classification
  min_followers: 1        # min followers to filter bots
  cache_days: 7           # prune events older than this after classification
  max_workers: 4
  max_retries: 3
  image_download_timeout_secs: 30

database:
  path: "nostr_classify.db"

image_cache:
  dir: "/tmp/nostr-classify-images"
  cleanup_days: 1

logging:
  level: "info"
```

### Run

```bash
cargo run
```

Dashboard available at `http://localhost:3000`.

### Docker

```bash
docker build -t nostr-classify .
docker run -v $(pwd)/config.yaml:/app/config.yaml -p 3000:3000 nostr-classify
```

## API

| Endpoint | Description |
|----------|-------------|
| `GET /` | Web dashboard |
| `GET /api/profile/{pubkey}` | Profile details and classification |
| `GET /api/recent?limit=20` | Recent classifications |
| `GET /api/search?q=bitcoin&limit=20` | Full-text search across labels and bios |

## License

MIT
