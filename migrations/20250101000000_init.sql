-- Initial schema: canonical baseline for nostr-classify

CREATE TABLE IF NOT EXISTS profiles (
    pubkey TEXT PRIMARY KEY,
    nip05 TEXT,
    name TEXT,
    about TEXT,
    picture TEXT,
    is_classified BOOLEAN DEFAULT FALSE,
    follower_count INTEGER,
    metadata_json TEXT,
    metadata_created_at INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS events (
    id TEXT NOT NULL,
    pubkey TEXT NOT NULL,
    kind INTEGER NOT NULL,
    created_at INTEGER NOT NULL,
    raw_json TEXT NOT NULL,
    PRIMARY KEY (id, pubkey)
);

CREATE INDEX IF NOT EXISTS idx_events_pubkey_kind
    ON events(pubkey, kind, created_at);

CREATE TABLE IF NOT EXISTS classifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pubkey TEXT NOT NULL UNIQUE,
    labels TEXT NOT NULL,
    scores TEXT NOT NULL DEFAULT '{}',
    bio TEXT NOT NULL,
    confidence REAL,
    analyzed_event_count INTEGER,
    analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (pubkey) REFERENCES profiles(pubkey)
);

CREATE VIRTUAL TABLE IF NOT EXISTS classifications_fts USING fts5(
    name,
    about,
    nip05,
    labels,
    scores,
    bio,
    pubkey UNINDEXED
);

CREATE TABLE IF NOT EXISTS image_descriptions (
    hash TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
