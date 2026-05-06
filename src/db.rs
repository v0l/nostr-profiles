use anyhow::Result;
use nostr_sdk::JsonUtil;
use serde::{Deserialize, Serialize};
use sqlx::{FromRow, SqlitePool};

/// Kinds that are useful for classifying a profile.
/// We only count and fetch these when determining if someone should be classified.
///
/// Based on NIPs: https://github.com/nostr-protocol/nips
///
/// - 0:      Metadata (profile info) — NIP-01
/// - 1:      Short Text Note — NIP-10
/// - 6:      Repost — NIP-18
/// - 7:      Reaction — NIP-25
/// - 16:     Generic Repost — NIP-18
/// - 17:     Reaction to a website — NIP-25
/// - 20:     Picture — NIP-68
/// - 21:     Video Event — NIP-71
/// - 22:     Short-form Portrait Video — NIP-71
/// - 1111:   Comment — NIP-22
/// - 9735:   Zap Receipt — NIP-57
/// - 9802:   Highlights — NIP-84
/// - 30023:  Long-form Content — NIP-23
pub const CLASSIFIABLE_KINDS: &[u16] = &[
    0,      // Metadata
    1,      // Short Text Note
    6,      // Repost
    7,      // Reaction
    16,     // Generic Repost
    17,     // Reaction to a website
    20,     // Picture
    21,     // Video Event
    22,     // Short-form Portrait Video
    1111,   // Comment
    9735,   // Zap Receipt
    9802,   // Highlights
    30023,  // Long-form Content
];

/// Check if a nostr event kind is relevant for profile classification.
pub fn is_classifiable_kind(kind: u16) -> bool {
    CLASSIFIABLE_KINDS.contains(&kind)
}

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Profile {
    pub pubkey: String,
    pub nip05: Option<String>,
    pub name: Option<String>,
    pub about: Option<String>,
    pub picture: Option<String>,
    pub is_classified: bool,
    pub needs_processing: bool,
    pub follower_count: Option<i64>,
    pub metadata_json: Option<String>,
    pub metadata_created_at: Option<i64>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Event {
    pub id: String,
    pub pubkey: String,
    pub kind: i64,
    pub created_at: i64,
    pub raw_json: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Classification {
    pub labels: Vec<String>,
    pub scores: std::collections::HashMap<String, f64>,
    pub bio: String,
    pub confidence: f64,
}

#[derive(Clone, Debug)]
pub struct Database {
    pub pool: SqlitePool,
}

impl Database {
    pub async fn new(path: &str) -> Result<Self> {
        // Create database file with proper permissions if it doesn't exist
        use std::fs;
        use std::os::unix::fs::PermissionsExt;
        
        let path = std::path::Path::new(path);
        if !path.exists() {
            // Create parent directories if needed
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            // Create file with read/write permissions for owner only
            fs::OpenOptions::new()
                .create(true)
                .write(true)
                .open(path)?;
            // Set permissions to 0600 (rw-------)
            fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
        }
        
        let pool = SqlitePool::connect(path.to_str().unwrap()).await?;
        Self::migrate(&pool).await?;
        Ok(Self { pool })
    }

    async fn migrate(pool: &SqlitePool) -> Result<()> {
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS profiles (
                pubkey TEXT PRIMARY KEY,
                nip05 TEXT,
                name TEXT,
                about TEXT,
                picture TEXT,
                is_classified BOOLEAN DEFAULT FALSE,
                needs_processing BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            "#,
        )
        .execute(pool)
        .await?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS events (
                id TEXT NOT NULL,
                pubkey TEXT NOT NULL,
                kind INTEGER NOT NULL,
                created_at INTEGER NOT NULL,
                raw_json TEXT NOT NULL,
                PRIMARY KEY (id, pubkey)
            )
            "#,
        )
        .execute(pool)
        .await?;

        sqlx::query(
            r#"
            CREATE INDEX IF NOT EXISTS idx_events_pubkey_kind 
            ON events(pubkey, kind, created_at)
            "#,
        )
        .execute(pool)
        .await?;

        sqlx::query(
            r#"
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
            )
            "#,
        )
        .execute(pool)
        .await?;

        sqlx::query(
            r#"
            CREATE VIRTUAL TABLE IF NOT EXISTS classifications_fts USING fts5(
                name,
                about,
                nip05,
                labels,
                scores,
                bio,
                pubkey UNINDEXED
            )
            "#,
        )
        .execute(pool)
        .await?;

        // Migrate: recreate FTS table if it uses the old schema
        // Old schemas: (1) had 'pubkey' as indexed column, (2) used contentless fts5 (content=''),
        // (3) missing 'scores' column
        // FTS5 tables can't be ALTERed, so drop and rebuild
        let needs_migration: bool = {
            // Check for old 'pubkey' indexed column
            let has_indexed_pubkey: bool = sqlx::query_scalar::<_, i64>(
                r#"SELECT COUNT(*) FROM pragma_table_info('classifications_fts') WHERE name = 'pubkey'"#,
            )
            .fetch_one(pool)
            .await?
            > 0;

            // Check for missing 'scores' column
            let has_scores: bool = sqlx::query_scalar::<_, i64>(
                r#"SELECT COUNT(*) FROM pragma_table_info('classifications_fts') WHERE name = 'scores'"#,
            )
            .fetch_one(pool)
            .await?
            > 0;

            has_indexed_pubkey || !has_scores
        };

        if needs_migration {
            sqlx::query(r#"DROP TABLE classifications_fts"#)
                .execute(pool)
                .await?;
            sqlx::query(
                r#"
                CREATE VIRTUAL TABLE classifications_fts USING fts5(
                    name,
                    about,
                    nip05,
                    labels,
                    scores,
                    bio,
                    pubkey UNINDEXED
                )
                "#,
            )
            .execute(pool)
            .await?;

            // Rebuild FTS index from existing classified profiles
            sqlx::query(
                r#"
                INSERT INTO classifications_fts (rowid, name, about, nip05, labels, scores, bio, pubkey)
                SELECT c.id, p.name, p.about, p.nip05,
                       REPLACE(REPLACE(SUBSTR(c.labels, 2, LENGTH(c.labels) - 2), '"', ''), ',', ' '),
                       c.scores,
                       c.bio, c.pubkey
                FROM classifications c
                JOIN profiles p ON c.pubkey = p.pubkey
                "#,
            )
            .execute(pool)
            .await?;
        }

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS image_descriptions (
                hash TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            "#,
        )
        .execute(pool)
        .await?;

        // Migrate: add follower_count column if missing
        let _ = sqlx::query(
            r#"ALTER TABLE profiles ADD COLUMN follower_count INTEGER"#,
        )
        .execute(pool)
        .await;

        // Migrate: add scores column if missing
        let _ = sqlx::query(
            r#"ALTER TABLE classifications ADD COLUMN scores TEXT NOT NULL DEFAULT '{}'"#,
        )
        .execute(pool)
        .await;

        // Migrate: add metadata_json column if missing (stores raw kind 0 event for relay)
        let _ = sqlx::query(
            r#"ALTER TABLE profiles ADD COLUMN metadata_json TEXT"#,
        )
        .execute(pool)
        .await;

        // Migrate: add metadata_created_at column if missing (nostr event timestamp for kind 0)
        let _ = sqlx::query(
            r#"ALTER TABLE profiles ADD COLUMN metadata_created_at INTEGER"#,
        )
        .execute(pool)
        .await;

        // Migrate: events table composite PK (id, pubkey) to allow same event under multiple pubkeys
        // (needed for zap receipts attributed to both LNURL server and sender)
        let needs_events_migration: bool = {
            // Check if PK is just 'id' (old) vs composite (new)
            let pk_cols: Vec<String> = sqlx::query_as::<_, (String,)>(
                r#"SELECT name FROM pragma_table_info('events') WHERE pk > 0 ORDER BY pk"#,
            )
            .fetch_all(pool)
            .await
            .map(|rows| rows.into_iter().map(|(n,)| n).collect())
            .unwrap_or_default();

            // Old schema: only 'id' is PK. New schema: both 'id' and 'pubkey' are PK.
            !pk_cols.contains(&"pubkey".to_string())
        };

        if needs_events_migration {
            tracing::info!("Migrating events table to composite primary key (id, pubkey)...");
            sqlx::query(r#"CREATE TABLE events_new (
                id TEXT NOT NULL,
                pubkey TEXT NOT NULL,
                kind INTEGER NOT NULL,
                created_at INTEGER NOT NULL,
                raw_json TEXT NOT NULL,
                PRIMARY KEY (id, pubkey)
            )"#)
            .execute(pool)
            .await?;

            sqlx::query(r#"INSERT OR IGNORE INTO events_new SELECT * FROM events"#)
                .execute(pool)
                .await?;

            sqlx::query(r#"DROP TABLE events"#)
                .execute(pool)
                .await?;

            sqlx::query(r#"ALTER TABLE events_new RENAME TO events"#)
                .execute(pool)
                .await?;

            sqlx::query(
                r#"CREATE INDEX IF NOT EXISTS idx_events_pubkey_kind 
                ON events(pubkey, kind, created_at)"#,
            )
            .execute(pool)
            .await?;

            tracing::info!("Events table migration complete");
        }

        Ok(())
    }

    pub async fn upsert_profile(&self, pubkey: &str) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO profiles (pubkey, needs_processing)
            VALUES (?, FALSE)
            ON CONFLICT(pubkey) DO UPDATE SET
                updated_at = CURRENT_TIMESTAMP
            "#,
        )
        .bind(pubkey)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn cache_event(&self, event: &nostr_sdk::Event) -> Result<()> {
        let event_id = event.id.to_hex();
        let pubkey = event.pubkey.to_hex();
        let kind = (event.kind.as_u16() as u32) as i64;
        let created_at = event.created_at.as_secs() as i64;
        let raw_json = serde_json::to_string(event)?;
        
        // Ensure profile exists first (for foreign key constraint)
        self.upsert_profile(&pubkey).await?;

        // Kind 0 = metadata — extract profile fields and update
        if kind == 0 {
            self.update_profile_metadata(&pubkey, &event.content, &raw_json, created_at).await?;
        }

        sqlx::query(
            r#"
            INSERT INTO events (id, pubkey, kind, created_at, raw_json)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id, pubkey) DO NOTHING
            "#,
        )
        .bind(&event_id)
        .bind(&pubkey)
        .bind(kind)
        .bind(created_at)
        .bind(&raw_json)
        .execute(&self.pool)
        .await?;

        // For zap receipts (kind 9735), also index under the sender's pubkey.
        // Parse the description tag which contains the full 9734 zap request.
        // Verify the signature to ensure the sender pubkey is authentic.
        if kind == 9735 {
            if let Some(zap_request_json) = event.tags.iter()
                .filter_map(|t| match t.as_standardized() {
                    Some(nostr_sdk::TagStandard::Description(desc)) => Some(desc.as_str()),
                    _ => None,
                })
                .next()
            {
                if let Ok(zap_request) = nostr_sdk::Event::from_json(zap_request_json) {
                    if zap_request.verify().is_ok() {
                        let sender_pubkey = zap_request.pubkey.to_hex();
                        self.upsert_profile(&sender_pubkey).await?;
                        sqlx::query(
                            r#"
                            INSERT INTO events (id, pubkey, kind, created_at, raw_json)
                            VALUES (?, ?, ?, ?, ?)
                            ON CONFLICT(id, pubkey) DO NOTHING
                            "#,
                        )
                        .bind(&event_id)
                        .bind(&sender_pubkey)
                        .bind(kind)
                        .bind(created_at)
                        .bind(&raw_json)
                        .execute(&self.pool)
                        .await?;
                    } else {
                        tracing::warn!("Zap receipt {} has invalid 9734 signature, skipping sender indexing", event_id);
                    }
                }
            }
        }

        Ok(())
    }

    async fn update_profile_metadata(&self, pubkey: &str, content: &str, metadata_json: &str, nostr_created_at: i64) -> Result<()> {
        let meta: nostr_sdk::Metadata = match serde_json::from_str(content) {
            Ok(v) => v,
            Err(_) => return Ok(()), // skip unparseable metadata
        };

        let name = meta.name.as_deref();
        let about = meta.about.as_deref();
        let picture = meta.picture.as_deref();
        let nip05 = meta.nip05.as_deref();

        // Only update if this metadata event is newer than what we have
        sqlx::query(
            r#"
            UPDATE profiles SET
                name = COALESCE(?, name),
                about = COALESCE(?, about),
                picture = COALESCE(?, picture),
                nip05 = COALESCE(?, nip05),
                metadata_json = ?,
                metadata_created_at = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE pubkey = ? AND (metadata_created_at IS NULL OR metadata_created_at < ?)
            "#,
        )
        .bind(name)
        .bind(about)
        .bind(picture)
        .bind(nip05)
        .bind(metadata_json)
        .bind(nostr_created_at)
        .bind(pubkey)
        .bind(nostr_created_at)
        .execute(&self.pool)
        .await?;

        // Update FTS index if this profile is classified (name/about/nip05 changed)
        self.rebuild_fts_for_pubkey(pubkey).await?;

        Ok(())
    }

    pub async fn get_profile_event_count(&self, pubkey: &str) -> Result<usize> {
        let kinds: Vec<i64> = CLASSIFIABLE_KINDS.iter().map(|k| *k as i64).collect();
        let placeholders = kinds.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let sql = format!(
            "SELECT COUNT(*) FROM events WHERE pubkey = ? AND kind IN ({})",
            placeholders
        );

        let mut query = sqlx::query_scalar::<_, i64>(&sql).bind(pubkey);
        for kind in &kinds {
            query = query.bind(*kind);
        }
        let count = query.fetch_one(&self.pool).await?;

        Ok(count as usize)
    }

    pub async fn get_profile(&self, pubkey: &str) -> Result<(usize, bool)> {
        let kinds: Vec<i64> = CLASSIFIABLE_KINDS.iter().map(|k| *k as i64).collect();
        let placeholders = kinds.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let sql = format!(
            "SELECT (SELECT COUNT(*) FROM events WHERE pubkey = ? AND kind IN ({})), is_classified FROM profiles WHERE pubkey = ?",
            placeholders
        );

        let mut query = sqlx::query_as::<_, (i64, bool)>(&sql)
            .bind(pubkey);
        for kind in &kinds {
            query = query.bind(*kind);
        }
        let (count, is_classified) = query.bind(pubkey).fetch_one(&self.pool).await?;

        Ok((count as usize, is_classified))
    }

    pub async fn get_profile_details(&self, pubkey: &str) -> Result<Profile> {
        let profile = sqlx::query_as::<_, Profile>(
            r#"SELECT pubkey, nip05, name, about, picture, is_classified, needs_processing, follower_count, metadata_json, created_at, updated_at FROM profiles WHERE pubkey = ?"#,
        )
        .bind(pubkey)
        .fetch_one(&self.pool)
        .await?;

        Ok(profile)
    }

    pub async fn get_profile_events(
        &self,
        pubkey: &str,
        limit: usize,
    ) -> Result<Vec<Event>> {
        let kinds: Vec<i64> = CLASSIFIABLE_KINDS.iter().map(|k| *k as i64).collect();
        let placeholders = kinds.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let sql = format!(
            "SELECT id, pubkey, kind, created_at, raw_json FROM events WHERE pubkey = ? AND kind IN ({}) ORDER BY created_at DESC LIMIT ?",
            placeholders
        );

        let mut query = sqlx::query_as::<_, Event>(&sql).bind(pubkey);
        for kind in &kinds {
            query = query.bind(*kind);
        }
        let rows = query.bind(limit as i64).fetch_all(&self.pool).await?;

        Ok(rows)
    }

    pub async fn get_event(&self, event_id: &str) -> Result<Option<Event>> {
        let row = sqlx::query_as::<_, Event>(
            r#"SELECT id, pubkey, kind, created_at, raw_json FROM events WHERE id = ?"#,
        )
        .bind(event_id)
        .fetch_optional(&self.pool)
        .await?;

        Ok(row)
    }

    pub async fn get_profile_by_pubkey(&self, pubkey: &str) -> Result<Option<Profile>> {
        let profile = sqlx::query_as::<_, Profile>(
            r#"SELECT pubkey, nip05, name, about, picture, is_classified, needs_processing, follower_count, metadata_json, metadata_created_at, created_at, updated_at FROM profiles WHERE pubkey = ?"#,
        )
        .bind(pubkey)
        .fetch_optional(&self.pool)
        .await?;

        Ok(profile)
    }

    pub async fn set_follower_count(&self, pubkey: &str, count: usize) -> Result<()> {
        sqlx::query(
            r#"UPDATE profiles SET follower_count = ?, updated_at = CURRENT_TIMESTAMP WHERE pubkey = ?"#,
        )
        .bind(count as i64)
        .bind(pubkey)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn save_classification(
        &self,
        pubkey: &str,
        classification: &Classification,
        event_count: usize,
    ) -> Result<()> {
        let labels = serde_json::to_string(&classification.labels)?;
        let scores = serde_json::to_string(&classification.scores)?;

        sqlx::query(
            r#"
            INSERT INTO classifications (pubkey, labels, scores, bio, confidence, analyzed_event_count)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(pubkey) DO UPDATE SET
                labels = excluded.labels,
                scores = excluded.scores,
                bio = excluded.bio,
                confidence = excluded.confidence,
                analyzed_event_count = excluded.analyzed_event_count,
                analyzed_at = CURRENT_TIMESTAMP
            "#,
        )
        .bind(pubkey)
        .bind(&labels)
        .bind(&scores)
        .bind(&classification.bio)
        .bind(classification.confidence)
        .bind(event_count as i64)
        .execute(&self.pool)
        .await?;

        // Index into FTS (delete old entry by pubkey UNINDEXED column, then insert new)
        sqlx::query(
            r#"DELETE FROM classifications_fts WHERE pubkey = ?"#,
        )
        .bind(pubkey)
        .execute(&self.pool)
        .await?;

        // Fetch profile fields to include in FTS index
        let profile_fields: (Option<String>, Option<String>, Option<String>) =
            sqlx::query_as::<_, (Option<String>, Option<String>, Option<String>)>(
                r#"SELECT name, about, nip05 FROM profiles WHERE pubkey = ?"#,
            )
            .bind(pubkey)
            .fetch_one(&self.pool)
            .await
            .unwrap_or((None, None, None));

        // Join labels into plain text for FTS indexing
        let labels_text = classification.labels.join(" ");
        // Build scores text for FTS: repeat each label proportional to its score
        // e.g. bitcoin:0.9 → "bitcoin bitcoin", rust:0.4 → "rust"
        // This gives higher-scoring labels more weight in BM25 ranking
        let scores_text = scores_to_fts_text(&classification.scores);
        sqlx::query(
            r#"INSERT INTO classifications_fts (rowid, name, about, nip05, labels, scores, bio, pubkey)
               VALUES ((SELECT id FROM classifications WHERE pubkey = ?), ?, ?, ?, ?, ?, ?, ?)"#,
        )
        .bind(pubkey)
        .bind(&profile_fields.0)
        .bind(&profile_fields.1)
        .bind(&profile_fields.2)
        .bind(&labels_text)
        .bind(&scores_text)
        .bind(&classification.bio)
        .bind(pubkey)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn mark_profile_classified(&self, pubkey: &str) -> Result<()> {
        sqlx::query(
            r#"
            UPDATE profiles 
            SET is_classified = TRUE,
                needs_processing = FALSE,
                updated_at = CURRENT_TIMESTAMP
            WHERE pubkey = ?
            "#,
        )
        .bind(pubkey)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn get_classification(&self, pubkey: &str) -> Result<Classification> {
        let (labels, scores, bio, confidence) = sqlx::query_as::<_, (String, String, String, f64)>(
            r#"SELECT labels, scores, bio, confidence FROM classifications WHERE pubkey = ?"#,
        )
        .bind(pubkey)
        .fetch_one(&self.pool)
        .await?;

        let labels: Vec<String> = serde_json::from_str(&labels)
            .map_err(|e| anyhow::anyhow!("Failed to parse labels: {}", e))?;
        let scores: std::collections::HashMap<String, f64> = serde_json::from_str(&scores)
            .map_err(|e| anyhow::anyhow!("Failed to parse scores: {}", e))?;

        Ok(Classification {
            labels,
            scores,
            bio,
            confidence,
        })
    }

    pub async fn get_classification_if_exists(&self, pubkey: &str) -> Result<Option<Classification>> {
        let result = sqlx::query_as::<_, (String, String, String, f64)>(
            r#"SELECT labels, scores, bio, confidence FROM classifications WHERE pubkey = ?"#,
        )
        .bind(pubkey)
        .fetch_optional(&self.pool)
        .await?;

        match result {
            Some((labels, scores, bio, confidence)) => {
                let labels: Vec<String> = serde_json::from_str(&labels)
                    .map_err(|e| anyhow::anyhow!("Failed to parse labels: {}", e))?;
                let scores: std::collections::HashMap<String, f64> = serde_json::from_str(&scores)
                    .map_err(|e| anyhow::anyhow!("Failed to parse scores: {}", e))?;

                Ok(Some(Classification {
                    labels,
                    scores,
                    bio,
                    confidence,
                }))
            }
            None => Ok(None),
        }
    }

    pub async fn get_image_description(&self, hash: &str) -> Result<Option<String>> {
        let desc = sqlx::query_scalar::<_, String>(
            r#"SELECT description FROM image_descriptions WHERE hash = ?"#,
        )
        .bind(hash)
        .fetch_optional(&self.pool)
        .await?;

        Ok(desc)
    }

    pub async fn save_image_description(&self, hash: &str, description: &str) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO image_descriptions (hash, description)
            VALUES (?, ?)
            ON CONFLICT(hash) DO NOTHING
            "#,
        )
        .bind(hash)
        .bind(description)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn delete_old_events_for_pubkey(&self, pubkey: &str, max_age_days: u64) -> Result<u64> {
        let cutoff = chrono::Utc::now() - chrono::Duration::days(max_age_days as i64);
        let cutoff_ts = cutoff.timestamp();

        let kinds: Vec<i64> = CLASSIFIABLE_KINDS.iter().map(|k| *k as i64).collect();
        let placeholders = kinds.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let sql = format!(
            "DELETE FROM events WHERE pubkey = ? AND created_at < ? AND kind IN ({})",
            placeholders
        );

        let mut query = sqlx::query(&sql).bind(pubkey).bind(cutoff_ts);
        for kind in &kinds {
            query = query.bind(*kind);
        }
        let result = query.execute(&self.pool).await?;

        Ok(result.rows_affected())
    }

    pub async fn get_recent_classifications(&self, limit: i64) -> Result<Vec<crate::http_server::RecentClassification>> {
        let rows = sqlx::query_as::<_, (String, Option<String>, Option<String>, String, String, String, f64, Option<chrono::DateTime<chrono::Utc>>)>(
            r#"
            SELECT p.pubkey, p.name, p.picture, c.labels, c.scores, c.bio, c.confidence, c.analyzed_at
            FROM classifications c
            JOIN profiles p ON c.pubkey = p.pubkey
            ORDER BY c.analyzed_at DESC
            LIMIT ?
            "#,
        )
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        let results = rows.into_iter().map(|(pubkey, name, picture, labels, scores, bio, confidence, analyzed_at)| {
            let parsed_labels: Vec<String> = serde_json::from_str(&labels).unwrap_or_default();
            let parsed_scores: std::collections::HashMap<String, f64> = serde_json::from_str(&scores).unwrap_or_default();
            crate::http_server::RecentClassification {
                pubkey,
                name,
                picture,
                labels: parsed_labels,
                scores: parsed_scores,
                bio,
                confidence,
                analyzed_at: analyzed_at.map(|t| t.to_rfc3339()),
            }
        }).collect();

        Ok(results)
    }

    /// Rebuild the FTS entry for a classified profile (e.g. when profile metadata changes).
    /// No-op if the profile isn't classified yet.
    async fn rebuild_fts_for_pubkey(&self, pubkey: &str) -> Result<()> {
        // Only proceed if this profile has a classification
        let has_classification: bool = sqlx::query_scalar::<_, i64>(
            r#"SELECT COUNT(*) FROM classifications WHERE pubkey = ?"#,
        )
        .bind(pubkey)
        .fetch_one(&self.pool)
        .await?
        > 0;

        if !has_classification {
            return Ok(()); // Not classified yet, nothing to index
        }

        // Delete old FTS entry and re-insert with fresh profile data
        sqlx::query(r#"DELETE FROM classifications_fts WHERE pubkey = ?"#)
            .bind(pubkey)
            .execute(&self.pool)
            .await?;

        sqlx::query(
            r#"
            INSERT INTO classifications_fts (rowid, name, about, nip05, labels, scores, bio, pubkey)
            SELECT c.id, p.name, p.about, p.nip05,
                   REPLACE(REPLACE(SUBSTR(c.labels, 2, LENGTH(c.labels) - 2), '"', ''), ',', ' '),
                   c.scores,
                   c.bio, c.pubkey
            FROM classifications c
            JOIN profiles p ON c.pubkey = p.pubkey
            WHERE c.pubkey = ?
            "#,
        )
        .bind(pubkey)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn search_classifications(&self, query: &str, limit: i64) -> Result<Vec<crate::http_server::RecentClassification>> {
        // Add prefix matching to each term so "bitc" matches "bitcoin"
        let fts_query = Self::prepare_fts_query(query);

        let rows = sqlx::query_as::<_, (String, Option<String>, Option<String>, String, String, String, f64, Option<chrono::DateTime<chrono::Utc>>)>(
            r#"
            SELECT p.pubkey, p.name, p.picture, c.labels, c.scores, c.bio, c.confidence, c.analyzed_at
            FROM classifications c
            JOIN profiles p ON c.pubkey = p.pubkey
            JOIN classifications_fts fts ON fts.rowid = c.id
            WHERE classifications_fts MATCH ?
            ORDER BY bm25(classifications_fts) ASC
            LIMIT ?
            "#,
        )
        .bind(&fts_query)
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        let results = rows.into_iter().map(|(pubkey, name, picture, labels, scores, bio, confidence, analyzed_at)| {
            let parsed_labels: Vec<String> = serde_json::from_str(&labels).unwrap_or_default();
            let parsed_scores: std::collections::HashMap<String, f64> = serde_json::from_str(&scores).unwrap_or_default();
            crate::http_server::RecentClassification {
                pubkey,
                name,
                picture,
                labels: parsed_labels,
                scores: parsed_scores,
                bio,
                confidence,
                analyzed_at: analyzed_at.map(|t| t.to_rfc3339()),
            }
        }).collect();

        Ok(results)
    }

    /// Prepare a user query for FTS5: add prefix wildcard to each term.
    /// Handles FTS5 special characters by stripping them.
    /// e.g. "bitcoin rust" → "bitcoin* rust*"
    fn prepare_fts_query(query: &str) -> String {
        query
            .split_whitespace()
            .filter(|term| !term.is_empty())
            .map(|term| {
                // Strip FTS5 special characters that could cause parse errors
                let cleaned: String = term
                    .chars()
                    .filter(|c| !matches!(c, '"' | '\'' | ':' | '^' | '+' | '-' | '|' | '(' | ')' | '{' | '}'))
                    .collect();
                if cleaned.is_empty() {
                    String::new()
                } else {
                    format!("{}*", cleaned)
                }
            })
            .filter(|term| !term.is_empty())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// Convert scores map to FTS text where each label is repeated proportional to its score.
/// Score 0.0–0.33 → 1 occurrence, 0.34–0.66 → 2, 0.67–1.0 → 3.
/// This gives higher-scoring labels more weight in BM25 ranking.
fn scores_to_fts_text(scores: &std::collections::HashMap<String, f64>) -> String {
    let mut parts: Vec<String> = Vec::new();
    for (label, score) in scores {
        let repeats = (*score * 3.0).round() as usize;
        for _ in 0..repeats.max(1) {
            parts.push(label.clone());
        }
    }
    parts.join(" ")
}
