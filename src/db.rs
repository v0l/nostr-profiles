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
/// - 34235:  Addressable Normal Video — NIP-71
/// - 34236:  Addressable Short Video — NIP-71
pub const CLASSIFIABLE_KINDS: &[u16] = &[
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
    34235,  // Addressable Normal Video
    34236,  // Addressable Short Video
];

/// Classification status derived from the classifications table.
#[derive(Debug, Clone)]
pub enum ClassificationStatus {
    /// No classification exists at all
    None,
    /// Classification exists but is from a stale epoch
    Stale { epoch: u32 },
    /// Classification is current
    Current,
}

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Profile {
    pub pubkey: String,
    pub nip05: Option<String>,
    pub name: Option<String>,
    pub display_name: Option<String>,
    pub about: Option<String>,
    pub picture: Option<String>,
    pub lud16: Option<String>,
    pub lud06: Option<String>,
    pub website: Option<String>,
    pub is_classified: bool,
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

/// Compute the storage ID for an event.
/// For replaceable (kind 0) and addressable (30000-39999) kinds, returns the
/// NIP-01 coordinate (`kind:pubkey` or `kind:pubkey:dtag`) so that
/// `ON CONFLICT(id, pubkey) DO UPDATE` naturally replaces stale versions.
/// For all other kinds, returns the event's own ID (unique per event).
pub fn event_storage_id(event: &nostr_sdk::Event) -> String {
    let kind = event.kind.as_u16();
    if event.kind.is_replaceable() {
        format!("{}:{}", kind, event.pubkey.to_hex())
    } else if event.kind.is_addressable() {
        let dtag = event.tags.identifier().unwrap_or("");
        format!("{}:{}:{}", kind, event.pubkey.to_hex(), dtag)
    } else {
        event.id.to_hex()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Classification {
    pub labels: Vec<String>,
    pub scores: std::collections::HashMap<String, f64>,
    pub bio: String,
    pub confidence: f64,
    pub kind_breakdown: Vec<KindCount>,
    pub analyzed_event_count: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KindCount {
    pub kind: i64,
    pub name: String,
    pub count: i64,
}

/// Cached full result of `get_stats`: (total_profiles, classified_profiles,
/// total_unique_labels, label_counts, images_classified, total_events).
type StatsTuple = (i64, i64, i64, Vec<(String, i64)>, i64, i64);

/// How long a cached `get_stats` result stays fresh. The COUNT(*) queries over
/// the events table are expensive on large databases (>10M rows), so we serve
/// slightly stale counts rather than re-scanning on every request.
const STATS_CACHE_TTL: std::time::Duration = std::time::Duration::from_secs(30);

#[derive(Debug)]
pub struct Database {
    pub pool: SqlitePool,
    label_min_score: f64,
    label_counts_cache: std::sync::Mutex<Option<(Vec<(String, i64)>, f64, i64)>>,
    stats_cache: std::sync::Mutex<Option<(StatsTuple, std::time::Instant)>>,
}

// Database is shared behind Arc, so Clone is manual (pool is Clone).
// Use Arc<Database> for multi-owner sharing.
impl Clone for Database {
    fn clone(&self) -> Self {
        Self {
            pool: self.pool.clone(),
            label_min_score: self.label_min_score,
            label_counts_cache: std::sync::Mutex::new(None),
            stats_cache: std::sync::Mutex::new(None),
        }
    }
}

impl Database {
    pub async fn new(path: &str, label_min_score: f64) -> Result<Self> {
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
        Self::run_migrations(&pool).await?;
        Ok(Self { pool, label_min_score, label_counts_cache: std::sync::Mutex::new(None), stats_cache: std::sync::Mutex::new(None) })
    }

    /// Derive the label list from scores: keys where score >= min_score, sorted descending.
    fn derive_labels(scores: &std::collections::HashMap<String, f64>, min_score: f64) -> Vec<String> {
        let mut scored: Vec<(String, f64)> = scores.iter()
            .filter(|(_, s)| **s >= min_score)
            .map(|(l, s)| (l.clone(), *s))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().map(|(l, _)| l).collect()
    }

    /// Run sqlx migrations. For legacy DBs (tables exist but no _sqlx_migrations table),
    /// mark the init migration as already applied before running.
    async fn run_migrations(pool: &SqlitePool) -> Result<()> {
        let has_migration_table: bool = sqlx::query_scalar::<_, i64>(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='_sqlx_migrations'",
        )
        .fetch_one(pool)
        .await?
        > 0;

        if has_migration_table {
            // Prod DB may have the old _sqlx_migrations schema with a `type` column
            // that sqlx 0.8 doesn't include in its INSERT. Detect and fix it.
            let has_type_column: bool = sqlx::query_scalar::<_, i64>(
                "SELECT COUNT(*) FROM pragma_table_info('_sqlx_migrations') WHERE name='type'",
            )
            .fetch_one(pool)
            .await?
            > 0;

            if has_type_column {
                tracing::info!("Fixing _sqlx_migrations table schema (dropping stray `type` column)");
                // SQLite doesn't support DROP COLUMN before 3.35.0, and even then
                // only if it's not referenced. Recreate the table to match sqlx's schema.
                sqlx::query(
                    r#"
                    CREATE TABLE _sqlx_migrations_new (
                        version BIGINT PRIMARY KEY,
                        description TEXT NOT NULL,
                        installed_on TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        success BOOLEAN NOT NULL,
                        checksum BLOB NOT NULL,
                        execution_time BIGINT NOT NULL
                    );
                    INSERT INTO _sqlx_migrations_new (version, description, installed_on, success, checksum, execution_time)
                        SELECT version, description, installed_on, success, checksum, execution_time FROM _sqlx_migrations;
                    DROP TABLE _sqlx_migrations;
                    ALTER TABLE _sqlx_migrations_new RENAME TO _sqlx_migrations;
                    "#,
                )
                .execute(pool)
                .await?;
            }
        }

        if !has_migration_table {
            // Check if this is a legacy DB (has actual data tables but no migration tracking)
            let has_profiles: bool = sqlx::query_scalar::<_, i64>(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='profiles'",
            )
            .fetch_one(pool)
            .await?
            > 0;

            if has_profiles {
                // Legacy DB — schema was created by old manual migrations.
                // Record the init migration as applied so sqlx::migrate skips it.
                let migrator = sqlx::migrate!();
                if let Some(init) = migrator.iter().find(|m| m.version == 20250101000000) {
                    // Match the exact schema sqlx-sqlite uses (no `type` column)
                    sqlx::query(
                        r#"CREATE TABLE IF NOT EXISTS _sqlx_migrations (
                            version BIGINT PRIMARY KEY,
                            description TEXT NOT NULL,
                            installed_on TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                            success BOOLEAN NOT NULL,
                            checksum BLOB NOT NULL,
                            execution_time BIGINT NOT NULL
                        )"#
                    )
                    .execute(pool)
                    .await?;

                    sqlx::query(
                        r#"INSERT INTO _sqlx_migrations (version, description, success, checksum, execution_time)
                           VALUES (?, ?, TRUE, ?, -1)"#
                    )
                    .bind(init.version)
                    .bind(&init.description)
                    .bind(init.checksum.as_ref() as &[u8])
                    .execute(pool)
                    .await?;
                }
            }
            // Fresh DB: no tables, no migration table — just let sqlx::migrate run normally
        }

        sqlx::migrate!().run(pool).await?;

        // After migrations, rebuild FTS if it was dropped (schema changes).
        let fts_count: i64 = sqlx::query_scalar(
            r#"SELECT COUNT(*) FROM classifications_fts"#,
        )
        .fetch_one(pool)
        .await?;
        if fts_count == 0 {
            sqlx::query(
                r#"
                INSERT INTO classifications_fts (rowid, name, display_name, about, nip05, lud16, lud06, website, labels, scores, bio, pubkey)
                SELECT c.id, p.name, p.display_name, p.about, p.nip05, p.lud16, p.lud06, p.website,
                       (SELECT group_concat(cl.label, ' ') FROM classification_labels cl WHERE cl.pubkey = c.pubkey),
                       c.scores,
                       c.bio, c.pubkey
                FROM classifications c
                JOIN profiles p ON p.pubkey = c.pubkey
                "#,
            )
            .execute(pool)
            .await?;
        }
        Ok(())
    }

    /// Get pubkeys of profiles that need classification.
    /// A profile needs classification if it has enough events and either
    /// has no classification or its classification_epoch < current_epoch.
    pub async fn get_profiles_needing_classification(&self, min_events: i64, current_epoch: u32) -> Result<Vec<String>> {
        let kinds: Vec<i64> = CLASSIFIABLE_KINDS.iter().map(|k| *k as i64).collect();
        let placeholders = kinds.iter().map(|_| "?").collect::<Vec<_>>().join(",");

        let sql = format!(
            r#"SELECT p.pubkey FROM profiles p
               INNER JOIN events e ON e.pubkey = p.pubkey AND e.kind IN ({})
               LEFT JOIN classifications c ON c.pubkey = p.pubkey
               WHERE c.pubkey IS NULL OR c.classification_epoch < ?
               GROUP BY p.pubkey
               HAVING COUNT(*) >= ?"#,
            placeholders
        );

        let mut query = sqlx::query_scalar::<_, String>(&sql);
        for kind in &kinds {
            query = query.bind(*kind);
        }
        query = query.bind(current_epoch as i64);
        query = query.bind(min_events);
        let rows = query.fetch_all(&self.pool).await?;

        Ok(rows)
    }

    pub async fn upsert_profile(&self, pubkey: &str) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO profiles (pubkey)
            VALUES (?)
            ON CONFLICT(pubkey) DO UPDATE SET
                updated_at = CURRENT_TIMESTAMP
            "#,
        )
        .bind(pubkey)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Insert a batch of events in a single transaction. Much faster than individual inserts.
    /// Returns the net new event count per pubkey (accounts for replaceable events that
    /// overwrite older versions — those result in net 0, not +1).
    pub async fn cache_events_batch(&self, events: &[nostr_sdk::Event]) -> Result<std::collections::HashMap<String, i64>> {
        let mut tx = self.pool.begin().await?;
        let mut net_new: std::collections::HashMap<String, i64> = std::collections::HashMap::new();

        for event in events {
            let storage_id = event_storage_id(event);
            let pubkey = event.pubkey.to_hex();
            let kind = (event.kind.as_u16() as u32) as i64;
            let created_at = event.created_at.as_secs() as i64;
            let raw_json = serde_json::to_string(event)?;

            // Ensure profile exists
            sqlx::query(
                r#"INSERT INTO profiles (pubkey) VALUES (?)
                   ON CONFLICT(pubkey) DO UPDATE SET updated_at = CURRENT_TIMESTAMP"#,
            )
            .bind(&pubkey)
            .execute(&mut *tx)
            .await?;

            // Kind 0 = metadata — extract profile fields and store in profiles table only.
            // Metadata events are not inserted into the events table since they are already
            // represented by the profiles.metadata_json column and provide no classification signal.
            if kind == 0 {
                let content = &event.content;
                let meta: nostr_sdk::Metadata = match serde_json::from_str(content) {
                    Ok(v) => v,
                    Err(_) => continue, // skip unparseable metadata
                };

                sqlx::query(
                    r#"UPDATE profiles SET
                        name = COALESCE(?, name),
                        display_name = COALESCE(?, display_name),
                        about = COALESCE(?, about),
                        picture = COALESCE(?, picture),
                        nip05 = COALESCE(?, nip05),
                        lud16 = COALESCE(?, lud16),
                        lud06 = COALESCE(?, lud06),
                        website = COALESCE(?, website),
                        metadata_json = ?,
                        metadata_created_at = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE pubkey = ? AND (metadata_created_at IS NULL OR metadata_created_at < ?)"#,
                )
                .bind(meta.name.as_deref())
                .bind(meta.display_name.as_deref())
                .bind(meta.about.as_deref())
                .bind(meta.picture.as_deref())
                .bind(meta.nip05.as_deref())
                .bind(meta.lud16.as_deref())
                .bind(meta.lud06.as_deref())
                .bind(meta.website.as_deref())
                .bind(&raw_json)
                .bind(created_at)
                .bind(&pubkey)
                .bind(created_at)
                .execute(&mut *tx)
                .await?;

                continue; // Don't insert kind:0 into the events table
            }

            // Insert event — replaceable/addressable events use coordinate as ID,
            // so ON CONFLICT ... DO UPDATE replaces the old version (only if newer).
            let result = sqlx::query(
                r#"INSERT INTO events (id, pubkey, kind, created_at, raw_json)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(id, pubkey) DO UPDATE SET
                       kind = excluded.kind,
                       created_at = excluded.created_at,
                       raw_json = excluded.raw_json
                   WHERE excluded.created_at > events.created_at"#,
            )
            .bind(&storage_id)
            .bind(&pubkey)
            .bind(kind)
            .bind(created_at)
            .bind(&raw_json)
            .execute(&mut *tx)
            .await?;

            // If this was a new row (not an update of an existing one), count it.
            // For replaceable events, an update means net 0 (old replaced, not added).
            if result.rows_affected() > 0 {
                *net_new.entry(pubkey.clone()).or_insert(0) += 1;
            }

            // For zap receipts, also index under sender pubkey
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
                            sqlx::query(
                                r#"INSERT INTO profiles (pubkey) VALUES (?)
                                   ON CONFLICT(pubkey) DO UPDATE SET updated_at = CURRENT_TIMESTAMP"#,
                            )
                            .bind(&sender_pubkey)
                            .execute(&mut *tx)
                            .await?;

                            let result = sqlx::query(
                                r#"INSERT INTO events (id, pubkey, kind, created_at, raw_json)
                                   VALUES (?, ?, ?, ?, ?)
                                   ON CONFLICT(id, pubkey) DO UPDATE SET
                                       kind = excluded.kind,
                                       created_at = excluded.created_at,
                                       raw_json = excluded.raw_json
                                   WHERE excluded.created_at > events.created_at"#,
                            )
                            .bind(&storage_id)
                            .bind(&sender_pubkey)
                            .bind(kind)
                            .bind(created_at)
                            .bind(&raw_json)
                            .execute(&mut *tx)
                            .await?;

                            if result.rows_affected() > 0 {
                                *net_new.entry(sender_pubkey).or_insert(0) += 1;
                            }
                        }
                    }
                }
            }
        }

        tx.commit().await?;
        Ok(net_new)
    }

    /// Get the classification status for a profile.
    pub async fn get_classification_status(&self, pubkey: &str, current_epoch: u32) -> Result<ClassificationStatus> {
        let epoch: Option<i64> = sqlx::query_scalar(
            r#"SELECT classification_epoch FROM classifications WHERE pubkey = ?"#,
        )
        .bind(pubkey)
        .fetch_optional(&self.pool)
        .await?;

        Ok(match epoch {
            None => ClassificationStatus::None,
            Some(e) if (e as u32) < current_epoch => ClassificationStatus::Stale { epoch: e as u32 },
            Some(_) => ClassificationStatus::Current,
        })
    }

    /// Check if a profile has a current-epoch classification.
    /// Returns true if the profile has a classification with epoch >= current_epoch.
    pub async fn has_current_classification(&self, pubkey: &str, current_epoch: u32) -> Result<bool> {
        let status = self.get_classification_status(pubkey, current_epoch).await?;
        Ok(matches!(status, ClassificationStatus::Current))
    }

    pub async fn cache_event(&self, event: &nostr_sdk::Event) -> Result<()> {
        let storage_id = event_storage_id(event);
        let pubkey = event.pubkey.to_hex();
        let kind = (event.kind.as_u16() as u32) as i64;
        let created_at = event.created_at.as_secs() as i64;
        let raw_json = serde_json::to_string(event)?;
        
        // Ensure profile exists first (for foreign key constraint)
        self.upsert_profile(&pubkey).await?;

        // Kind 0 = metadata — extract profile fields and store in profiles table only.
        // Metadata events are not inserted into the events table since they are already
        // represented by the profiles.metadata_json column and provide no classification signal.
        if kind == 0 {
            self.update_profile_metadata(&pubkey, &event.content, &raw_json, created_at).await?;
            return Ok(());
        }

        sqlx::query(
            r#"
            INSERT INTO events (id, pubkey, kind, created_at, raw_json)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id, pubkey) DO UPDATE SET
                kind = excluded.kind,
                created_at = excluded.created_at,
                raw_json = excluded.raw_json
            WHERE excluded.created_at > events.created_at
            "#,
        )
        .bind(&storage_id)
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
                            ON CONFLICT(id, pubkey) DO UPDATE SET
                                kind = excluded.kind,
                                created_at = excluded.created_at,
                                raw_json = excluded.raw_json
                            WHERE excluded.created_at > events.created_at
                            "#,
                        )
                        .bind(&storage_id)
                        .bind(&sender_pubkey)
                        .bind(kind)
                        .bind(created_at)
                        .bind(&raw_json)
                        .execute(&self.pool)
                        .await?;
                    } else {
                        tracing::warn!("Zap receipt {} has invalid 9734 signature, skipping sender indexing", storage_id);
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
        let display_name = meta.display_name.as_deref();
        let about = meta.about.as_deref();
        let picture = meta.picture.as_deref();
        let nip05 = meta.nip05.as_deref();
        let lud16 = meta.lud16.as_deref();
        let lud06 = meta.lud06.as_deref();
        let website = meta.website.as_deref();

        // Only update if this metadata event is newer than what we have
        sqlx::query(
            r#"
            UPDATE profiles SET
                name = COALESCE(?, name),
                display_name = COALESCE(?, display_name),
                about = COALESCE(?, about),
                picture = COALESCE(?, picture),
                nip05 = COALESCE(?, nip05),
                lud16 = COALESCE(?, lud16),
                lud06 = COALESCE(?, lud06),
                website = COALESCE(?, website),
                metadata_json = ?,
                metadata_created_at = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE pubkey = ? AND (metadata_created_at IS NULL OR metadata_created_at < ?)
            "#,
        )
        .bind(name)
        .bind(display_name)
        .bind(about)
        .bind(picture)
        .bind(nip05)
        .bind(lud16)
        .bind(lud06)
        .bind(website)
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

    pub async fn get_profile_details(&self, pubkey: &str) -> Result<Profile> {
        let profile = sqlx::query_as::<_, Profile>(
            r#"SELECT pubkey, nip05, name, display_name, about, picture, lud16, lud06, website, is_classified, follower_count, metadata_json, metadata_created_at, created_at, updated_at FROM profiles WHERE pubkey = ?"#,
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

    pub async fn get_profile_events_by_kind(
        &self,
        pubkey: &str,
        kind: i64,
        since: Option<i64>,
        until: Option<i64>,
        limit: usize,
    ) -> Result<Vec<Event>> {
        let mut builder = sqlx::QueryBuilder::new(
            "SELECT id, pubkey, kind, created_at, raw_json FROM events WHERE pubkey = ? AND kind = ?"
        );
        builder.push_bind(pubkey).push_bind(kind);

        if let Some(s) = since {
            builder.push(" AND created_at >= ").push_bind(s);
        }
        if let Some(u) = until {
            builder.push(" AND created_at <= ").push_bind(u);
        }
        builder.push(" ORDER BY created_at DESC LIMIT ").push_bind(limit as i64);

        let rows = builder.build_query_as::<Event>().fetch_all(&self.pool).await?;

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
            r#"SELECT pubkey, nip05, name, display_name, about, picture, lud16, lud06, website, is_classified, follower_count, metadata_json, metadata_created_at, created_at, updated_at FROM profiles WHERE pubkey = ?"#,
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
        epoch: u32,
    ) -> Result<()> {
        let scores = serde_json::to_string(&classification.scores)?;
        let kind_breakdown = serde_json::to_string(&classification.kind_breakdown)?;

        // Delete old label rows for this pubkey
        sqlx::query("DELETE FROM classification_labels WHERE pubkey = ?")
            .bind(pubkey)
            .execute(&self.pool)
            .await?;

        sqlx::query(
            r#"
            INSERT INTO classifications (pubkey, scores, bio, confidence, analyzed_event_count, kind_breakdown, classification_epoch)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(pubkey) DO UPDATE SET
                scores = excluded.scores,
                bio = excluded.bio,
                confidence = excluded.confidence,
                analyzed_event_count = excluded.analyzed_event_count,
                kind_breakdown = excluded.kind_breakdown,
                classification_epoch = excluded.classification_epoch,
                analyzed_at = CURRENT_TIMESTAMP
            "#,
        )
        .bind(pubkey)
        .bind(&scores)
        .bind(&classification.bio)
        .bind(classification.confidence)
        .bind(event_count as i64)
        .bind(&kind_breakdown)
        .bind(epoch as i64)
        .execute(&self.pool)
        .await?;

        // Populate indexed label rows
        for (label, score) in &classification.scores {
            sqlx::query(
                "INSERT INTO classification_labels (pubkey, label, score) VALUES (?, ?, ?)",
            )
            .bind(pubkey)
            .bind(label)
            .bind(score)
            .execute(&self.pool)
            .await?;
        }

        // Rebuild FTS entry for this profile
        self.rebuild_fts_for_pubkey(pubkey).await?;

        // Invalidate label counts cache — a new classification changes the counts
        *self.label_counts_cache.lock().unwrap() = None;
        // The cached stats snapshot is now stale too (counts changed).
        *self.stats_cache.lock().unwrap() = None;

        Ok(())
    }

    pub async fn get_classification(&self, pubkey: &str) -> Result<Classification> {
        let (scores, bio, confidence, kind_breakdown, analyzed_event_count) = sqlx::query_as::<_, (String, String, f64, String, i64)>(
            r#"SELECT scores, bio, confidence, kind_breakdown, analyzed_event_count FROM classifications WHERE pubkey = ?"#,
        )
        .bind(pubkey)
        .fetch_one(&self.pool)
        .await?;

        let scores: std::collections::HashMap<String, f64> = serde_json::from_str(&scores)
            .map_err(|e| anyhow::anyhow!("Failed to parse scores: {}", e))?;
        let kind_breakdown: Vec<KindCount> = serde_json::from_str(&kind_breakdown)
            .unwrap_or_default();

        // Derive labels from scores (sorted by score descending)
        let labels = Self::derive_labels(&scores, self.label_min_score);

        Ok(Classification {
            labels,
            scores,
            bio,
            confidence,
            kind_breakdown,
            analyzed_event_count,
        })
    }

    pub async fn get_classification_if_exists(&self, pubkey: &str) -> Result<Option<Classification>> {
        let result = sqlx::query_as::<_, (String, String, f64, String, i64)>(
            r#"SELECT scores, bio, confidence, kind_breakdown, analyzed_event_count FROM classifications WHERE pubkey = ?"#,
        )
        .bind(pubkey)
        .fetch_optional(&self.pool)
        .await?;

        match result {
            Some((scores, bio, confidence, kind_breakdown, analyzed_event_count)) => {
                let scores: std::collections::HashMap<String, f64> = serde_json::from_str(&scores)
                    .map_err(|e| anyhow::anyhow!("Failed to parse scores: {}", e))?;
                let kind_breakdown: Vec<KindCount> = serde_json::from_str(&kind_breakdown)
                    .unwrap_or_default();

                let labels = Self::derive_labels(&scores, self.label_min_score);

                Ok(Some(Classification {
                    labels,
                    scores,
                    bio,
                    confidence,
                    kind_breakdown,
                    analyzed_event_count,
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
        let rows = sqlx::query_as::<_, (String, Option<String>, Option<String>, Option<String>, String, String, f64, Option<chrono::DateTime<chrono::Utc>>, Option<String>)>(
            r#"
            SELECT p.pubkey, p.name, p.display_name, p.picture, c.scores, c.bio, c.confidence, c.analyzed_at, p.metadata_json
            FROM classifications c
            JOIN profiles p ON c.pubkey = p.pubkey
            ORDER BY c.analyzed_at DESC
            LIMIT ?
            "#,
        )
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        let results = rows.into_iter().map(|(pubkey, name, display_name, picture, scores, bio, confidence, analyzed_at, metadata_json)| {
            let parsed_scores: std::collections::HashMap<String, f64> = serde_json::from_str(&scores).unwrap_or_default();
            crate::http_server::RecentClassification {
                pubkey,
                name,
                display_name,
                picture,
                scores: parsed_scores,
                bio,
                confidence,
                analyzed_at: analyzed_at.map(|t| t.to_rfc3339()),
                metadata_json,
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

        // Delete old FTS entry and re-insert with fresh data
        sqlx::query(r#"DELETE FROM classifications_fts WHERE pubkey = ?"#)
            .bind(pubkey)
            .execute(&self.pool)
            .await?;

        // Derive labels from the indexed classification_labels table, joined as space-separated text for FTS
        sqlx::query(
            r#"
            INSERT INTO classifications_fts (rowid, name, display_name, about, nip05, lud16, lud06, website, labels, scores, bio, pubkey)
            SELECT c.id, p.name, p.display_name, p.about, p.nip05, p.lud16, p.lud06, p.website,
                   (SELECT group_concat(cl.label, ' ') FROM classification_labels cl WHERE cl.pubkey = c.pubkey),
                   c.scores,
                   c.bio, c.pubkey
            FROM classifications c
            JOIN profiles p ON p.pubkey = c.pubkey
            WHERE c.pubkey = ?
            "#,
        )
        .bind(pubkey)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Rebuild the entire FTS index from classifications + profiles.
    /// Call this after batch operations or if the FTS index gets out of sync.
    #[allow(dead_code)]
    pub async fn rebuild_fts(&self) -> Result<()> {
        // Clear existing FTS data
        sqlx::query(r#"DELETE FROM classifications_fts"#)
            .execute(&self.pool)
            .await?;

        // Rebuild from classifications + profiles
        sqlx::query(
            r#"
            INSERT INTO classifications_fts (rowid, name, display_name, about, nip05, lud16, lud06, website, labels, scores, bio, pubkey)
            SELECT c.id, p.name, p.display_name, p.about, p.nip05, p.lud16, p.lud06, p.website,
                   (SELECT group_concat(cl.label, ' ') FROM classification_labels cl WHERE cl.pubkey = c.pubkey),
                   c.scores,
                   c.bio, c.pubkey
            FROM classifications c
            JOIN profiles p ON p.pubkey = c.pubkey
            "#,
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn search_classifications(&self, query: &str, limit: i64) -> Result<Vec<crate::http_server::RecentClassification>> {
        // Add prefix matching to each term so "bitc" matches "bitcoin"
        let fts_query = Self::prepare_fts_query(query);

        let rows = sqlx::query_as::<_, (String, Option<String>, Option<String>, Option<String>, String, String, f64, Option<chrono::DateTime<chrono::Utc>>, Option<String>)>(
            r#"
            SELECT p.pubkey, p.name, p.display_name, p.picture, c.scores, c.bio, c.confidence, c.analyzed_at, p.metadata_json
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

        let results = rows.into_iter().map(|(pubkey, name, display_name, picture, scores, bio, confidence, analyzed_at, metadata_json)| {
            let parsed_scores: std::collections::HashMap<String, f64> = serde_json::from_str(&scores).unwrap_or_default();
            crate::http_server::RecentClassification {
                pubkey,
                name,
                display_name,
                picture,
                scores: parsed_scores,
                bio,
                confidence,
                analyzed_at: analyzed_at.map(|t| t.to_rfc3339()),
                metadata_json,
            }
        }).collect();

        Ok(results)
    }

    /// Search profiles by exact label match using the indexed classification_labels table.
    pub async fn search_by_label(&self, label: &str, limit: i64) -> Result<Vec<crate::http_server::RecentClassification>> {
        let rows = sqlx::query_as::<_, (String, Option<String>, Option<String>, Option<String>, String, String, f64, Option<chrono::DateTime<chrono::Utc>>, Option<String>)>(
            r#"
            SELECT p.pubkey, p.name, p.display_name, p.picture, c.scores, c.bio, c.confidence, c.analyzed_at, p.metadata_json
            FROM classification_labels l
            JOIN classifications c ON c.pubkey = l.pubkey
            JOIN profiles p ON p.pubkey = c.pubkey
            WHERE l.label = ?
            ORDER BY c.analyzed_at DESC
            LIMIT ?
            "#,
        )
        .bind(label)
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        let results = rows.into_iter().map(|(pubkey, name, display_name, picture, scores, bio, confidence, analyzed_at, metadata_json)| {
            let parsed_scores: std::collections::HashMap<String, f64> = serde_json::from_str(&scores).unwrap_or_default();
            crate::http_server::RecentClassification {
                pubkey,
                name,
                display_name,
                picture,
                scores: parsed_scores,
                bio,
                confidence,
                analyzed_at: analyzed_at.map(|t| t.to_rfc3339()),
                metadata_json,
            }
        }).collect();

        Ok(results)
    }

    /// Get all stats in a single query.
    /// Returns (total_profiles, classified_profiles, total_unique_labels, label_counts, images_classified, total_events).
    /// Labels are derived from scores keys where score >= label_min_score.
    pub async fn get_stats(&self) -> Result<StatsTuple> {
        let min_score = self.label_min_score;
        let epoch = crate::CLASSIFICATION_EPOCH as i64;

        // Serve a recently-cached full result if still fresh. The COUNT(*)
        // queries below scan the entire events table (>10M rows), which is far
        // too expensive to run on every /api/stats request.
        {
            let cache = self.stats_cache.lock().unwrap();
            if let Some((ref cached, at)) = *cache {
                if at.elapsed() < STATS_CACHE_TTL {
                    return Ok(cached.clone());
                }
            }
        }

        let (total_profiles, classified_profiles, images_classified, total_events): (i64, i64, i64, i64) =
            sqlx::query_as(
                r#"SELECT
                    (SELECT COUNT(*) FROM profiles),
                    (SELECT COUNT(*) FROM classifications WHERE classification_epoch >= ?),
                    (SELECT COUNT(*) FROM image_descriptions),
                    (SELECT COUNT(*) FROM events)"#,
            )
            .bind(epoch)
            .fetch_one(&self.pool)
            .await?;

        // Label counts change rarely — only when a new classification is saved.
        // Cache in memory keyed by (min_score, epoch) to avoid the expensive
        // 75K-row aggregation on every /api/stats hit.
        {
            let cache = self.label_counts_cache.lock().unwrap();
            if let Some((ref cached_counts, cached_min_score, cached_epoch)) = *cache {
                if (cached_min_score - min_score).abs() < f64::EPSILON && cached_epoch == epoch {
                    let result = (total_profiles, classified_profiles, cached_counts.len() as i64, cached_counts.clone(), images_classified, total_events);
                    *self.stats_cache.lock().unwrap() = Some((result.clone(), std::time::Instant::now()));
                    return Ok(result);
                }
            }
        }

        let label_counts: Vec<(String, i64)> = sqlx::query_as(
            r#"SELECT classification_labels.label, COUNT(*) AS count
               FROM classifications
               JOIN classification_labels ON classification_labels.pubkey = classifications.pubkey
               WHERE classification_labels.score >= ? AND classifications.classification_epoch >= ?
               GROUP BY classification_labels.label
               ORDER BY count DESC"#,
        )
        .bind(min_score)
        .bind(epoch)
        .fetch_all(&self.pool)
        .await?;

        let total_unique_labels = label_counts.len() as i64;

        // Cache the result
        *self.label_counts_cache.lock().unwrap() = Some((label_counts.clone(), min_score, epoch));

        let result = (total_profiles, classified_profiles, total_unique_labels, label_counts, images_classified, total_events);
        *self.stats_cache.lock().unwrap() = Some((result.clone(), std::time::Instant::now()));
        Ok(result)
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
