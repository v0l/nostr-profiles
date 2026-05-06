use crate::db::Database;
use crate::nostr_client::NostrClient;
use anyhow::Result;
use std::sync::Arc;

/// Manages profile metadata caching and refresh.
///
/// Wraps the database cache and nostr relay client to provide
/// a single place for profile lookups with automatic stale-fetch logic.
#[derive(Clone)]
pub struct ProfileCache {
    db: Arc<Database>,
    nostr: NostrClient,
    stale_after_days: u64,
}

impl ProfileCache {
    pub fn new(db: Arc<Database>, nostr: NostrClient, stale_after_days: u64) -> Self {
        Self {
            db,
            nostr,
            stale_after_days,
        }
    }

    /// Get a profile from cache, fetching from relays if missing or stale.
    /// Returns None only if the profile doesn't exist anywhere.
    pub async fn get_profile(&self, pubkey: &str) -> Result<Option<crate::db::Profile>> {
        let cached = self.db.get_profile_by_pubkey(pubkey).await?;

        // Return cached profile if it's fresh enough and has metadata
        if let Some(ref profile) = cached {
            let age = chrono::Utc::now() - profile.updated_at;
            let has_metadata = profile.name.is_some() || profile.about.is_some();
            if has_metadata && age < chrono::Duration::days(self.stale_after_days as i64) {
                return Ok(cached);
            }
        }

        // Fetch kind 0 from relays
        self.fetch_and_cache_metadata(pubkey).await?;

        Ok(self.db.get_profile_by_pubkey(pubkey).await?)
    }

    /// Ensure a profile has metadata, fetching from relays if needed.
    /// No-op if the profile already has name or about fields.
    pub async fn ensure_metadata(&self, pubkey: &str) -> Result<()> {
        if let Some(profile) = self.db.get_profile_by_pubkey(pubkey).await? {
            if profile.name.is_some() || profile.about.is_some() {
                return Ok(());
            }
        }

        tracing::info!("Profile {} has no metadata, fetching from relays", pubkey);
        self.fetch_and_cache_metadata(pubkey).await?;
        Ok(())
    }

    /// Fetch kind 0 event from relays and cache it.
    async fn fetch_and_cache_metadata(&self, pubkey: &str) -> Result<()> {
        let pk = nostr_sdk::PublicKey::from_hex(pubkey)?;
        let filter = nostr_sdk::Filter::new()
            .kind(nostr_sdk::Kind::Metadata)
            .author(pk)
            .limit(1);

        let events = self
            .nostr
            .client()
            .fetch_events(filter, std::time::Duration::from_secs(10))
            .await?;

        if let Some(event) = events.into_iter().next() {
            if let Err(e) = self.db.cache_event(&event).await {
                tracing::warn!("Failed to cache fetched profile {}: {}", pubkey, e);
            }
        }

        Ok(())
    }
}
