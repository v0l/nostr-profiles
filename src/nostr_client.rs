use crate::db::Database;
use anyhow::Result;
use nostr_sdk::prelude::*;
use std::sync::Arc;

/// Shared nostr client for both event collection and on-demand fetches.
#[derive(Clone)]
pub struct NostrClient {
    client: Client,
    db: Arc<Database>,
}

impl NostrClient {
    pub async fn new(relays: &[String], nsec: Option<&str>, db: Arc<Database>) -> Result<Self> {
        let mut builder = Client::builder();

        let keys = match nsec {
            Some(nsec_str) if !nsec_str.is_empty() => {
                let secret_key = SecretKey::from_bech32(nsec_str)
                    .or_else(|_| SecretKey::from_hex(nsec_str))?;
                Keys::new(secret_key)
            }
            _ => Keys::generate(),
        };
        builder = builder.signer(keys);

        let client = builder.build();

        for relay_url in relays {
            client.add_relay(relay_url).await?;
        }

        Ok(Self { client, db })
    }

    pub async fn connect(&self) {
        self.client.connect().await;
    }

    pub fn client(&self) -> &Client {
        &self.client
    }

    /// Fetch a single event by ID from relays, cache it, return it.
    pub async fn fetch_event(&self, event_id: &str) -> Result<Option<nostr_sdk::Event>> {
        // Check DB cache first
        if let Some(cached) = self.db.get_event(event_id).await? {
            if let Ok(event) = serde_json::from_str::<nostr_sdk::Event>(&cached.raw_json) {
                return Ok(Some(event));
            }
        }

        // Fetch from relays
        let id = EventId::from_hex(event_id)?;
        let filter = Filter::new().id(id).limit(1);

        let events = self.client
            .fetch_events(filter, std::time::Duration::from_secs(10))
            .await?;

        if let Some(event) = events.into_iter().next() {
            if let Err(e) = self.db.cache_event(&event).await {
                tracing::warn!("Failed to cache fetched event {}: {}", event_id, e);
            }
            return Ok(Some(event));
        }

        Ok(None)
    }

    /// Fetch profile metadata (kind 0) for a pubkey from relays, cache it, return the profile.
    pub async fn fetch_profile(&self, pubkey: &str) -> Result<Option<crate::db::Profile>> {
        // Check DB cache first
        if let Some(profile) = self.db.get_profile_by_pubkey(pubkey).await? {
            if profile.name.is_some() || profile.about.is_some() {
                return Ok(Some(profile));
            }
        }

        // Fetch kind 0 from relays
        let pk = PublicKey::from_hex(pubkey)?;
        let filter = Filter::new().kind(Kind::Metadata).author(pk).limit(1);

        let events = self.client
            .fetch_events(filter, std::time::Duration::from_secs(10))
            .await?;

        if let Some(event) = events.into_iter().next() {
            if let Err(e) = self.db.cache_event(&event).await {
                tracing::warn!("Failed to cache fetched profile {}: {}", pubkey, e);
            }
            return Ok(self.db.get_profile_by_pubkey(pubkey).await?);
        }

        Ok(None)
    }

    /// Count how many kind 3 (Contacts) events reference this pubkey in a 'p' tag.
    /// This gives a rough follower count from the relays we're connected to.
    pub async fn fetch_follower_count(&self, pubkey: &str, timeout_secs: u64) -> Result<usize> {
        let pk = PublicKey::from_hex(pubkey)?;
        let filter = Filter::new()
            .kind(Kind::ContactList)
            .pubkey(pk);

        let events = self.client
            .fetch_events(filter, std::time::Duration::from_secs(timeout_secs))
            .await?;

        Ok(events.len())
    }
}
