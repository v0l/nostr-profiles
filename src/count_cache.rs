use crate::db::Database;
use crate::nostr_client::NostrClient;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

/// In-memory cache for follower counts with 3-tier lookup:
/// memory → DB → relay fetch.
#[derive(Clone)]
pub struct FollowerCache {
    inner: Arc<Mutex<HashMap<String, usize>>>,
    db: Arc<Database>,
    nostr: NostrClient,
}

impl FollowerCache {
    pub fn new(db: Arc<Database>, nostr: NostrClient) -> Self {
        Self {
            inner: Arc::new(Mutex::new(HashMap::new())),
            db,
            nostr,
        }
    }

    /// Get the follower count for a pubkey. Returns `None` only if the count
    /// couldn't be determined from any source (relay fetch failed).
    pub async fn get(&self, pubkey: &str) -> Option<usize> {
        // 1. In-memory
        {
            let map = self.inner.lock().await;
            if let Some(&count) = map.get(pubkey) {
                return Some(count);
            }
        }

        // 2. DB
        let db_cached = self.db.get_profile_by_pubkey(pubkey)
            .await
            .ok()
            .flatten()
            .and_then(|p| p.follower_count)
            .map(|c| c as usize);

        if let Some(count) = db_cached {
            self.inner.lock().await.insert(pubkey.to_string(), count);
            return Some(count);
        }

        // 3. Relay fetch
        match self.nostr.fetch_follower_count(pubkey, 5).await {
            Ok(count) => {
                if let Err(e) = self.db.set_follower_count(pubkey, count).await {
                    tracing::warn!("Failed to cache follower count for {}: {}", pubkey, e);
                }
                self.inner.lock().await.insert(pubkey.to_string(), count);
                Some(count)
            }
            Err(e) => {
                tracing::warn!(
                    "Failed to fetch follower count for {}: {}, skipping check",
                    pubkey, e
                );
                None
            }
        }
    }
}

/// In-memory cache for event counts. Loads from DB on first access for a pubkey,
/// then increments in memory.
#[derive(Clone)]
pub struct EventCountCache {
    inner: Arc<Mutex<HashMap<String, usize>>>,
    db: Arc<Database>,
}

impl EventCountCache {
    pub fn new(db: Arc<Database>) -> Self {
        Self {
            inner: Arc::new(Mutex::new(HashMap::new())),
            db,
        }
    }

    /// Add `n` new events for a pubkey and return the updated total.
    /// Loads from DB on first access, then increments in memory.
    pub async fn increment(&self, pubkey: &str, n: usize) -> usize {
        let mut map = self.inner.lock().await;
        if let Some(existing) = map.get_mut(pubkey) {
            *existing += n;
            *existing
        } else {
            let db_count = self.db.get_profile_event_count(pubkey).await.unwrap_or(0);
            let total = db_count + n;
            map.insert(pubkey.to_string(), total);
            total
        }
    }

    /// Get the current count for a pubkey without incrementing.
    /// Loads from DB on first access.
    #[allow(dead_code)]
    pub async fn get(&self, pubkey: &str) -> usize {
        let mut map = self.inner.lock().await;
        if let Some(&count) = map.get(pubkey) {
            return count;
        }
        let db_count = self.db.get_profile_event_count(pubkey).await.unwrap_or(0);
        map.insert(pubkey.to_string(), db_count);
        db_count
    }
}
