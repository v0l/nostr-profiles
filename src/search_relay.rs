use std::collections::BTreeSet;
use std::fmt;
use std::sync::Arc;

use nostr_database::prelude::*;
use nostr_database::{Backend, DatabaseError, DatabaseEventStatus, Events, SaveEventStatus};
use nostr_relay_builder::prelude::*;

use crate::db::Database;

/// A read-only nostr database backed by the nostr-classify FTS index.
///
/// This implements `NostrDatabase` to serve as the backend for a search relay.
/// It only returns kind 0 (Metadata) events constructed from the profiles table,
/// and uses the FTS5 index when the filter includes a `search` field.
#[derive(Clone)]
pub struct SearchDatabase {
    db: Arc<Database>,
}

impl fmt::Debug for SearchDatabase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SearchDatabase").finish()
    }
}

impl SearchDatabase {
    pub fn new(db: Arc<Database>) -> Self {
        Self { db }
    }

    /// Build a `LocalRelay` using this search database.
    pub fn build_relay(db: Arc<Database>) -> LocalRelay {
        let search_db = Self::new(db);
        LocalRelay::new(
            RelayBuilder::default()
                .database(search_db)
                .default_filter_limit(100),
        )
    }

    /// Query profiles from the database matching the given filter criteria.
    async fn query_profiles(
        &self,
        authors: Option<&BTreeSet<PublicKey>>,
        search: Option<&String>,
        limit: usize,
    ) -> Result<Vec<Event>, DatabaseError> {
        // If we have a search term, use FTS
        if let Some(search_term) = search {
            let search_term = search_term.trim();
            if !search_term.is_empty() {
                return self.search_profiles(search_term, limit).await;
            }
        }

        // If we have specific authors, look them up directly
        if let Some(author_keys) = authors {
            let mut events = Vec::new();
            for pk in author_keys.iter().take(limit) {
                if let Some(event) = self.get_metadata_event_for_pubkey(&pk.to_hex()).await {
                    events.push(event);
                }
            }
            return Ok(events);
        }

        // No specific query - return recent classified profiles
        self.get_recent_metadata_events(limit).await
    }

    /// Search profiles using the FTS5 index.
    async fn search_profiles(&self, search_term: &str, limit: usize) -> Result<Vec<Event>, DatabaseError> {
        let results = self
            .db
            .search_classifications(search_term, limit as i64)
            .await
            .map_err(|e| DatabaseError::Backend(Box::new(std::io::Error::other(e.to_string()))))?;

        let mut events = Vec::new();
        for rc in results {
            if let Some(event) = self.get_metadata_event_for_pubkey(&rc.pubkey).await {
                events.push(event);
            }
        }
        Ok(events)
    }

    /// Get recent classified profiles as metadata events.
    async fn get_recent_metadata_events(&self, limit: usize) -> Result<Vec<Event>, DatabaseError> {
        let results = self
            .db
            .get_recent_classifications(limit as i64)
            .await
            .map_err(|e| DatabaseError::Backend(Box::new(std::io::Error::other(e.to_string()))))?;

        let mut events = Vec::new();
        for rc in results {
            if let Some(event) = self.get_metadata_event_for_pubkey(&rc.pubkey).await {
                events.push(event);
            }
        }
        Ok(events)
    }

    /// Construct a kind 0 (Metadata) event for a given pubkey from the profiles table.
    async fn get_metadata_event_for_pubkey(&self, pubkey_hex: &str) -> Option<Event> {
        // Only return classified profiles from the search relay
        let has_classification: bool = sqlx::query_scalar::<_, i64>(
            r#"SELECT COUNT(*) FROM classifications WHERE pubkey = ? AND classification_epoch >= ?"#,
        )
        .bind(pubkey_hex)
        .bind(crate::CLASSIFICATION_EPOCH as i64)
        .fetch_one(&self.db.pool)
        .await
        .ok()
        .is_some_and(|c| c > 0);

        if !has_classification {
            return None;
        }

        let profile = self.db.get_profile_by_pubkey(pubkey_hex).await.ok()??;

        // Use the raw kind 0 event stored on the profiles table
        if let Some(ref json) = profile.metadata_json {
            if let Ok(event) = Event::from_json(json) {
                return Some(event);
            }
        }

        None
    }
}

impl NostrDatabase for SearchDatabase {
    fn backend(&self) -> Backend {
        Backend::Custom("nostr-classify-fts".to_string())
    }

    fn save_event<'a>(
        &'a self,
        _event: &'a Event,
    ) -> BoxedFuture<'a, Result<SaveEventStatus, DatabaseError>> {
        Box::pin(async move {
            // Read-only: reject all writes
            Ok(SaveEventStatus::Rejected(nostr_database::RejectedReason::Other))
        })
    }

    fn check_id<'a>(
        &'a self,
        _event_id: &'a EventId,
    ) -> BoxedFuture<'a, Result<DatabaseEventStatus, DatabaseError>> {
        Box::pin(async move { Ok(DatabaseEventStatus::NotExistent) })
    }

    fn event_by_id<'a>(
        &'a self,
        _event_id: &'a EventId,
    ) -> BoxedFuture<'a, Result<Option<Event>, DatabaseError>> {
        Box::pin(async move { Ok(None) })
    }

    fn count(&self, filter: Filter) -> BoxedFuture<'_, Result<usize, DatabaseError>> {
        Box::pin(async move {
            let events = self.query(filter).await?;
            Ok(events.len())
        })
    }

    fn query(&self, filter: Filter) -> BoxedFuture<'_, Result<Events, DatabaseError>> {
        Box::pin(async move {
            // Only serve kind 0 (Metadata) events
            let kinds = filter.kinds.as_ref();
            if let Some(kinds) = kinds {
                if !kinds.contains(&Kind::Metadata) && !kinds.is_empty() {
                    return Ok(Events::new(&filter));
                }
            }

            let limit = filter.limit.unwrap_or(100);
            let authors = filter.authors.as_ref();
            let search = filter.search.as_ref();

            let nostr_events = self.query_profiles(authors, search, limit).await?;

            // Filter events against the full filter
            let mut events = Events::new(&filter);
            for event in nostr_events {
                if filter.match_event(&event, MatchEventOptions::new()) {
                    events.insert(event);
                }
            }

            Ok(events)
        })
    }

    fn delete(&self, _filter: Filter) -> BoxedFuture<'_, Result<(), DatabaseError>> {
        Box::pin(async move {
            Err(DatabaseError::Backend(Box::new(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "read-only database",
            ))))
        })
    }

    fn wipe(&self) -> BoxedFuture<'_, Result<(), DatabaseError>> {
        Box::pin(async move {
            Err(DatabaseError::Backend(Box::new(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "read-only database",
            ))))
        })
    }
}
