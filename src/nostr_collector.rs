use crate::config::Config;
use crate::count_cache::{EventCountCache, FollowerCache};
use crate::db::Database;
use crate::job_queue::Job;
use crate::nostr_client::NostrClient;
use anyhow::Result;
use nostr_sdk::prelude::*;
use std::sync::Arc;

/// Number of events to accumulate before flushing to DB.
const BATCH_SIZE: usize = 50;
/// Max time (ms) to hold a batch before flushing, even if under BATCH_SIZE.
const FLUSH_INTERVAL_MS: u64 = 2000;

pub struct NostrCollector;

impl NostrCollector {
    pub async fn run(
        &self,
        db: Arc<Database>,
        nostr: NostrClient,
        job_queue: Arc<crate::job_queue::JobQueue>,
        config: &Config,
    ) -> Result<()> {
        let client = nostr.client();

        let kinds: Vec<Kind> = crate::db::CLASSIFIABLE_KINDS.iter().map(|k| Kind::from(*k)).collect();

        let event_counts = EventCountCache::new(db.clone());
        let follower_cache = FollowerCache::new(db.clone(), nostr.clone());

        loop {
            let filter = Filter::new()
                .kinds(kinds.clone())
                .since(Timestamp::now());

            if let Err(e) = client.subscribe(filter, None).await {
                tracing::error!("Failed to subscribe: {}, retrying in 5s", e);
                tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                continue;
            }

            tracing::info!("Subscribed to classifiable event kinds {:?}", crate::db::CLASSIFIABLE_KINDS);

            let mut rx = client.notifications();
            let mut batch: Vec<Event> = Vec::with_capacity(BATCH_SIZE);
            let mut flush_tick = tokio::time::interval(std::time::Duration::from_millis(FLUSH_INTERVAL_MS));
            flush_tick.tick().await; // consume initial tick

            loop {
                tokio::select! {
                    notification = rx.recv() => {
                        match notification {
                            Ok(RelayPoolNotification::Event { event, .. }) => {
                                batch.push(*event);
                                if batch.len() >= BATCH_SIZE {
                                    Self::flush_batch(
                                        &batch, &db, &job_queue, &config, &event_counts, &follower_cache,
                                    ).await;
                                    batch.clear();
                                }
                            }
                            Ok(_) => {} // ignore non-event notifications
                            Err(_) => break, // stream closed
                        }
                    }
                    _ = flush_tick.tick() => {
                        if !batch.is_empty() {
                            Self::flush_batch(
                                &batch, &db, &job_queue, &config, &event_counts, &follower_cache,
                            ).await;
                            batch.clear();
                        }
                    }
                }
            }

            tracing::warn!("Notification stream closed, reconnecting in 5s...");
            // Flush any remaining events before reconnecting
            if !batch.is_empty() {
                Self::flush_batch(
                    &batch, &db, &job_queue, &config, &event_counts, &follower_cache,
                ).await;
            }
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
        }
    }

    /// Flush a batch of events to the DB, update in-memory counts, and enqueue
    /// any profiles that cross the threshold.
    async fn flush_batch(
        events: &[Event],
        db: &Arc<Database>,
        job_queue: &Arc<crate::job_queue::JobQueue>,
        config: &Config,
        event_counts: &EventCountCache,
        follower_cache: &FollowerCache,
    ) {
        // 1. Batch-write events to DB, get net new count per pubkey
        let net_new = match db.cache_events_batch(events).await {
            Ok(n) => n,
            Err(e) => {
                tracing::error!("Failed to cache event batch: {}", e);
                return;
            }
        };

        // 2. Update in-memory counts and collect pubkeys that cross the threshold
        let mut to_check: Vec<(String, usize)> = Vec::new();
        for (pubkey, added) in &net_new {
            if *added == 0 {
                continue; // replaceable event that updated an existing row — no net change
            }
            let total = event_counts.increment(pubkey, *added as usize).await;
            to_check.push((pubkey.clone(), total));
        }

        // 4. For profiles crossing threshold, check classification status and follower count
        for (pubkey, count) in to_check {
            if count < config.processing.event_threshold {
                continue;
            }

            let is_current = match db.has_current_classification(&pubkey, crate::CLASSIFICATION_EPOCH).await {
                Ok(c) => c,
                Err(e) => {
                    tracing::error!("Failed to check classification status for {}: {}", pubkey, e);
                    continue;
                }
            };

            if is_current {
                continue;
            }

            // Follower check — spawn so we don't block the collector loop
            let follower_cache = follower_cache.clone();
            let job_queue = Arc::clone(job_queue);
            let min_followers = config.processing.min_followers;
            let event_count = count;

            tokio::spawn(async move {
                let meets_threshold = if min_followers > 0 {
                    match follower_cache.get(&pubkey).await {
                        Some(c) => c >= min_followers,
                        None => false, // couldn't determine — skip, will retry next event
                    }
                } else {
                    true
                };

                if meets_threshold {
                    let job = Job { pubkey: pubkey.clone(), retry_count: 0 };
                    if job_queue.enqueue(job).await.unwrap_or(false) {
                        tracing::info!(
                            "Queued profile {} for processing ({} events)",
                            pubkey, event_count
                        );
                    }
                } else if min_followers > 0 {
                    tracing::debug!(
                        "Skipping profile {} — below min_followers threshold (min: {})",
                        pubkey, min_followers
                    );
                }
            });
        }
    }
}
