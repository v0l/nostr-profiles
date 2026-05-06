use crate::config::Config;
use crate::db::{Database, CLASSIFIABLE_KINDS};
use crate::job_queue::Job;
use crate::nostr_client::NostrClient;
use anyhow::Result;
use nostr_sdk::prelude::*;
use std::sync::Arc;

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

        let kinds: Vec<Kind> = CLASSIFIABLE_KINDS.iter().map(|k| Kind::from(*k)).collect();

        loop {
            let filter = Filter::new()
                .kinds(kinds.clone())
                .since(Timestamp::now());

            if let Err(e) = client.subscribe(filter, None).await {
                tracing::error!("Failed to subscribe: {}, retrying in 5s", e);
                tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                continue;
            }

            tracing::info!("Subscribed to classifiable event kinds {:?}", CLASSIFIABLE_KINDS);

            let mut rx = client.notifications();

            while let Ok(notification) = rx.recv().await {
                if let RelayPoolNotification::Event { event, .. } = notification {
                    // Only cache classifiable kinds — skip everything else
                    if !crate::db::is_classifiable_kind(event.kind.as_u16()) {
                        continue;
                    }

                    // Store the event
                    if let Err(e) = db.cache_event(&event).await {
                        tracing::error!("Failed to cache event: {}", e);
                        continue;
                    }

                    // Check if this pubkey has enough events to trigger classification
                    let pubkey_hex = event.pubkey.to_hex();
                    let (event_count, is_classified) = match db.get_profile(&pubkey_hex).await {
                        Ok(r) => r,
                        Err(e) => {
                            tracing::error!("Failed to get profile for {}: {}", pubkey_hex, e);
                            continue;
                        }
                    };

                    if event_count >= config.processing.event_threshold && !is_classified {
                        // Check minimum follower count to filter out bots / test accounts
                        if config.processing.min_followers > 0 {
                            let cached_count = db.get_profile_by_pubkey(&pubkey_hex)
                                .await?
                                .and_then(|p| p.follower_count);

                            let follower_count = match cached_count {
                                Some(c) => c as usize,
                                None => {
                                    // Not cached — fetch from relays and cache the result
                                    match nostr.fetch_follower_count(&pubkey_hex, 5).await {
                                        Ok(count) => {
                                            if let Err(e) = db.set_follower_count(&pubkey_hex, count).await {
                                                tracing::warn!("Failed to cache follower count for {}: {}", pubkey_hex, e);
                                            }
                                            count
                                        }
                                        Err(e) => {
                                            tracing::warn!(
                                                "Failed to fetch follower count for {}: {}, skipping check",
                                                pubkey_hex,
                                                e
                                            );
                                            // Can't determine — skip the check, let it through
                                            continue;
                                        }
                                    }
                                }
                            };

                            if follower_count < config.processing.min_followers {
                                tracing::debug!(
                                    "Skipping profile {} — only {} followers (min: {})",
                                    pubkey_hex,
                                    follower_count,
                                    config.processing.min_followers
                                );
                                continue;
                            }
                        }

                        let job = Job {
                            pubkey: pubkey_hex.clone(),
                        };

                        if job_queue.enqueue(job).await? {
                            tracing::info!(
                                "Queued profile {} for processing ({} events)",
                                pubkey_hex,
                                event_count
                            );
                        } else {
                            tracing::debug!(
                                "Profile {} already queued, skipping",
                                pubkey_hex
                            );
                        }
                    }
                }
            }

            // If we get here, the notification stream closed (relay disconnect, etc.)
            tracing::warn!("Notification stream closed, reconnecting in 5s...");
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
        }
    }
}
