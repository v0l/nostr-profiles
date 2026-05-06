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

        // Subscribe only to classifiable event kinds from now onwards
        let kinds: Vec<Kind> = CLASSIFIABLE_KINDS.iter().map(|k| Kind::from(*k)).collect();
        let filter = Filter::new()
            .kinds(kinds)
            .since(Timestamp::now());

        client
            .subscribe(filter.clone(), None)
            .await?;

        tracing::info!("Subscribed to classifiable event kinds {:?} on relays", CLASSIFIABLE_KINDS);

        // Process incoming events using notifications stream
        let mut rx = client.notifications();
        
        while let Ok(notification) = rx.recv().await {
            if let RelayPoolNotification::Event { event, .. } = notification {
                // Only cache classifiable kinds — skip everything else
                if !crate::db::is_classifiable_kind(event.kind.as_u16()) {
                    continue;
                }

                // Store the event
                db.cache_event(&event).await?;

                // Check if this pubkey has enough events to trigger classification
                let pubkey_hex = event.pubkey.to_hex();
                let (event_count, is_classified) = db.get_profile(&pubkey_hex).await?;

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

        Ok(())
    }
}
