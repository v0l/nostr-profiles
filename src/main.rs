mod asr;
mod chat_log;
mod classifier;
mod config;
mod count_cache;
mod db;
mod e2e_test;
mod format;
mod http_server;
mod image_cache;
mod job_queue;
mod nostr_client;
mod nostr_collector;
mod opengraph;
mod profile_cache;
mod search_relay;
mod video;

use crate::classifier::Classifier;
use crate::config::Config;
use crate::db::Database;
use crate::image_cache::ImageCache;
use crate::job_queue::{JobQueue, Job};
use crate::nostr_client::NostrClient;
use crate::profile_cache::ProfileCache;
use anyhow::Result;
use std::sync::Arc;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

/// Classification epoch. Increment this when the system prompt, taxonomy,
/// classification flow, or any other factor changes enough that all previously
/// classified profiles should be re-processed.
pub const CLASSIFICATION_EPOCH: u32 = 5;

#[tokio::main]
async fn main() -> Result<()> {
    rustls::crypto::ring::default_provider()
        .install_default()
        .expect("Failed to install rustls crypto provider");

    let config = Config::load("config.yaml")?;
    
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(&config.logging.level));
    
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .with(filter)
        .init();

    tracing::info!("Starting Nostr Profile Classifier");

    let db = Arc::new(Database::new(&config.database.path, config.labels.min_score).await?);
    tracing::info!("Database initialized at {}", config.database.path);

    let nostr = NostrClient::new(&config.nostr.relays, config.nostr.nsec.as_deref()).await?;
    nostr.connect().await;
    tracing::info!("Nostr client connected to {} relays", config.nostr.relays.len());

    let image_cache = ImageCache::new(&config.image_cache.dir, config.processing.asr.clone())?;
    tracing::info!("Image cache initialized at {}", config.image_cache.dir);

    let profile_cache = ProfileCache::new(db.clone(), nostr.clone(), 7);

    let og_cache = crate::opengraph::OpenGraphCache::new(2048);

    let classifier = Classifier::new(
        &config.llm,
        nostr.clone(),
        profile_cache,
        image_cache.clone(),
        og_cache,
        db.clone(),
        crate::config::load_label_taxonomy(config.labels.taxonomy_file.as_deref()),
        config.labels.min_score,
        std::time::Duration::from_secs(config.processing.tool_call_timeout_secs),
        config.chat_logs.dir.clone(),
    );
    tracing::info!("Classifier initialized with model {} and {} labels (min_score={})",
        config.llm.model,
        crate::config::load_label_taxonomy(config.labels.taxonomy_file.as_deref()).len(),
        config.labels.min_score
    );

    let job_queue = Arc::new(JobQueue::new(config.processing.max_workers, config.processing.max_retries, config.processing.cache_days, std::time::Duration::from_secs(config.processing.job_timeout_secs), config.processing.classification_event_limit));
    tracing::info!("Job queue initialized with {} workers", config.processing.max_workers);

    // Build the search relay backed by our FTS index
    let relay = search_relay::SearchDatabase::build_relay(db.clone());

    // Start HTTP server with the search relay on the same port
    let db_clone = Arc::clone(&db);
    let job_queue_clone = Arc::clone(&job_queue);
    let db_path = config.database.path.clone();
    let chat_log_dir = config.chat_logs.dir.clone();
    let server_handle = tokio::spawn(async move {
        http_server::serve(db_clone, relay, job_queue_clone, 3000, db_path, chat_log_dir).await;
    });

    // Start job queue workers BEFORE enqueuing — otherwise the channel fills
    // up and enqueue blocks forever with no consumers
    let job_queue_clone = Arc::clone(&job_queue);
    let db_clone = Arc::clone(&db);
    let classifier_clone = classifier.clone();
    let image_cache_clone = image_cache.clone();
    let processor_handle = tokio::spawn(async move {
        job_queue_clone.run(db_clone, classifier_clone, image_cache_clone).await;
    });

    // Enqueue profiles that need classification (new or stale epoch)
    {
        let unclassified = db.get_profiles_needing_classification(config.processing.event_threshold as i64, CLASSIFICATION_EPOCH).await?;
        if !unclassified.is_empty() {
            tracing::info!("Found {} profiles needing classification, enqueuing...", unclassified.len());
            let mut queued = 0usize;
            for pubkey in &unclassified {
                if job_queue.enqueue(Job { pubkey: pubkey.clone(), retry_count: 0 }).await? {
                    queued += 1;
                }
            }
            tracing::info!("Enqueued {} profiles for classification ({} already in queue)", queued, unclassified.len() - queued);
        }
    }

    // Periodic per-pubkey event retention sweep: caps the events table by
    // count-per-pubkey so it can't grow unbounded, while keeping all events for
    // infrequent posters so they can still reach the classification threshold.
    let retention_db = Arc::clone(&db);
    let max_per_pubkey = config.processing.max_events_per_pubkey as i64;
    let retention_interval = config.processing.event_retention_interval_secs;
    let _retention_handle = tokio::spawn(async move {
        let mut tick = tokio::time::interval(std::time::Duration::from_secs(retention_interval.max(1)));
        loop {
            tick.tick().await;
            match retention_db.retain_events_per_pubkey(max_per_pubkey).await {
                Ok(0) => {}
                Ok(deleted) => tracing::info!(
                    "Event retention sweep: deleted {} events over the per-pubkey cap of {}",
                    deleted, max_per_pubkey
                ),
                Err(e) => tracing::error!("Event retention sweep failed: {}", e),
            }
        }
    });

    let collector = crate::nostr_collector::NostrCollector;
    let collector_handle = tokio::spawn(async move {
        if let Err(e) = collector.run(db, nostr, Arc::clone(&job_queue), &config).await {
            tracing::error!("Collector error: {}", e);
        }
    });

    tokio::select! {
        _ = collector_handle => { tracing::warn!("Collector stopped"); }
        _ = processor_handle => { tracing::warn!("Processor stopped"); }
        _ = server_handle => { tracing::warn!("Server stopped"); }
    }

    Ok(())
}
