mod config;
mod db;
mod format;
mod http_server;
mod image_cache;
mod job_queue;
mod llm_client;
mod nostr_client;
mod nostr_collector;

use crate::config::Config;
use crate::db::Database;
use crate::image_cache::ImageCache;
use crate::job_queue::JobQueue;
use crate::llm_client::LLMClient;
use crate::nostr_client::NostrClient;
use anyhow::Result;
use std::sync::Arc;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

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

    let db = Arc::new(Database::new(&config.database.path).await?);
    tracing::info!("Database initialized at {}", config.database.path);

    let nostr = NostrClient::new(&config.nostr.relays, config.nostr.nsec.as_deref(), db.clone()).await?;
    nostr.connect().await;
    tracing::info!("Nostr client connected to {} relays", config.nostr.relays.len());

    let image_cache = ImageCache::new(&config.image_cache.dir)?;
    tracing::info!("Image cache initialized at {}", config.image_cache.dir);

    let llm = LLMClient::new(&config.llm, nostr.clone());
    tracing::info!("LLM client initialized with model {}", config.llm.model);

    let job_queue = Arc::new(JobQueue::new(config.processing.max_workers, config.processing.cache_days));
    tracing::info!("Job queue initialized with {} workers", config.processing.max_workers);

    // Start HTTP server for dashboard
    let db_clone = Arc::clone(&db);
    let server_handle = tokio::spawn(async move {
        http_server::serve(db_clone, 3000).await;
    });

    let job_queue_clone = Arc::clone(&job_queue);
    let db_clone = Arc::clone(&db);
    let llm_clone = llm.clone();
    let image_cache_clone = image_cache.clone();
    let processor_handle = tokio::spawn(async move {
        job_queue_clone.run(db_clone, llm_clone, image_cache_clone).await;
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
