use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub llm: LlmConfig,
    pub nostr: NostrConfig,
    pub processing: ProcessingConfig,
    pub database: DatabaseConfig,
    pub image_cache: ImageCacheConfig,
    pub logging: LoggingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    pub api_base_url: String,
    pub model: String,
    pub api_key: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NostrConfig {
    pub relays: Vec<String>,
    pub nsec: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingConfig {
    pub event_threshold: usize,
    pub cache_days: u64,
    pub max_workers: usize,
    pub max_retries: u8,
    pub image_download_timeout_secs: u64,
    pub min_followers: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    pub path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageCacheConfig {
    pub dir: String,
    pub cleanup_days: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub level: String,
}

impl Config {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, config::ConfigError> {
        let config = config::Config::builder()
            .set_default("processing.min_followers", 1)?
            .add_source(config::File::from(path.as_ref()))
            .build()?;
        
        config.try_deserialize()
    }
}
