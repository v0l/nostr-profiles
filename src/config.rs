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
    pub labels: LabelsConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    pub api_base_url: String,
    pub model: String,
    pub api_key: String,
    /// Per-request HTTP timeout in seconds (default: 120)
    pub timeout_secs: u64,
    /// Overall classification timeout in seconds, covering all LLM calls + tool calls for a single job (default: 300)
    pub classify_timeout_secs: u64,
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
    /// Timeout in seconds for a single classification job (profile fetch + classify + save) (default: 600)
    pub job_timeout_secs: u64,
    /// Timeout in seconds for individual LLM tool calls (get_event, get_profile, etc.) (default: 30)
    pub tool_call_timeout_secs: u64,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelsConfig {
    /// Path to a text file with one label per line, or null to use the built-in set.
    pub taxonomy_file: Option<String>,
    /// Minimum score (0.0–1.0) for a label to be included in classification output.
    pub min_score: f64,
}

impl Config {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, config::ConfigError> {
        let config = config::Config::builder()
            .set_default("llm.timeout_secs", 120)?
            .set_default("llm.classify_timeout_secs", 300)?
            .set_default("processing.min_followers", 1)?
            .set_default("processing.job_timeout_secs", 600)?
            .set_default("processing.tool_call_timeout_secs", 30)?
            .set_default("labels.min_score", 0.4)?
            .add_source(config::File::from(path.as_ref()))
            .build()?;
        
        config.try_deserialize()
    }
}

/// Built-in label taxonomy covering common Nostr community topics.
/// Labels use lowercase kebab-case for consistency.
pub const BUILTIN_LABEL_TAXONOMY: &[&str] = &[
    // Technology & Development
    "software-developer",
    "rust",
    "python",
    "javascript",
    "golang",
    "web-development",
    "mobile-development",
    "devops",
    "open-source",
    "linux",
    "self-hosting",
    "ai-ml",
    "cryptography",
    "hardware",
    "3d-printing",

    // Bitcoin & Crypto
    "bitcoin",
    "bitcoin-mining",
    "lightning-network",
    "nostr-developer",
    "nostr-enthusiast",
    "altcoin",
    "defi",
    "trading",

    // Privacy & Freedom
    "privacy-advocate",
    "censorship-resistance",
    "free-speech",
    "cypherpunk",
    "decentralization",

    // Content Creation
    "writer",
    "blogger",
    "podcaster",
    "musician",
    "artist",
    "photographer",
    "video-creator",
    "memer",

    // Professional
    "entrepreneur",
    "investor",
    "educator",
    "researcher",
    "consultant",
    "designer",

    // Lifestyle & Interests
    "gaming",
    "fitness",
    "food",
    "cooking",
    "coffee",
    "beer",
    "wine",
    "travel",
    "sports",
    "nature",
    "reading",
    "philosophy",
    "religion",
    "spirituality",
    "parenting",
    "diy",
    "gardening",
    "music",
    "pets",
    "animals",
    "fashion",
    "home-improvement",
    "motorcycles",
    "cars",
    "boating",
    "fishing",
    "hunting",
    "hiking",
    "camping",
    "surfing",
    "skiing",
    "yoga",
    "meditation",
    "mental-health",
    "weather",

    // Politics & Society
    "libertarian",
    "anarchist",
    "politics",
    "activism",
    "prepper",

    // Community
    "community-builder",
    "event-organizer",
    "moderator",

    // Content Quality
    "nsfw",
    "bot",
    "spam",
];

/// Load the label taxonomy. If `taxonomy_file` is set, read labels from that file
/// (one per line, blank lines and lines starting with # are skipped).
/// Otherwise returns the built-in set.
pub fn load_label_taxonomy(taxonomy_file: Option<&str>) -> Vec<String> {
    if let Some(path) = taxonomy_file {
        if let Ok(contents) = std::fs::read_to_string(path) {
            let labels: Vec<String> = contents
                .lines()
                .map(|l| l.trim())
                .filter(|l| !l.is_empty() && !l.starts_with('#'))
                .map(|l| l.to_string())
                .collect();
            if !labels.is_empty() {
                return labels;
            }
            tracing::warn!("Taxonomy file {} was empty or had no valid labels, using built-in set", path);
        } else {
            tracing::warn!("Could not read taxonomy file {}, using built-in set", path);
        }
    }
    BUILTIN_LABEL_TAXONOMY.iter().map(|s| s.to_string()).collect()
}
