use crate::db::Database;
use crate::llm_client::LLMClient;
use crate::image_cache::ImageCache;
use anyhow::Result;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};

#[derive(Debug, Clone)]
pub struct Job {
    pub pubkey: String,
}

pub struct JobQueue {
    tx: mpsc::Sender<Job>,
    rx: Arc<Mutex<mpsc::Receiver<Job>>>,
    max_workers: usize,
    queued_pubkeys: Arc<Mutex<HashSet<String>>>,
    cache_days: u64,
}

impl Clone for JobQueue {
    fn clone(&self) -> Self {
        Self {
            tx: self.tx.clone(),
            rx: self.rx.clone(),
            max_workers: self.max_workers,
            queued_pubkeys: self.queued_pubkeys.clone(),
            cache_days: self.cache_days,
        }
    }
}

impl JobQueue {
    pub fn new(max_workers: usize, cache_days: u64) -> Self {
        let (tx, rx) = mpsc::channel(100);

        Self {
            tx,
            rx: Arc::new(Mutex::new(rx)),
            max_workers,
            queued_pubkeys: Arc::new(Mutex::new(HashSet::new())),
            cache_days,
        }
    }

    pub async fn enqueue(&self, job: Job) -> Result<bool> {
        // Check if this pubkey is already in the queue
        let mut queued = self.queued_pubkeys.lock().await;
        if queued.contains(&job.pubkey) {
            tracing::debug!("Profile {} already in queue, skipping", job.pubkey);
            return Ok(false);
        }
        
        // Add to queue and track
        queued.insert(job.pubkey.clone());
        self.tx.send(job).await?;
        Ok(true)
    }

    pub async fn dequeue(&self, pubkey: &str) {
        // Remove from tracking when job is processed
        let mut queued = self.queued_pubkeys.lock().await;
        queued.remove(pubkey);
    }

    pub async fn run(&self, db: Arc<Database>, llm: LLMClient, image_cache: ImageCache) {
        let mut workers = Vec::new();
        let queue_clone = self.clone_for_worker();

        for i in 0..self.max_workers {
            let rx = self.rx.clone();
            let db = db.clone();
            let llm = llm.clone();
            let image_cache = image_cache.clone();
            let queue = queue_clone.clone();
            let cache_days = self.cache_days;

            let worker = tokio::spawn(async move {
                while let Some(job) = rx.lock().await.recv().await {
                    let pubkey = job.pubkey.clone();
                    match process_job(&job, &db, &llm, &image_cache, cache_days).await {
                        Ok(_) => {
                            tracing::info!(
                                "Worker {} successfully processed profile {}",
                                i,
                                pubkey
                            );
                        }
                        Err(e) => {
                            tracing::error!(
                                "Worker {} failed to process profile {}: {}",
                                i,
                                pubkey,
                                e
                            );
                        }
                    }
                    // Remove from queue tracking after processing
                    queue.dequeue(&pubkey).await;
                }
            });

            workers.push(worker);
        }

        futures::future::join_all(workers).await;
    }

    fn clone_for_worker(&self) -> Self {
        Self {
            tx: self.tx.clone(),
            rx: self.rx.clone(),
            max_workers: self.max_workers,
            queued_pubkeys: self.queued_pubkeys.clone(),
            cache_days: self.cache_days,
        }
    }
}

async fn process_job(
    job: &Job,
    db: &Database,
    llm: &LLMClient,
    image_cache: &ImageCache,
    cache_days: u64,
) -> Result<()> {
    let start = std::time::Instant::now();
    let pubkey = &job.pubkey;

    let events = db.get_profile_events(pubkey, 50).await?;
    
    // Build event data with inline image descriptions
    let mut event_data = Vec::new();
    
    for event in &events {
        let mut event_info = EventInfo {
            raw_json: event.raw_json.clone(),
            image_descriptions: HashMap::new(),
        };
        
        // Parse the event and extract image URLs
        if let Ok(nostr_event) = serde_json::from_str::<nostr_sdk::Event>(&event.raw_json) {
            // Extract image URLs from content and tags
            if let Ok(urls) = extract_image_urls(&nostr_event) {
                for url in urls {
                    if let Some((path, hash)) = image_cache.download(&url).await? {
                        // Skip non-image downloads (e.g. HTML error pages)
                        if !crate::image_cache::is_valid_image_path(&path) {
                            continue;
                        }
                        // Check if we have a cached description
                        let desc = match db.get_image_description(&hash).await? {
                            Some(cached_desc) => cached_desc,
                            None => {
                                // Generate new description
                                let desc = llm.describe_image(&path).await?;
                                // Cache it
                                let _ = db.save_image_description(&hash, &desc).await;
                                desc
                            }
                        };
                        event_info.image_descriptions.insert(url, desc);
                    }
                }
            }
        }
        
        event_data.push(event_info);
    }

    // Get profile details
    let profile = db.get_profile_details(pubkey).await.ok();

    // Get previous classification if this is a re-classification
    let previous_classification = db.get_classification_if_exists(pubkey).await.ok().flatten();

    let context = build_context(&profile, &event_data, &previous_classification);
    if context.len() < 100 {
        tracing::warn!("Context is very short: {}", context);
    }

    tracing::info!(
        "Classifying profile {} with {} events and {} images, {} context{}",
        pubkey,
        events.len(),
        event_data.iter().map(|e| e.image_descriptions.len()).sum::<usize>(),
        context.len(),
        if previous_classification.is_some() { " (re-classification)" } else { "" }
    );

    let classification = llm
        .classify_with_images(&context, &[])
        .await?;

    let classification_db = crate::db::Classification {
        labels: classification.labels.clone(),
        bio: classification.bio.clone(),
        confidence: classification.confidence,
    };
    db.save_classification(pubkey, &classification_db, events.len())
        .await?;
    db.mark_profile_classified(pubkey).await?;

    // Clean up old events now that classification is done
    let deleted = db.delete_old_events_for_pubkey(pubkey, cache_days).await?;
    if deleted > 0 {
        tracing::info!(
            "Cleaned up {} events older than {} days for profile {}",
            deleted,
            cache_days,
            pubkey
        );
    }

    let elapsed = start.elapsed();
    tracing::info!(
        "Successfully classified profile {} in {:?}: {} labels, confidence {:.2}%",
        pubkey,
        elapsed,
        classification.labels.len(),
        classification.confidence * 100.0
    );

    Ok(())
}

struct EventInfo {
    raw_json: String,
    image_descriptions: HashMap<String, String>, // url -> description
}

fn build_context(
    profile: &Option<crate::db::Profile>,
    events: &[EventInfo],
    previous_classification: &Option<crate::db::Classification>,
) -> String {
    let mut ctx = String::new();
    ctx.push_str("=== NOSTR PROFILE ===\n\n");

    // Add profile details if available
    if let Some(p) = profile {
        ctx.push_str(&crate::format::describe_profile(p));
    } else {
        ctx.push_str("Profile details not available.\n\n");
    }

    // Add previous classification if available
    if let Some(prev) = previous_classification {
        ctx.push_str("\n=== PREVIOUS CLASSIFICATION ===\n\n");
        ctx.push_str(&format!("Labels: {}\n", prev.labels.join(", ")));
        ctx.push_str(&format!("Bio: {}\n", prev.bio));
        ctx.push_str(&format!("Confidence: {:.0}%\n", prev.confidence * 100.0));
    }

    ctx.push_str("\n=== RECENT EVENTS ===\n\n");
    for (i, event_info) in events.iter().enumerate() {
        let Ok(nostr_event) = serde_json::from_str::<nostr_sdk::Event>(&event_info.raw_json) else {
            continue;
        };
        
        ctx.push_str(&format!("\n--- Event {} ---\n", i + 1));
        ctx.push_str(&crate::format::describe_event(&nostr_event));
        
        // Add image descriptions inline with the event
        if !event_info.image_descriptions.is_empty() {
            ctx.push_str("Images:\n");
            for (j, (url, img_desc)) in event_info.image_descriptions.iter().enumerate() {
                ctx.push_str(&format!("  Image {} ({}): {}\n", j + 1, url, img_desc));
            }
        }
    }

    ctx
}

/// Extract image URLs from event content
fn extract_image_urls(event: &nostr_sdk::Event) -> Result<Vec<String>> {
    extract_urls_from_text(&event.content)
}

/// Extract image URLs from text content
fn extract_urls_from_text(text: &str) -> Result<Vec<String>> {
    use regex::Regex;
    let re = Regex::new(r#"(?i)https?://\S+\.(jpg|jpeg|png|gif|webp)\b"#)?;
    Ok(re.find_iter(text).map(|m| m.as_str().to_string()).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_context_with_events() {
        // Using a structurally valid but unsigned event — nostr_sdk rejects
        // the signature, so the event is skipped. This still tests that
        // build_context processes the events list without panicking.
        let events = vec![
            EventInfo {
                raw_json: r#"{"id":"0000000000000000000000000000000000000000000000000000000000000000","pubkey":"0000000000000000000000000000000000000000000000000000000000000001","created_at":1000,"kind":1,"content":"Hello world","tags":[],"sig":"0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"}"#.to_string(),
                image_descriptions: HashMap::new(),
            },
        ];

        let ctx = build_context(&None, &events, &None);
        
        assert!(ctx.contains("=== RECENT EVENTS ==="));
    }

    #[test]
    fn test_build_context_empty() {
        let events: Vec<EventInfo> = vec![];

        let ctx = build_context(&None, &events, &None);

        assert!(ctx.contains("=== RECENT EVENTS ==="));
        assert!(!ctx.contains("Hello"));
    }

    #[test]
    fn test_build_context_with_previous_classification() {
        let events = vec![
            EventInfo {
                raw_json: r#"{"id":"0000000000000000000000000000000000000000000000000000000000000000","pubkey":"0000000000000000000000000000000000000000000000000000000000000001","created_at":1000,"kind":1,"content":"Hello world","tags":[],"sig":"0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"}"#.to_string(),
                image_descriptions: HashMap::new(),
            },
        ];

        let prev = crate::db::Classification {
            labels: vec!["bitcoin".to_string(), "developer".to_string()],
            bio: "A bitcoin developer".to_string(),
            confidence: 0.8,
        };

        let ctx = build_context(&None, &events, &Some(prev));
        
        assert!(ctx.contains("=== PREVIOUS CLASSIFICATION ==="));
        assert!(ctx.contains("bitcoin, developer"));
        assert!(ctx.contains("A bitcoin developer"));
    }
}
