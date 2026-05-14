use serde::{Deserialize, Serialize};
use std::path::Path;

/// A single LLM call entry in the chat log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatLogEntry {
    /// Type of LLM call: "classify", "describe_image", or "summarize".
    pub call_type: String,
    /// Ordering index within this classification run.
    pub call_index: usize,
    /// The full request (CreateChatCompletionRequest) serialized to JSON.
    pub request: serde_json::Value,
    /// The full response (CreateChatCompletionResponse) serialized to JSON.
    pub response: serde_json::Value,
}

/// Complete chat log for one classification run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatLog {
    pub pubkey: String,
    pub timestamp: String,
    pub entries: Vec<ChatLogEntry>,
}

/// Accumulates chat log entries during a classification run.
pub struct ChatLogCollector {
    log: ChatLog,
    counter: usize,
}

impl ChatLogCollector {
    pub fn new(pubkey: &str) -> Self {
        Self {
            log: ChatLog {
                pubkey: pubkey.to_string(),
                timestamp: chrono::Utc::now().to_rfc3339(),
                entries: Vec::new(),
            },
            counter: 0,
        }
    }

    /// Record an LLM call. `request` and `response` are the raw chat completion
    /// request and response, serialized to JSON.
    pub fn record(
        &mut self,
        call_type: &str,
        request: serde_json::Value,
        response: serde_json::Value,
    ) {
        let entry = ChatLogEntry {
            call_type: call_type.to_string(),
            call_index: self.counter,
            request,
            response,
        };
        self.counter += 1;
        self.log.entries.push(entry);
    }

    pub fn finalize(self) -> ChatLog {
        self.log
    }
}

/// Write a ChatLog to disk as a JSON file.
/// Path: {dir}/{pubkey}/{timestamp}.json
pub async fn write_log(log: &ChatLog, dir: &str) -> Result<(), std::io::Error> {
    let path = Path::new(dir).join(&log.pubkey);
    tokio::fs::create_dir_all(&path).await?;

    // Sanitize timestamp for use as a filename
    let ts = log.timestamp.replace(':', "-");
    let file_path = path.join(format!("{}.json", ts));

    let json = serde_json::to_string_pretty(log)?;
    tokio::fs::write(&file_path, json).await?;

    tracing::info!("Wrote chat log to {}", file_path.display());
    Ok(())
}
