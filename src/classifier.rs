use crate::config::LlmConfig;
use crate::db::Database;
use crate::image_cache::ImageCache;
use crate::nostr_client::NostrClient;
use crate::opengraph::OpenGraphCache;
use crate::profile_cache::ProfileCache;
use thiserror::Error;
use nostr_sdk::JsonUtil;
use async_openai::{
    config::OpenAIConfig,
    types::chat::{
        ChatCompletionRequestMessage, ChatCompletionRequestUserMessage,
        ChatCompletionRequestUserMessageContent, CreateChatCompletionRequest,
    },
    Client,
};
use base64::{engine::general_purpose::STANDARD, Engine};
use image::GenericImageView;
use serde::{Deserialize, Serialize};
use async_openai::types::chat::{
    ChatCompletionRequestMessageContentPartImage, ChatCompletionRequestMessageContentPartText,
    ChatCompletionRequestUserMessageContentPart, ImageDetail, ImageUrl,
    ChatCompletionTools, ChatCompletionTool, FunctionObject,
    ChatCompletionRequestToolMessage, ChatCompletionRequestToolMessageContent,
    ChatCompletionRequestAssistantMessage,
    ChatCompletionMessageToolCalls,
};
use std::sync::Arc;
use std::time::Duration;
use tracing::info;

#[derive(Debug, Error)]
pub enum ClassifierError {
    #[error("LLM API error: {0}")]
    Llm(#[from] async_openai::error::OpenAIError),
    #[error("Classification timed out after {0}s")]
    Timeout(u64),
    #[error("Tool call '{tool}' timed out after {timeout}s")]
    ToolTimeout { tool: String, timeout: u64 },
    #[error("Tool call '{tool}' failed: {source}")]
    ToolFailed { tool: String, #[source] source: Box<dyn std::error::Error + Send + Sync> },
    #[error("Invalid tool arguments: {0}")]
    InvalidToolArgs(String),
    #[error("Unknown tool: {0}")]
    UnknownTool(String),
    #[error("Parse error: {0}")]
    Parse(String),
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),
    #[error("Image error: {0}")]
    Image(String),
    #[error("Nostr error: {0}")]
    Nostr(#[from] nostr_sdk::client::Error),
    #[error("{0}")]
    Other(String),
}

impl From<anyhow::Error> for ClassifierError {
    fn from(e: anyhow::Error) -> Self {
        ClassifierError::Other(e.to_string())
    }
}

impl ClassifierError {
    /// Returns true if this error is caused by an LLM server error (HTTP 5xx,
    /// connection failure, or timeout) that should trigger backoff.
    pub fn is_server_error(&self) -> bool {
        match self {
            ClassifierError::Llm(openai_err) => {
                match openai_err {
                    async_openai::error::OpenAIError::ApiError(_) => true,
                    async_openai::error::OpenAIError::Reqwest(req_err) => {
                        req_err.status().is_some_and(|s| s.is_server_error())
                            || req_err.is_connect()
                            || req_err.is_timeout()
                    }
                    _ => false,
                }
            }
            ClassifierError::Timeout(_) => true,
            ClassifierError::ToolTimeout { .. } => true,
            _ => false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Classification {
    pub labels: Vec<String>,
    pub scores: std::collections::HashMap<String, f64>,
    pub bio: String,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifyResult {
    pub classification: Classification,
    pub tool_event_counts: std::collections::HashMap<i64, usize>,
}

#[derive(Clone)]
pub struct Classifier {
    client: Client<OpenAIConfig>,
    model: String,
    nostr: NostrClient,
    profile_cache: ProfileCache,
    image_cache: ImageCache,
    og_cache: OpenGraphCache,
    db: Arc<Database>,
    label_taxonomy: Vec<String>,
    label_min_score: f64,
    classify_timeout: Duration,
    tool_call_timeout: Duration,
}

impl Classifier {
    pub fn profile_cache(&self) -> &ProfileCache {
        &self.profile_cache
    }

    pub fn new(config: &LlmConfig, nostr: NostrClient, profile_cache: ProfileCache, image_cache: ImageCache, og_cache: OpenGraphCache, db: Arc<Database>, label_taxonomy: Vec<String>, label_min_score: f64, tool_call_timeout: Duration) -> Self {
        let openai_config = OpenAIConfig::new()
            .with_api_base(&config.api_base_url)
            .with_api_key(&config.api_key);

        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .expect("failed to build reqwest client");

        let client = Client::with_config(openai_config)
            .with_http_client(http_client);

        let classify_timeout = Duration::from_secs(config.classify_timeout_secs);

        Self {
            client,
            model: config.model.clone(),
            nostr,
            profile_cache,
            image_cache,
            og_cache,
            db,
            label_taxonomy,
            label_min_score,
            classify_timeout,
            tool_call_timeout,
        }
    }

    pub async fn classify(
        &self,
        pubkey: &str,
        context: &str,
    ) -> Result<ClassifyResult, ClassifierError> {
        let prompt = self.build_classification_prompt();

        let messages: Vec<ChatCompletionRequestMessage> = vec![
            ChatCompletionRequestMessage::User(
                ChatCompletionRequestUserMessage {
                    content: ChatCompletionRequestUserMessageContent::Text(
                        format!("{}\n\n{}", prompt, context)
                    ),
                    name: None,
                }
            )
        ];

        // Define tools
        let tools = Some(vec![
            ChatCompletionTools::Function(ChatCompletionTool {
                function: FunctionObject {
                    name: "get_event".to_string(),
                    description: Some("Get an event by its ID. Returns the full event including content, author, kind, and tags. Fetches from relays if not cached locally.".to_string()),
                    parameters: Some(serde_json::json!({
                        "type": "object",
                        "properties": {
                            "event_id": {
                                "type": "string",
                                "description": "The event ID (hex string) to look up"
                            }
                        },
                        "required": ["event_id"]
                    })),
                    strict: None,
                },
            }),
            ChatCompletionTools::Function(ChatCompletionTool {
                function: FunctionObject {
                    name: "get_profile".to_string(),
                    description: Some("Get a profile by their pubkey. Returns name, bio, NIP-05, and picture. Fetches from relays if not cached locally.".to_string()),
                    parameters: Some(serde_json::json!({
                        "type": "object",
                        "properties": {
                            "pubkey": {
                                "type": "string",
                                "description": "The pubkey (hex string) to look up"
                            }
                        },
                        "required": ["pubkey"]
                    })),
                    strict: None,
                },
            }),
            ChatCompletionTools::Function(ChatCompletionTool {
                function: FunctionObject {
                    name: "describe_image".to_string(),
                    description: Some("Describe an image or video by its URL. Downloads the media and returns a detailed text description. For images, describes objects, people, text, scenes, and style. For videos, extracts key frames and describes what's happening across the video — actions, settings, people, and visual content. ALWAYS call this for any image or video URL you see — you cannot determine what media contains from its URL alone. This includes profile picture URLs (e.g. Profile Image: https://...), image URLs (.jpg, .png, .gif, .webp), and video URLs (.mp4, .webm, .mov, .avi, .mkv).".to_string()),
                    parameters: Some(serde_json::json!({
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "The URL of the image to describe"
                            }
                        },
                        "required": ["url"]
                    })),
                    strict: None,
                },
            }),
            ChatCompletionTools::Function(ChatCompletionTool {
                function: FunctionObject {
                    name: "resolve_nip21".to_string(),
                    description: Some("Resolve a NIP-21 nostr: URI to get details about the referenced entity. Accepts nostr:npub1..., nostr:nprofile1..., nostr:note1..., nostr:nevent1..., and nostr:naddr1... URIs. For profile references (npub/nprofile), returns profile metadata. For event references (note/nevent/naddr), returns the full event including content and tags. ALWAYS call this for any nostr: URI you encounter in event content — you cannot determine what it references from the URI alone.".to_string()),
                    parameters: Some(serde_json::json!({
                        "type": "object",
                        "properties": {
                            "uri": {
                                "type": "string",
                                "description": "The NIP-21 URI to resolve (e.g. nostr:npub1..., nostr:nevent1..., nostr:naddr1...)"
                            }
                        },
                        "required": ["uri"]
                    })),
                    strict: None,
                },
            }),
            ChatCompletionTools::Function(ChatCompletionTool {
                function: FunctionObject {
                    name: "get_opengraph".to_string(),
                    description: Some("Fetch OpenGraph metadata for a URL found in a post. Returns the page title, description, site name, type, and image URL. This reveals what a shared link is about — for example, a URL shared in a post might link to a political news article, a tech blog, or a product page. ALWAYS call this for any non-nostr, non-image URL you see in event content — you cannot determine what a link contains from its URL alone. Do NOT call this for nostr: URIs (use resolve_nip21 instead) or image/video URLs (use describe_image instead).".to_string()),
                    parameters: Some(serde_json::json!({
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "The HTTPS URL to fetch OpenGraph metadata for"
                            }
                        },
                        "required": ["url"]
                    })),
                    strict: None,
                },
            }),
            ChatCompletionTools::Function(ChatCompletionTool {
                function: FunctionObject {
                    name: "get_profile_events".to_string(),
                    description: Some("Fetch additional events for the profile being classified, filtered by kind and optional time range. Use this when the initial event batch is dominated by one kind (e.g. mostly zap receipts) and you need more content from other kinds to make an accurate classification. Returns the most recent events matching the criteria.".to_string()),
                    parameters: Some(serde_json::json!({
                        "type": "object",
                        "properties": {
                            "kind": {
                                "type": "integer",
                                "description": "The Nostr event kind to filter by (e.g. 1 for short text notes, 7 for reactions, 9735 for zap receipts, 30023 for long-form content)"
                            },
                            "since": {
                                "type": "integer",
                                "description": "Only return events created at or after this Unix timestamp (optional)"
                            },
                            "until": {
                                "type": "integer",
                                "description": "Only return events created at or before this Unix timestamp (optional)"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of events to return (default 20, max 50)"
                            }
                        },
                        "required": ["kind"]
                    })),
                    strict: None,
                },
            }),
            ChatCompletionTools::Function(ChatCompletionTool {
                function: FunctionObject {
                    name: "summarize_event".to_string(),
                    description: Some("Get a concise AI-generated summary of a long event's content. Use this when an event's content was truncated (showing '[content truncated: N total chars, showing first M]') and you need to understand what the full content is about. Returns a summary of the main topics, stance, and key points. Only call this for events whose content was explicitly truncated.".to_string()),
                    parameters: Some(serde_json::json!({
                        "type": "object",
                        "properties": {
                            "event_id": {
                                "type": "string",
                                "description": "The event ID (hex string) whose content was truncated and needs summarization"
                            }
                        },
                        "required": ["event_id"]
                    })),
                    strict: None,
                },
            }),
        ]);

        let request = CreateChatCompletionRequest {
            model: self.model.clone(),
            messages,
            temperature: Some(0.2),
            tools,
            ..Default::default()
        };

        // Handle tool calls with iterative loop, wrapped in an overall timeout
        let classify_timeout = self.classify_timeout;
        let pubkey_owned = pubkey.to_string();
        let result = tokio::time::timeout(
            classify_timeout,
            self.call_with_tool_handling(request, &pubkey_owned),
        )
        .await
        .map_err(|_| {
            ClassifierError::Timeout(
                classify_timeout.as_secs()
            )
        })??;
        
        Ok(result)
    }

    async fn call_with_tool_handling(&self, mut request: CreateChatCompletionRequest, pubkey: &str) -> Result<ClassifyResult, ClassifierError> {
        let mut max_iterations = 15;
        let mut tool_event_counts: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();

        let make_result = |c: Classification, counts: std::collections::HashMap<i64, usize>| ClassifyResult {
            classification: c,
            tool_event_counts: counts,
        };
        
        loop {
            if max_iterations == 0 {
                break;
            }
            max_iterations -= 1;

            let response = self.client.chat().create(request.clone()).await?;
            let choice = &response.choices[0];
            let finish_reason = format!("{:?}", choice.finish_reason);
            let content = choice.message.content.as_deref().unwrap_or("");
            let has_content = !content.is_empty();
            let tool_calls = choice.message.tool_calls.as_deref();
            info!("LLM response: finish_reason={}, has_content={}, tool_calls={}", 
                finish_reason, has_content, tool_calls.map(|tc| tc.len()).unwrap_or(0));
            
            if let Some(ref refusal) = choice.message.refusal {
                tracing::warn!("LLM refused: {}", refusal);
            }

            // If the model is done (Stop) and gave us content, parse it regardless of tool_calls
            if has_content && matches!(choice.finish_reason, Some(async_openai::types::chat::FinishReason::Stop)) {
                match self.parse_classification(content) {
                    Ok(c) => return Ok(make_result(c, tool_event_counts)),
                    Err(e) => {
                        tracing::warn!("Failed to parse classification, retrying: {}", e);
                        request.tools = None; // no more tool calls, just fix the JSON
                        request.messages.push(ChatCompletionRequestMessage::Assistant(
                            ChatCompletionRequestAssistantMessage {
                                content: Some(async_openai::types::chat::ChatCompletionRequestAssistantMessageContent::Text(content.to_string())),
                                ..Default::default()
                            }
                        ));
                        request.messages.push(ChatCompletionRequestMessage::User(
                            ChatCompletionRequestUserMessage {
                                content: ChatCompletionRequestUserMessageContent::Text(
                                    format!("Your previous response could not be parsed as valid JSON: {}. Please output ONLY valid JSON with the exact structure specified.", e)
                                ),
                                name: None,
                            }
                        ));
                        continue;
                    }
                }
            }
            
            // Check if the model wants to call a tool
            if let Some(ref tool_calls) = choice.message.tool_calls {
                if tool_calls.is_empty() {
                    // No tool calls and no content - try again without tools
                    tracing::warn!("LLM returned empty tool_calls and no content");
                    request.tools = None;
                    continue;
                }

                // Add assistant message with tool calls
                request.messages.push(ChatCompletionRequestMessage::Assistant(
                    ChatCompletionRequestAssistantMessage {
                        content: choice.message.content.clone().map(async_openai::types::chat::ChatCompletionRequestAssistantMessageContent::Text),
                        tool_calls: Some(tool_calls.clone()),
                        ..Default::default()
                    }
                ));

                // Process each tool call
                for tool_call in tool_calls {
                    let (id, name, arguments) = match tool_call {
                        ChatCompletionMessageToolCalls::Function(f) => {
                            (f.id.clone(), f.function.name.clone(), f.function.arguments.clone())
                        }
                        ChatCompletionMessageToolCalls::Custom(c) => {
                            (c.id.clone(), c.custom_tool.name.clone(), c.custom_tool.input.clone())
                        }
                    };
                    info!("Tool call: {}({})", name, arguments);
                    let (result, event_info) = tokio::time::timeout(
                        self.tool_call_timeout,
                        self.call_tool(&name, &arguments, pubkey),
                    )
                    .await
                    .map_err(|_| ClassifierError::ToolTimeout { tool: name.clone(), timeout: self.tool_call_timeout.as_secs() })?
                    .map_err(|e| ClassifierError::ToolFailed { tool: name.clone(), source: Box::new(e) })?;
                    info!("Tool response: {} -> {:.200}", name, result);

                    // Track additional events fetched via get_profile_events
                    if let Some((kind, count)) = event_info {
                        *tool_event_counts.entry(kind).or_insert(0) += count;
                    }
                    
                    // Add tool response message
                    request.messages.push(ChatCompletionRequestMessage::Tool(
                        ChatCompletionRequestToolMessage {
                            content: ChatCompletionRequestToolMessageContent::Text(result),
                            tool_call_id: id,
                        }
                    ));
                }
            } else if has_content {
                match self.parse_classification(content) {
                    Ok(c) => return Ok(make_result(c, tool_event_counts)),
                    Err(e) => {
                        tracing::warn!("Failed to parse classification, retrying: {}", e);
                        request.tools = None;
                        request.messages.push(ChatCompletionRequestMessage::Assistant(
                            ChatCompletionRequestAssistantMessage {
                                content: Some(async_openai::types::chat::ChatCompletionRequestAssistantMessageContent::Text(content.to_string())),
                                ..Default::default()
                            }
                        ));
                        request.messages.push(ChatCompletionRequestMessage::User(
                            ChatCompletionRequestUserMessage {
                                content: ChatCompletionRequestUserMessageContent::Text(
                                    format!("Your previous response could not be parsed as valid JSON: {}. Please output ONLY valid JSON with the exact structure specified.", e)
                                ),
                                name: None,
                            }
                        ));
                        continue;
                    }
                }
            } else {
                tracing::warn!("LLM returned no content and no tool calls, retrying without tools");
                request.tools = None;
                continue;
            }
        }

        // Fallback: make one final request without tools to force a text response
        info!("Max iterations reached, making final request without tools");
        request.tools = None;
        let response = self.client.chat().create(request.clone()).await?;
        let content = response.choices[0].message.content.as_deref().unwrap_or("");
        self.parse_classification(content).map(|c| make_result(c, tool_event_counts))
    }

    fn parse_classification(&self, content: &str) -> Result<Classification, ClassifierError> {
        let content = content.trim();
        
        if content.is_empty() {
            return Err(ClassifierError::Parse("LLM returned empty response".to_string()));
        }

        // Helper: try to parse the raw JSON and convert scores → labels
        let valid_labels: std::collections::HashSet<&str> = self.label_taxonomy.iter().map(|s| s.as_str()).collect();
        let try_parse = |json_str: &str| -> Option<Classification> {
            // First try the new scores format
            if let Ok(raw) = serde_json::from_str::<serde_json::Value>(json_str) {
                if let Some(scores_map) = raw.get("scores").and_then(|v| v.as_object()) {
                    let bio = raw.get("bio").and_then(|v| v.as_str()).unwrap_or("").to_string();
                    let confidence = raw.get("confidence").and_then(|v| v.as_f64()).unwrap_or(0.5);
                    
                    // Build the full scores map — only include labels that are in the taxonomy
                    let mut scores: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
                    for (label, score) in scores_map.iter() {
                        if let Some(s) = score.as_f64() {
                            if valid_labels.contains(label.as_str()) {
                                scores.insert(label.clone(), s);
                            }
                        }
                    }

                    // Sort labels by score descending for consistent ordering
                    let mut scored: Vec<(String, f64)> = scores.iter()
                        .filter_map(|(label, s)| {
                            if *s >= self.label_min_score {
                                Some((label.clone(), *s))
                            } else {
                                None
                            }
                        })
                        .collect();
                    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    let labels: Vec<String> = scored.into_iter().map(|(l, _)| l).collect();

                    if !labels.is_empty() || !bio.is_empty() {
                        return Some(Classification { labels, scores, bio, confidence });
                    }
                }
                
                // Fallback: try old labels format for backward compatibility
                if let Ok(c) = serde_json::from_str::<Classification>(json_str) {
                    return Some(c);
                }
            }
            None
        };

        // Try the whole content first
        if let Some(c) = try_parse(content) {
            return Ok(c);
        }

        // Strip markdown code block if present
        let stripped = content.strip_prefix("```json")
            .or_else(|| content.strip_prefix("```"))
            .and_then(|c| c.strip_suffix("```"))
            .map(|c| c.trim())
            .unwrap_or(content);

        if let Some(c) = try_parse(stripped) {
            return Ok(c);
        }

        // Find the outermost { ... } JSON object in the response
        if let Some(start) = stripped.find('{') {
            let mut depth = 0i32;
            let mut end = start;
            for (i, ch) in stripped[start..].char_indices() {
                match ch {
                    '{' => depth += 1,
                    '}' => {
                        depth -= 1;
                        if depth == 0 {
                            end = start + i + 1;
                            break;
                        }
                    }
                    '"' => {
                        // Skip string contents so braces inside strings don't affect depth
                        let mut j = i + 1;
                        while j < stripped[start..].len() {
                            let c = stripped[start + j..].chars().next().unwrap();
                            j += c.len_utf8();
                            if c == '\\' { j += 1; continue; } // skip escaped char
                            if c == '"' { break; }
                        }
                    }
                    _ => {}
                }
            }
            if depth == 0 {
                let json_str = &stripped[start..end];
                if let Some(c) = try_parse(json_str) {
                    return Ok(c);
                }
            }
        }

        Err(ClassifierError::Parse(format!("Failed to parse classification from response (first 200 chars): {:.200}", content)))
    }

    /// Returns (response_text, optional (kind, count) for profile event fetches)
    async fn call_tool(&self, name: &str, arguments: &str, pubkey: &str) -> Result<(String, Option<(i64, usize)>), ClassifierError> {
        match name {
            "get_event" => {
                let args: serde_json::Value = serde_json::from_str(arguments)
                    .map_err(|e| ClassifierError::InvalidToolArgs(format!("Invalid JSON arguments: {}", e)))?;
                let event_id = args.get("event_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| ClassifierError::InvalidToolArgs("Missing event_id".to_string()))?;

                // Check DB cache first
                if let Some(cached) = self.db.get_event(event_id).await? {
                    if let Ok(event) = nostr_sdk::Event::from_json(&cached.raw_json) {
                        return Ok((crate::format::describe_event(&event), None));
                    }
                }

                // Fetch from relays and cache
                match self.nostr.fetch_event_by_id(event_id).await? {
                    Some(event) => {
                        if let Err(e) = self.db.cache_event(&event).await {
                            tracing::warn!("Failed to cache fetched event {}: {}", event_id, e);
                        }
                        Ok((crate::format::describe_event(&event), None))
                    }
                    None => Ok((format!("Event not found: {}", event_id), None)),
                }
            }
            "get_profile" => {
                let args: serde_json::Value = serde_json::from_str(arguments)
                    .map_err(|e| ClassifierError::InvalidToolArgs(format!("Invalid JSON arguments: {}", e)))?;
                let pubkey = args.get("pubkey")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| ClassifierError::InvalidToolArgs("Missing pubkey".to_string()))?;

                match self.profile_cache.get_profile(pubkey).await? {
                    Some(profile) => Ok((crate::format::describe_profile(&profile), None)),
                    None => Ok((format!("Profile not found: {}", pubkey), None)),
                }
            }
            "describe_image" => {
                let args: serde_json::Value = serde_json::from_str(arguments)
                    .map_err(|e| ClassifierError::InvalidToolArgs(format!("Invalid JSON arguments: {}", e)))?;
                let url = args.get("url")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| ClassifierError::InvalidToolArgs("Missing url".to_string()))?;

                match self.image_cache.download(url).await? {
                    Some((path, hash)) => {
                        if !crate::image_cache::is_valid_image_path(&path) {
                            return Ok(("Could not download or decode image (invalid format)".to_string(), None));
                        }
                        // Check for cached description
                        match self.db.get_image_description(&hash).await? {
                            Some(cached) => Ok((cached, None)),
                            None => {
                                let desc = self.describe_image(&path).await?;
                                let _ = self.db.save_image_description(&hash, &desc).await;
                                Ok((desc, None))
                            }
                        }
                    }
                    None => Ok((format!("Could not download image: {}", url), None)),
                }
            }
            "resolve_nip21" => {
                let args: serde_json::Value = serde_json::from_str(arguments)
                    .map_err(|e| ClassifierError::InvalidToolArgs(format!("Invalid JSON arguments: {}", e)))?;
                let uri = args.get("uri")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| ClassifierError::InvalidToolArgs("Missing uri".to_string()))?;

                match nostr_sdk::nips::nip21::Nip21::parse(uri) {
                    Ok(nip21) => match nip21 {
                        nostr_sdk::nips::nip21::Nip21::Pubkey(pk) => {
                            let pubkey_hex = pk.to_hex();
                            match self.profile_cache.get_profile(&pubkey_hex).await? {
                                Some(profile) => Ok((crate::format::describe_profile(&profile), None)),
                                None => Ok((format!("Profile not found: {}", pubkey_hex), None)),
                            }
                        }
                        nostr_sdk::nips::nip21::Nip21::Profile(nprofile) => {
                            let pubkey_hex = nprofile.public_key.to_hex();
                            match self.profile_cache.get_profile(&pubkey_hex).await? {
                                Some(profile) => Ok((crate::format::describe_profile(&profile), None)),
                                None => Ok((format!("Profile not found: {}", pubkey_hex), None)),
                            }
                        }
                        nostr_sdk::nips::nip21::Nip21::EventId(event_id) => {
                            let event_id_hex = event_id.to_hex();
                            // Check DB cache first
                            if let Some(cached) = self.db.get_event(&event_id_hex).await? {
                                if let Ok(event) = nostr_sdk::Event::from_json(&cached.raw_json) {
                                    return Ok((crate::format::describe_event(&event), None));
                                }
                            }
                            // Fetch from relays
                            match self.nostr.fetch_event_by_id(&event_id_hex).await? {
                                Some(event) => {
                                    if let Err(e) = self.db.cache_event(&event).await {
                                        tracing::warn!("Failed to cache fetched event {}: {}", event_id_hex, e);
                                    }
                                    Ok((crate::format::describe_event(&event), None))
                                }
                                None => Ok((format!("Event not found: {}", event_id_hex), None)),
                            }
                        }
                        nostr_sdk::nips::nip21::Nip21::Event(nevent) => {
                            let event_id_hex = nevent.event_id.to_hex();
                            // Check DB cache first
                            if let Some(cached) = self.db.get_event(&event_id_hex).await? {
                                if let Ok(event) = nostr_sdk::Event::from_json(&cached.raw_json) {
                                    return Ok((crate::format::describe_event(&event), None));
                                }
                            }
                            // Fetch from relays
                            match self.nostr.fetch_event_by_id(&event_id_hex).await? {
                                Some(event) => {
                                    if let Err(e) = self.db.cache_event(&event).await {
                                        tracing::warn!("Failed to cache fetched event {}: {}", event_id_hex, e);
                                    }
                                    Ok((crate::format::describe_event(&event), None))
                                }
                                None => Ok((format!("Event not found: {}", event_id_hex), None)),
                            }
                        }
                        nostr_sdk::nips::nip21::Nip21::Coordinate(naddr) => {
                            // For naddr, we need to fetch by coordinate (kind:pubkey:dtag)
                            // Build a filter for the addressable event
                            let coord = &naddr.coordinate;
                            let filter = nostr_sdk::Filter::new()
                                .kind(coord.kind)
                                .author(coord.public_key)
                                .identifier(&coord.identifier)
                                .limit(1);

                            match self.nostr.client()
                                .fetch_events(filter, std::time::Duration::from_secs(10))
                                .await
                            {
                                Ok(events) => {
                                    if let Some(event) = events.into_iter().next() {
                                        if let Err(e) = self.db.cache_event(&event).await {
                                            tracing::warn!("Failed to cache fetched coordinate event: {}", e);
                                        }
                                        Ok((crate::format::describe_event(&event), None))
                                    } else {
                                        Ok((format!("Addressable event not found: {}:{}", coord.kind, coord.public_key.to_hex()), None))
                                    }
                                }
                                Err(e) => Ok((format!("Error fetching addressable event: {}", e), None)),
                            }
                        }
                    },
                    Err(e) => Ok((format!("Invalid NIP-21 URI '{}': {}", uri, e), None)),
                }
            }
            "get_opengraph" => {
                let args: serde_json::Value = serde_json::from_str(arguments)
                    .map_err(|e| ClassifierError::InvalidToolArgs(format!("Invalid JSON arguments: {}", e)))?;
                let url = args.get("url")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| ClassifierError::InvalidToolArgs("Missing url".to_string()))?;

                match self.og_cache.get_preview(url).await {
                    Some(data) => Ok((crate::format::describe_opengraph(&data), None)),
                    None => Ok((format!("No OpenGraph data found for: {}", url), None)),
                }
            }
            "get_profile_events" => {
                let args: serde_json::Value = serde_json::from_str(arguments)
                    .map_err(|e| ClassifierError::InvalidToolArgs(format!("Invalid JSON arguments: {}", e)))?;
                let kind = args.get("kind")
                    .and_then(|v| v.as_i64())
                    .ok_or_else(|| ClassifierError::InvalidToolArgs("Missing kind".to_string()))?;
                let since = args.get("since").and_then(|v| v.as_i64());
                let until = args.get("until").and_then(|v| v.as_i64());
                let limit = args.get("limit")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(20)
                    .min(50) as usize;

                let events = self.db.get_profile_events_by_kind(pubkey, kind, since, until, limit).await?;
                let count = events.len();
                if events.is_empty() {
                    return Ok((format!("No events found for kind {} for this profile", kind), None));
                }

                let mut result = String::new();
                for (i, event) in events.iter().enumerate() {
                    let Ok(nostr_event) = nostr_sdk::Event::from_json(&event.raw_json) else {
                        continue;
                    };
                    result.push_str(&format!("--- Event {} ---\n", i + 1));
                    result.push_str(&crate::format::describe_event(&nostr_event));
                    result.push('\n');
                }
                Ok((result, Some((kind, count))))
            }
            "summarize_event" => {
                let args: serde_json::Value = serde_json::from_str(arguments)
                    .map_err(|e| ClassifierError::InvalidToolArgs(format!("Invalid JSON arguments: {}", e)))?;
                let event_id = args.get("event_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| ClassifierError::InvalidToolArgs("Missing event_id".to_string()))?;

                // Fetch the event
                let event = if let Some(cached) = self.db.get_event(event_id).await? {
                    nostr_sdk::Event::from_json(&cached.raw_json).ok()
                } else {
                    self.nostr.fetch_event_by_id(event_id).await.ok().flatten()
                };

                let Some(event) = event else {
                    return Ok((format!("Event not found: {}", event_id), None));
                };

                let content = &event.content;
                if content.is_empty() {
                    return Ok(("(event has no content)".to_string(), None));
                }

                // Summarize with a lightweight LLM call
                let summary = self.summarize_content(content).await?;
                Ok((summary, None))
            }
            _ => Err(ClassifierError::UnknownTool(name.to_string())),
        }
    }

    /// Summarize a long piece of content into a concise paragraph.
    async fn summarize_content(&self, content: &str) -> Result<String, ClassifierError> {
        let prompt = format!(
            "Summarize the following Nostr post content concisely (max 400 chars). Focus on: main topic(s), what the author wrote about, their stance/opinion, and any key points relevant to classifying their interests.\n\nContent:\n{}",
            content
        );

        let request = CreateChatCompletionRequest {
            model: self.model.clone(),
            messages: vec![
                ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
                    content: ChatCompletionRequestUserMessageContent::Text(prompt),
                    name: None,
                })
            ],
            temperature: Some(0.0),
            max_completion_tokens: Some(300),
            ..Default::default()
        };

        let response = self.client.chat().create(request).await?;
        let summary = response.choices[0]
            .message
            .content
            .as_deref()
            .unwrap_or("(no summary available)");
        Ok(summary.to_string())
    }
    
    async fn encode_image(&self, path: &str) -> Result<String, ClassifierError> {
        let bytes = tokio::fs::read(path).await.map_err(|e| ClassifierError::Image(format!("Failed to read image: {}", e)))?;
        if bytes.is_empty() {
            return Err(ClassifierError::Image("Failed to decode image".to_string()));
        }
        
        // Load image
        let img = image::load_from_memory(&bytes).map_err(|e| ClassifierError::Image(format!("Failed to load image: {}", e)))?;
        let (width, height) = img.dimensions();
        
        // Resize if either dimension exceeds 1024, preserving aspect ratio
        let img = if width > 1024 || height > 1024 {
            img.resize(1024, 1024, image::imageops::FilterType::Lanczos3)
        } else {
            img
        };
        
        // Convert to RGB if needed
        let img = img.to_rgb8();
        
        // Encode as JPEG with quality 85
        let mut jpeg_bytes = Vec::new();
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg_bytes, 85)
            .encode(&img, img.width(), img.height(), image::ExtendedColorType::Rgb8).map_err(|e| ClassifierError::Image(format!("Failed to encode JPEG: {}", e)))?;
        
        Ok(STANDARD.encode(&jpeg_bytes))
    }

pub async fn describe_image(&self, path: &str) -> Result<String, ClassifierError> {
        let image_content = format!("data:image/jpeg;base64,{}", self.encode_image(path).await?);

        // Detect if this is a video collage (files ending in .video.collage.jpg)
        let is_video_collage = path.contains(".video.collage.");
        let prompt = if is_video_collage {
            "This image is a collage of key frames extracted from a video. Describe what is happening in the video based on these frames. What actions, scenes, people, text, and settings are shown? How does the content progress across the frames? Be specific and objective. Keep description concise.".to_string()
        } else {
            "Describe this image in detail. What is shown? What colors, objects, people, text, or scenes are visible? Be specific and objective. Keep description concise.".to_string()
        };

        let request = CreateChatCompletionRequest {
            model: self.model.to_string(),
            messages: vec![
                ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
                    content: ChatCompletionRequestUserMessageContent::Array(vec![
                        ChatCompletionRequestUserMessageContentPart::Text(ChatCompletionRequestMessageContentPartText{
                            text: prompt,
                        }),
                        ChatCompletionRequestUserMessageContentPart::ImageUrl(ChatCompletionRequestMessageContentPartImage {
                            image_url: ImageUrl {
                                url: image_content,
                                detail: Some(ImageDetail::Auto),
                            },
                        })
                    ]),
                    name: None,
                })
            ],
            temperature: Some(0.3),
            ..Default::default()
        };

        let response = self.client.chat().create(request).await?;
        let content = response.choices[0].message.content.as_deref().unwrap_or("No description available");
        
        Ok(content.to_string())
    }

    fn build_classification_prompt(&self) -> String {
        let label_list = self.label_taxonomy.iter()
            .enumerate()
            .map(|(i, l)| format!("{}. {}", i + 1, l))
            .collect::<Vec<_>>()
            .join("\n");

        format!(r#"Classify this person's interests based on their Nostr activity and score against a fixed label taxonomy.

NIP-21 entities are how people reference other entities in their posts. They appear as `nostr:xxx` URIs in event content and embed rich context about what's being referenced. You MUST use the resolve_nip21 tool to look up any `nostr:` URI you see — you cannot determine what it contains from the URI alone.

NIP-21 URI types:
- nostr:npub1... — a public key (hex 32 bytes, bech32 encoded). Use resolve_nip21 to get their profile.
- nostr:nprofile1... — a public key with relay hints. Use resolve_nip21 to get their profile.
- nostr:note1... — an event ID (hex 32 bytes, bech32 encoded). Use resolve_nip21 to get the full event.
- nostr:nevent1... — an event ID with optional author, kind, and relay hints. Use resolve_nip21 to get the full event.
- nostr:naddr1... — a coordinate reference (kind:pubkey:d-tag) for replaceable/parameterized events like long-form articles (kind 30023), channels, or communities. Use resolve_nip21 to get the referenced event.

When you see a `nostr:` URI in event content, it means the person is explicitly referencing another entity — a person, a post, or a replaceable event. Call resolve_nip21 for each one to understand the context.

Nostr event tags are how people reference other people and content at the protocol level. Understanding them is critical for classification:

- ["e", "<event_id>"] — references another event. Used in replies, reactions, and reposts.
- ["e", "<event_id>", "<relay_url>", "<marker>", "<pubkey>"] — marked e tag. The marker is "root" (the event starting a thread) or "reply" (the direct parent being responded to). This indicates a reply in a conversation.
- ["p", "<pubkey>"] — mentions or notifies another person. Frequent ["p", ...] tags to the same pubkey indicate a social connection or ongoing conversation.
- ["a", "<kind>:<pubkey>:<d-tag>"] — references a replaceable or parameterized event by its coordinate. Common for referencing long-form articles (kind 30023), channels, or communities.
- ["q", "<event_id>", "<relay_url>", "<pubkey>"] — quote repost. Unlike e tags (which are replies in a thread), q tags mean the person is quoting another event in their own post. This is NOT a reply — it's a citation.
- ["k", "<kind_number>"] — indicates the kind of the event being referenced (used in reactions kind 7, generic reposts kind 16, and external content reactions kind 17).

Zap events (kind 9735 = zap receipt) are particularly rich signals:
- The 9735 event's pubkey is the recipient's LNURL server (not the sender or receiver).
- The ["p", "<pubkey>"] tag is the zap RECIPIENT's pubkey.
- The ["P", "<pubkey>"] tag (uppercase) is the zap SENDER's pubkey.
- The ["e", "<event_id>"] tag (if present) is the event being zapped.
- The ["bolt11", "..."] tag contains the lightning invoice, which encodes the amount in millisats.
- The ["description", "..."] tag contains the full JSON of the original kind 9734 zap request, which includes:
  - The sender's pubkey (the 9734 event's pubkey field)
  - An optional message in the content field (zap comment)
  - The ["amount", "<millisats>"] tag for the payment amount
  - The ["e", "..."] and ["p", "..."] tags from the request
- The ["amount", "<millisats>"] tag may also appear directly on the 9735.

When you see a 9735 event, parse the description tag to understand who sent the zap, how much, and what event or profile it was for. Zaps received indicate what content the person's audience values.

When you see these tags, consider what the person is interacting with — replying to specific posts (e tags with "reply" marker), quoting content (q tags), mentioning people (p tags), or joining communities (a tags) all reveal interests and social connections.

Consider all available signals:
- Posts: What topics are discussed? What tone and expertise level?
- Replies (events with ["e", ...] tags): What topics come up in conversations? What communities?
- Mentions (["p", ...] tags): Who do they talk to or about? Frequent mentions of the same pubkey suggest a close social connection.
- Reactions: What content do they react to? Use get_event to look up the referenced event — this reveals their interests.
- Reposts: What content do they amplify? This shows what they associate with.
- Picture posts (kind 20): These are photo-first posts with structured imeta tags listing image URLs. Call describe_image for each image URL to understand the visual content. Someone posting nature photos is interested in nature/photography; someone posting food photos is interested in food/cooking.
- Video posts (kind 21, 22, 34235, 34236): These are video-first posts with structured imeta tags listing video URLs and thumbnail images. Call describe_image for thumbnail URLs to understand the video content. Someone posting tech tutorials is interested in software development; someone posting travel vlogs is interested in travel.
- Zaps received: What content earns tips? This shows what the audience values.
- Zaps sent: Who do they tip? This shows who they support financially.
- Profile picture and images: Call describe_image for any image URL you see (profile pictures, images in posts). You cannot judge visual content from a URL alone — always call describe_image first.
- Shared links: Call get_opengraph for HTTPS URLs in posts to understand what they link to.
- Profile metadata: Name, about section, NIP-05 domain, and picture can all signal identity and interests.

If a PREVIOUS CLASSIFICATION section is present, use it as context — adjust scores based on new activity.

IMPORTANT: Before scoring, call the appropriate tools for every reference you encounter:
1. Call describe_image for every image URL in the profile data and media events. This includes:
   - "Profile Image: https://..." lines
   - "Images: https://..." lines in Picture events (kind 20)
   - "Thumbnails: https://..." lines in Video events (kind 21/22/34235/34236)
   - Any URLs ending in .jpg, .jpeg, .png, .gif, .webp in event content
   You MUST call describe_image for these URLs — do not skip them or guess what they contain.
2. Call resolve_nip21 for every nostr: URI in event content (e.g. nostr:npub1..., nostr:nevent1..., nostr:naddr1...).
   These are explicit references to other entities — understanding what they point to is critical for classification.
   Do NOT skip them or guess what they reference from the URI string alone.
3. Call get_opengraph for every non-nostr, non-image, non-video HTTPS URL in event content.
   Shared links reveal interests — someone linking to a political news article is interested in politics, someone linking to a GitHub repo is interested in software development.
   Do NOT call get_opengraph for nostr: URIs (use resolve_nip21) or image/video URLs (use describe_image).
4. Events may have their content truncated — look for "[content truncated: N total chars, showing first M]" markers.
   When content is truncated, call summarize_event(event_id) to get an AI summary of the full content.
   The event ID is shown at the top of each described event as "ID: <hex>". Only summarize when content was explicitly truncated.

LABEL TAXONOMY (score each one):
{label_list}

Scoring rules:
- Score each label 0.0–1.0 based on how well it fits
- 0.0 = not relevant at all, 1.0 = perfectly fits
- Be selective: most labels should score 0.0 or close to it
- A label should only score above {min_score} if there is clear evidence
- Multiple related labels can score high (e.g. both "bitcoin" and "lightning-network")
- Use image descriptions to inform labels like "artist", "photographer", "nsfw", "bot" etc.
- Reactions and reposts reveal interests just as much as original posts — someone who reacts to bitcoin content is interested in bitcoin
- Language labels: Score the corresponding language label based on how frequently that language appears in the person's posts. A person who mostly posts in Japanese should score "japanese" high (0.8–1.0). Someone who occasionally posts in German alongside English should score "german" moderate (0.3–0.5). Score "english" the same way — if someone posts in English, score it accordingly.

Generate:
1. scores: An object mapping each label name to its score (0.0–1.0). Include ALL labels, even those scoring 0.0.
2. bio: A summary of who this person is and what they care about. Rules:
   - Describe ONLY their interests and topics. Write about what they care about, not what they do for a living or how they behave online.
   - Do NOT speculate about professional roles, expertise, or technical skill levels unless directly evidenced by their content. If someone reacts to a bitcoin post, they are interested in bitcoin — they are not necessarily a "bitcoin developer" or "expert".
   - Do NOT describe HOW they use the platform — no "casual observer", "active participant", "engages with", "engagement style", "doesn't create content", "low-key", "supportive", "frequently reacts", "actively engages", or any similar phrasing. These are forbidden.
   - Do NOT mention Nostr at all — no "Nostr user", "Nostr community", "active on Nostr", etc.
   - Bad: "Alice is a casual observer who reacts to bitcoin posts and doesn't create much content. While not a developer herself..."
   - Good: "Alice is interested in Bitcoin and the Lightning Network, with an appreciation for nature and quiet mornings."
3. confidence: 0.0-1.0 indicating how confident you are in the overall classification

Output ONLY valid JSON with this exact structure:
{{"scores": {{"label-name": 0.8, ...}}, "bio": "summary text", "confidence": 0.85}}"#,

            label_list = label_list,
            min_score = self.label_min_score,
        )
    }
}
