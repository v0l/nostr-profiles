use crate::config::LlmConfig;
use crate::nostr_client::NostrClient;
use anyhow::{bail, Result};
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
use tracing::info;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Classification {
    pub labels: Vec<String>,
    pub bio: String,
    pub confidence: f64,
}

#[derive(Clone)]
pub struct LLMClient {
    client: Client<OpenAIConfig>,
    model: String,
    nostr: NostrClient,
}

impl LLMClient {
    pub fn new(config: &LlmConfig, nostr: NostrClient) -> Self {
        let openai_config = OpenAIConfig::new()
            .with_api_base(&config.api_base_url)
            .with_api_key(&config.api_key);

        let client = Client::with_config(openai_config);
        Self {
            client,
            model: config.model.clone(),
            nostr,
        }
    }

    pub async fn classify_with_images(
        &self,
        context: &str,
        image_paths: &[String],
    ) -> Result<Classification> {
        let mut messages: Vec<ChatCompletionRequestMessage> = vec![
            ChatCompletionRequestMessage::User(
                ChatCompletionRequestUserMessage {
                    content: ChatCompletionRequestUserMessageContent::Text(
                        format!("{}\n\n{}", CLASSIFICATION_PROMPT, context)
                    ),
                    name: None,
                }
            )
        ];

        // If vision model is configured, generate text descriptions instead of passing raw images
        if !image_paths.is_empty() {
            info!("Generating text descriptions for {} images (reducing context size)", image_paths.len());
            let descriptions = self.generate_image_descriptions(image_paths).await?;
            messages.push(ChatCompletionRequestMessage::User(
                ChatCompletionRequestUserMessage {
                    content: ChatCompletionRequestUserMessageContent::Text(descriptions),
                    name: None,
                },
            ));
        }

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
        ]);

        let request = CreateChatCompletionRequest {
            model: self.model.clone(),
            messages,
            temperature: Some(0.2),
            tools,
            ..Default::default()
        };

        // Handle tool calls with iterative loop
        let classification = self.call_with_tool_handling(request).await?;
        
        Ok(classification)
    }

    async fn call_with_tool_handling(&self, mut request: CreateChatCompletionRequest) -> Result<Classification> {
        let mut max_iterations = 5;
        
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
                return self.parse_classification(content);
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
                        content: choice.message.content.clone().map(|c| async_openai::types::chat::ChatCompletionRequestAssistantMessageContent::Text(c)),
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
                    let result = self.call_tool(&name, &arguments).await?;
                    info!("Tool response: {} -> {:.200}", name, result);
                    
                    // Add tool response message
                    request.messages.push(ChatCompletionRequestMessage::Tool(
                        ChatCompletionRequestToolMessage {
                            content: ChatCompletionRequestToolMessageContent::Text(result),
                            tool_call_id: id,
                        }
                    ));
                }
            } else if has_content {
                return self.parse_classification(content);
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
        self.parse_classification(content)
    }

    fn parse_classification(&self, content: &str) -> Result<Classification> {
        let content = content.trim();
        
        if content.is_empty() {
            bail!("LLM returned empty response");
        }

        // Try the whole content first
        if let Ok(c) = serde_json::from_str::<Classification>(content) {
            return Ok(c);
        }

        // Strip markdown code block if present
        let stripped = content.strip_prefix("```json")
            .or_else(|| content.strip_prefix("```"))
            .and_then(|c| c.strip_suffix("```"))
            .map(|c| c.trim())
            .unwrap_or(content);

        if let Ok(c) = serde_json::from_str::<Classification>(stripped) {
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
                if let Ok(c) = serde_json::from_str::<Classification>(json_str) {
                    return Ok(c);
                }
            }
        }

        bail!("Failed to parse classification from response (first 200 chars): {:.200}", content)
    }

    async fn call_tool(&self, name: &str, arguments: &str) -> Result<String> {
        match name {
            "get_event" => {
                let args: serde_json::Value = serde_json::from_str(arguments)
                    .map_err(|e| anyhow::anyhow!("Invalid JSON arguments: {}", e))?;
                let event_id = args.get("event_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("Missing event_id"))?;

                match self.nostr.fetch_event(event_id).await? {
                    Some(event) => Ok(crate::format::describe_event(&event)),
                    None => Ok(format!("Event not found: {}", event_id)),
                }
            }
            "get_profile" => {
                let args: serde_json::Value = serde_json::from_str(arguments)
                    .map_err(|e| anyhow::anyhow!("Invalid JSON arguments: {}", e))?;
                let pubkey = args.get("pubkey")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("Missing pubkey"))?;

                match self.nostr.fetch_profile(pubkey).await? {
                    Some(profile) => Ok(crate::format::describe_profile(&profile)),
                    None => Ok(format!("Profile not found: {}", pubkey)),
                }
            }
            _ => Err(anyhow::anyhow!("Unknown tool: {}", name)),
        }
    }
    
    async fn encode_image(&self, path: &str) -> Result<String> {
        let bytes = tokio::fs::read(path).await?;
        if bytes.is_empty() {
            bail!("Failed to decode image");
        }
        
        // Load image
        let img = image::load_from_memory(&bytes)?;
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
            .encode(&img, img.width(), img.height(), image::ExtendedColorType::Rgb8)?;
        
        Ok(STANDARD.encode(&jpeg_bytes))
    }

    async fn generate_image_descriptions(&self, image_paths: &[String]) -> Result<String> {
        let mut descriptions = String::from("=== PROFILE IMAGES ===\n\n");
        
        for (i, path) in image_paths.iter().enumerate() {
            info!("Describing image {}/{}: {}", i + 1, image_paths.len(), path);
            match self.describe_image(path).await {
                Ok(desc) => {
                    info!("Image descr {}: {}", path, desc);
                    descriptions.push_str(&format!("Image {} description: {}\n\n", i + 1, desc));
                }
                Err(e) => {
                    tracing::warn!("Failed to describe image {}: {}", path, e);
                    descriptions.push_str(&format!("Image {} description: [Failed to generate description]\n\n", i + 1));
                }
            }
        }
        
        Ok(descriptions)
    }

    pub async fn describe_image(&self, path: &str) -> Result<String> {
        let image_content = format!("data:image/jpeg;base64,{}", self.encode_image(&path).await?);

        let request = CreateChatCompletionRequest {
            model: self.model.to_string(),
            messages: vec![
                ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
                    content: ChatCompletionRequestUserMessageContent::Array(vec![
                        ChatCompletionRequestUserMessageContentPart::Text(ChatCompletionRequestMessageContentPartText{
                            text: "Describe this image in detail. What is shown? What colors, objects, people, text, or scenes are visible? Be specific and objective. Keep description concise.".to_string(),
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
}

const CLASSIFICATION_PROMPT: &str = r#"Analyze this Nostr profile activity and generate a classification.

Consider:
- Posts: What topics does the user post about?
- Reactions: What content do they like/react to? This indicates their interests and agreements.
- Reposts: What content do they amplify?
- Zaps received: What content earns them tips?

If a PREVIOUS CLASSIFICATION section is present, use it as context. Refine and update it based on new activity — keep labels that still apply, remove ones that no longer fit, and add any new ones warranted by recent events.

You have access to tools (get_event, get_profile) to fetch additional context if needed.

Generate:
1. labels: 5-15 searchable tags (e.g., "rust developer", "bitcoin", "privacy advocate", "artist")
2. bio: A 2-3 sentence neutral summary of who they are based on activity
3. confidence: 0.0-1.0 indicating how confident you are in the classification

Output ONLY valid JSON with this exact structure:
{"labels": ["tag1", "tag2"], "bio": "summary text", "confidence": 0.85}"#;
