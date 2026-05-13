use crate::db::Profile;
use crate::opengraph::OpenGraphData;

/// Maximum characters of event content to include before truncating.
/// Long-form posts can exceed 50k chars and blow up LLM context windows.
const CONTENT_MAX_CHARS: usize = 4000;

/// Human-readable names for common Nostr event kinds.
pub fn kind_name(kind: u16) -> &'static str {
    match kind {
        0 => "Metadata",
        1 => "Short Text Note",
        6 => "Repost",
        7 => "Reaction",
        16 => "Generic Repost",
        17 => "Reaction to Website",
        20 => "Picture",
        21 => "Video",
        22 => "Short-form Portrait Video",
        1111 => "Comment",
        9734 => "Zap Request",
        9735 => "Zap Receipt",
        9802 => "Highlight",
        30023 => "Long-form Content",
        34235 => "Addressable Normal Video",
        34236 => "Addressable Short Video",
        _ => "Unknown",
    }
}

pub fn describe_profile(p: &Profile) -> String {
    let mut s = String::new();
    s.push_str(&format!("Pubkey: {}\n", p.pubkey));
    if let Some(nip05) = &p.nip05 {
        s.push_str(&format!("NIP-05: {}\n", nip05));
    }
    if let Some(name) = &p.name {
        s.push_str(&format!("Name: {}\n", name));
    }
    if let Some(about) = &p.about {
        s.push_str(&format!("Bio: {}\n", about));
    }
    if let Some(picture) = &p.picture {
        s.push_str(&format!("Profile Image: {}\n", picture));
    }
    s
}

pub fn describe_event(event: &nostr_sdk::Event) -> String {
    let kind = event.kind.as_u16();
    let mut s = String::new();
    s.push_str(&format!("ID: {}\n", event.id.to_hex()));
    s.push_str(&format!("Kind: {} ({})\n", kind, kind_name(kind)));
    s.push_str(&format!("Author: {}\n", event.pubkey.to_hex()));

    // For media events, extract structured metadata from tags
    let is_media = matches!(kind, 20 | 21 | 22 | 34235 | 34236);
    if is_media {
        // Title tag
        if let Some(title) = event.tags.iter().find_map(|t| {
            let vals = t.as_slice();
            if vals.len() >= 2 && vals[0] == "title" {
                Some(vals[1].as_str())
            } else {
                None
            }
        }) {
            s.push_str(&format!("Title: {}\n", title));
        }

        // Content warning
        if let Some(cw) = event.tags.iter().find_map(|t| {
            let vals = t.as_slice();
            if vals.len() >= 2 && vals[0] == "content-warning" {
                Some(vals[1].as_str())
            } else {
                None
            }
        }) {
            s.push_str(&format!("Content Warning: {}\n", cw));
        }

        // Hashtags
        let hashtags: Vec<&str> = event
            .tags
            .iter()
            .filter_map(|t| {
                let vals = t.as_slice();
                if vals.len() >= 2 && vals[0] == "t" {
                    Some(vals[1].as_str())
                } else {
                    None
                }
            })
            .collect();
        if !hashtags.is_empty() {
            s.push_str(&format!("Hashtags: {}\n", hashtags.join(", ")));
        }

        // Parse imeta tags — extract image URLs (kind 20) and video URLs (kind 21/22/34235/34236)
        let mut image_urls: Vec<&str> = Vec::new();
        let mut video_urls: Vec<&str> = Vec::new();
        let mut thumbnail_urls: Vec<&str> = Vec::new();
        let mut alt_texts: Vec<&str> = Vec::new();
        let mut dimensions: Vec<&str> = Vec::new();

        for tag in event.tags.iter() {
            let vals = tag.as_slice();
            if vals.len() >= 2 && vals[0] == "imeta" {
                let mut url: Option<&str> = None;
                let mut mime: Option<&str> = None;
                let mut alt: Option<&str> = None;
                let mut dim: Option<&str> = None;
                let mut thumbnails: Vec<&str> = Vec::new();

                for entry in &vals[1..] {
                    if let Some(u) = entry.strip_prefix("url ") {
                        url = Some(u);
                    } else if let Some(m) = entry.strip_prefix("m ") {
                        mime = Some(m);
                    } else if let Some(a) = entry.strip_prefix("alt ") {
                        alt = Some(a);
                    } else if let Some(d) = entry.strip_prefix("dim ") {
                        dim = Some(d);
                    } else if let Some(i) = entry.strip_prefix("image ") {
                        thumbnails.push(i);
                    }
                }

                if let Some(u) = url {
                    if let Some(m) = mime {
                        if m.starts_with("video/") || m == "application/x-mpegURL" {
                            video_urls.push(u);
                        } else if m.starts_with("image/") {
                            image_urls.push(u);
                        } else {
                            image_urls.push(u);
                        }
                    } else {
                        if crate::video::is_video_url(u) {
                            video_urls.push(u);
                        } else {
                            image_urls.push(u);
                        }
                    }
                }

                for img in thumbnails {
                    thumbnail_urls.push(img);
                }
                if let Some(a) = alt {
                    alt_texts.push(a);
                }
                if let Some(d) = dim {
                    dimensions.push(d);
                }
            }
        }

        if !image_urls.is_empty() {
            s.push_str(&format!("Images: {}\n", image_urls.join(", ")));
        }
        if !video_urls.is_empty() {
            s.push_str(&format!("Videos: {}\n", video_urls.join(", ")));
        }
        if !thumbnail_urls.is_empty() {
            s.push_str(&format!("Thumbnails: {}\n", thumbnail_urls.join(", ")));
        }
        if !alt_texts.is_empty() {
            s.push_str(&format!("Alt text: {}\n", alt_texts.join("; ")));
        }
        if !dimensions.is_empty() {
            s.push_str(&format!("Dimensions: {}\n", dimensions.join(", ")));
        }
    }

    if !event.content.is_empty() {
        let content_len = event.content.chars().count();
        if content_len > CONTENT_MAX_CHARS {
            // Truncate, breaking at the last whitespace before the cutoff
            let truncated: String = event.content.chars().take(CONTENT_MAX_CHARS).collect();
            // rfind returns a byte index, which is safe because truncated is a standalone String
            let break_byte = truncated
                .rfind(|c: char| c.is_whitespace())
                .unwrap_or(truncated.len());
            let snippet = &truncated[..break_byte];
            s.push_str(&format!(
                "Content: {}\n[content truncated: {} total chars, showing first {}]\n",
                snippet,
                content_len,
                snippet.chars().count()
            ));
        } else {
            s.push_str(&format!("Content: {}\n", event.content));
        }
    }
    s.push_str(&format!("Created: {}\n", event.created_at.as_secs()));

    // For non-media events, dump raw tags as before.
    // For media events, we already parsed the important tags above,
    // but still include them for completeness.
    let tags = serde_json::to_string(&event.tags).unwrap_or_default();
    s.push_str(&format!("Tags: {}\n", tags));
    s
}

pub fn describe_opengraph(data: &OpenGraphData) -> String {
    let mut s = String::new();
    if let Some(title) = &data.title {
        s.push_str(&format!("Title: {}\n", title));
    }
    if let Some(desc) = &data.description {
        s.push_str(&format!("Description: {}\n", desc));
    }
    if let Some(site) = &data.site_name {
        s.push_str(&format!("Site: {}\n", site));
    }
    if let Some(type_name) = &data.type_name {
        s.push_str(&format!("Type: {}\n", type_name));
    }
    if let Some(url) = &data.url {
        s.push_str(&format!("URL: {}\n", url));
    }
    if let Some(image) = &data.image {
        s.push_str(&format!("Image: {}\n", image));
    }
    if s.is_empty() {
        s.push_str("(no OpenGraph data)\n");
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kind_name() {
        assert_eq!(kind_name(0), "Metadata");
        assert_eq!(kind_name(1), "Short Text Note");
        assert_eq!(kind_name(7), "Reaction");
        assert_eq!(kind_name(9735), "Zap Receipt");
        assert_eq!(kind_name(999), "Unknown");
    }
}
