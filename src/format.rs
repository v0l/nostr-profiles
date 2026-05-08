use crate::db::Profile;
use crate::opengraph::OpenGraphData;

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
    s.push_str(&format!("Kind: {} ({})\n", kind, kind_name(kind)));
    s.push_str(&format!("Author: {}\n", event.pubkey.to_hex()));
    if !event.content.is_empty() {
        s.push_str(&format!("Content: {}\n", event.content));
    }
    s.push_str(&format!("Created: {}\n", event.created_at.as_secs()));
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
