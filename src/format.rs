use crate::db::Profile;

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
    let mut s = String::new();
    s.push_str(&format!("Author: {}\n", event.pubkey.to_hex()));
    s.push_str(&format!("Kind: {}\n", event.kind.as_u16()));
    if !event.content.is_empty() {
        s.push_str(&format!("Content: {}\n", event.content));
    }
    s.push_str(&format!("Created: {}\n", event.created_at.as_secs()));
    let tags = serde_json::to_string(&event.tags).unwrap_or_default();
    s.push_str(&format!("Tags: {}\n", tags));
    s
}
