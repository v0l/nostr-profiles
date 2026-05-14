//! End-to-end classification test.
//!
//! Uses the local config.yaml, inserts events into a temp DB,
//! runs the real classifier, and asserts the LLM picks up interests
//! from reaction signals.
//!
//! Run with: cargo test e2e_ -- --ignored

#![allow(unused_imports, dead_code)]

use crate::classifier::Classifier;
use crate::config::Config;
use crate::db::Database;
use crate::image_cache::ImageCache;
use crate::job_queue::build_context;
use crate::nostr_client::NostrClient;
use crate::profile_cache::ProfileCache;
use anyhow::Result;
use nostr_sdk::JsonUtil;
use std::sync::Arc;

fn make_event(keys: &nostr_sdk::Keys, kind: u16, content: &str, tags: Vec<nostr_sdk::Tag>, created_at: u64) -> nostr_sdk::Event {
    nostr_sdk::EventBuilder::new(nostr_sdk::Kind::from(kind), content)
        .custom_created_at(nostr_sdk::Timestamp::from(created_at))
        .tags(tags)
        .sign_with_keys(keys)
        .unwrap()
}

/// Alice only has reactions — she reacts to a bitcoin post and a nostr dev post.
/// The reacted-to posts are cached in DB so get_event can resolve them.
/// Classification should pick up bitcoin/nostr labels from her reactions.
#[tokio::test]
#[ignore]
async fn e2e_classify_reaction_infers_interest() -> Result<()> {
    rustls::crypto::ring::default_provider()
        .install_default()
        .expect("Failed to install rustls crypto provider");

    let config = Config::load("config.yaml")?;

    let db_dir = tempfile::tempdir()?;
    let db = Arc::new(Database::new(db_dir.path().join("test.db").to_str().unwrap(), config.labels.min_score).await?);

    let alice = nostr_sdk::Keys::generate();
    let bob = nostr_sdk::Keys::generate();
    let charlie = nostr_sdk::Keys::generate();

    // Bob writes a bitcoin post
    let bob_post = make_event(
        &bob, 1,
        "Just stacked another 100k sats. Bitcoin is the only sound money. #bitcoin",
        vec![], 1700000000,
    );
    // Charlie writes a nostr dev post
    let charlie_post = make_event(
        &charlie, 1,
        "Shipped a new NIP-96 blob upload implementation in Rust. #nostr",
        vec![], 1700000100,
    );
    // Alice reacts to both
    let alice_reaction_bob = make_event(
        &alice, 7, "👍",
        vec![nostr_sdk::Tag::event(bob_post.id), nostr_sdk::Tag::public_key(bob.public_key)],
        1700000200,
    );
    let alice_reaction_charlie = make_event(
        &alice, 7, "🤙",
        vec![nostr_sdk::Tag::event(charlie_post.id), nostr_sdk::Tag::public_key(charlie.public_key)],
        1700000300,
    );
    // Add more reactions to strengthen the signal
    let bob_post2 = make_event(
        &bob, 1,
        "Lightning Network routing fees are negligible. Self-custody is king. #lightning #bitcoin",
        vec![], 1700000400,
    );
    let alice_reaction_bob2 = make_event(
        &alice, 7, "⚡",
        vec![nostr_sdk::Tag::event(bob_post2.id), nostr_sdk::Tag::public_key(bob.public_key)],
        1700000500,
    );
    // Alice has her own post but it's completely unrelated
    let alice_post = make_event(
        &alice, 1,
        "Beautiful sunrise this morning. Coffee hits different on the porch.",
        vec![], 1700000600,
    );
    // Alice's profile — with a profile picture so describe_image gets called
    let alice_meta = make_event(
        &alice, 0,
        &serde_json::json!({"name": "Alice", "about": "Just vibeing", "picture": "https://nostr.download/db83482b79fef80f35f75106db1909a3876edf939688243cac90a10288e2c39b.jpg"}).to_string(),
        vec![], 1699990000,
    );

    // Cache everything — reactions AND the posts they reference
    for e in [&alice_meta, &alice_reaction_bob, &alice_reaction_charlie, &alice_reaction_bob2, &alice_post, &bob_post, &charlie_post, &bob_post2] {
        db.cache_event(e).await?;
    }

    let alice_pk = alice.public_key.to_hex();

    // Build context the same way process_job does
    let events = db.get_profile_events(&alice_pk, 50).await?;
    let profile = db.get_profile_details(&alice_pk).await.ok();
    let prev = db.get_classification_if_exists(&alice_pk).await.ok().flatten();
    let context = build_context(&profile, &events, &prev);

    assert!(context.contains("Kind: 7 (Reaction)"), "context should show reactions");

    // Verify get_event works for the referenced posts
    for eid in [bob_post.id.to_hex(), charlie_post.id.to_hex(), bob_post2.id.to_hex()] {
        let found = db.get_event(&eid).await?;
        assert!(found.is_some(), "referenced event should be in DB cache");
    }

    // Set up classifier from config
    let nostr = NostrClient::new(&config.nostr.relays, config.nostr.nsec.as_deref()).await?;
    nostr.connect().await;

    let image_cache_dir = tempfile::tempdir()?;
    let image_cache = ImageCache::new(image_cache_dir.path().to_str().unwrap(), None)?;
    let profile_cache = ProfileCache::new(db.clone(), nostr.clone(), 7);
    let og_cache = crate::opengraph::OpenGraphCache::new(128);

    let classifier = Classifier::new(
        &config.llm,
        nostr, profile_cache, image_cache, og_cache, db,
        crate::config::load_label_taxonomy(config.labels.taxonomy_file.as_deref()),
        config.labels.min_score,
        std::time::Duration::from_secs(config.processing.tool_call_timeout_secs),
        config.chat_logs.dir.clone(),
    );

    let result = classifier.classify(&alice_pk, &context).await?;

    println!("=== Classification ===");
    println!("Bio: {}", result.classification.bio);
    println!("Confidence: {:.2}", result.classification.confidence);
    let mut sorted: Vec<_> = result.classification.scores.iter().collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    for (label, score) in sorted.iter().take(10) {
        println!("  {}: {:.2}", label, score);
    }

    let btc = result.classification.scores.get("bitcoin").copied().unwrap_or(0.0);
    let nostr_e = result.classification.scores.get("nostr-enthusiast").copied().unwrap_or(0.0);
    let nostr_d = result.classification.scores.get("nostr-developer").copied().unwrap_or(0.0);

    assert!(
        btc >= 0.3 || nostr_e >= 0.3 || nostr_d >= 0.3,
        "Alice reacted to bitcoin + nostr posts — should score >= 0.3 on a related label. bitcoin={btc:.2}, nostr-enthusiast={nostr_e:.2}, nostr-developer={nostr_d:.2}"
    );

    Ok(())
}

/// Unit test: reaction e-tags point to cached events in the DB.
#[tokio::test]
async fn test_reaction_references_cached_event() -> Result<()> {
    let db_dir = tempfile::tempdir()?;
    let db = Database::new(db_dir.path().join("test.db").to_str().unwrap(), 0.4).await?;

    let alice = nostr_sdk::Keys::generate();
    let bob = nostr_sdk::Keys::generate();

    let bob_post = make_event(&bob, 1, "Bitcoin is the future of money", vec![], 1700000000);
    let alice_reaction = make_event(
        &alice, 7, "👍",
        vec![nostr_sdk::Tag::event(bob_post.id), nostr_sdk::Tag::public_key(bob.public_key)],
        1700000100,
    );

    db.cache_event(&bob_post).await?;
    db.cache_event(&alice_reaction).await?;

    let events = db.get_profile_events(&alice.public_key.to_hex(), 50).await?;
    assert_eq!(events.len(), 1);
    let reaction = nostr_sdk::Event::from_json(&events[0].raw_json)?;
    assert_eq!(reaction.kind.as_u16(), 7);

    let e_tag = reaction.tags.iter().find_map(|t| {
        match t.as_standardized()? {
            nostr_sdk::TagStandard::Event { event_id, .. } => Some(event_id.to_hex()),
            _ => None,
        }
    }).unwrap();

    let cached = db.get_event(&e_tag).await?.unwrap();
    let cached_event = nostr_sdk::Event::from_json(&cached.raw_json)?;
    assert_eq!(cached_event.content, "Bitcoin is the future of money");

    Ok(())
}

/// End-to-end test: downloads a video, runs ASR transcription via Wyoming, then classifies
/// a profile that posted the video. The transcript should inform the classification.
///
/// Requires a Wyoming STT server running at the URI configured in config.yaml.
/// Run with: cargo test e2e_asr_video -- --ignored --nocapture
#[tokio::test]
#[ignore]
async fn e2e_asr_video_transcribe_and_classify() -> Result<()> {
    rustls::crypto::ring::default_provider()
        .install_default()
        .expect("Failed to install rustls crypto provider");

    let config = Config::load("config.yaml")?;
    let asr_config = config
        .processing
        .asr
        .as_ref()
        .expect("ASR must be configured in config.yaml for this test");

    println!("ASR config: uri={}, lang={:?}", asr_config.uri, asr_config.language);

    // --- Step 1: Download video, extract collage + ASR transcript ---
    let cache_dir = tempfile::tempdir()?;
    let image_cache =
        ImageCache::new(cache_dir.path().to_str().unwrap(), Some(asr_config.clone()))?;

    let video_url = "https://nostr.download/e8d6ff9dba138146f8d3cd26ea36a736093bbf99f6c9f8eff19bcd182c0f437c.mp4";
    println!("Downloading video: {}", video_url);

    let (path, _hash) = image_cache.download(video_url).await?.expect("Failed to download video");
    println!("Cached collage at: {}", path);
    assert!(path.contains(".video.collage.jpg"), "expected collage path, got {path}");
    assert!(
        std::path::Path::new(&path).exists(),
        "collage file should exist at {path}"
    );

    // Check transcript exists (may fail if Wyoming server isn't running)
    let transcript_path = path.replace(".video.collage.jpg", ".video.transcript.txt");
    let transcript = match std::fs::read_to_string(&transcript_path) {
        Ok(t) => t.trim().to_string(),
        Err(e) => {
            eprintln!("ASR transcript not found at {transcript_path}: {e}");
            eprintln!("Skipping classification — Wyoming STT server may not be running.");
            return Ok(());
        }
    };
    println!("ASR transcript ({} chars): {}", transcript.len(), &transcript[..transcript.len().min(500)]);
    assert!(!transcript.is_empty(), "transcript should not be empty");

    // --- Step 2: Classify (best-effort, may time out on slow LLMs) ---
    let db_dir = tempfile::tempdir()?;
    let db = Arc::new(
        Database::new(
            db_dir.path().join("test.db").to_str().unwrap(),
            config.labels.min_score,
        )
        .await?,
    );

    let alice = nostr_sdk::Keys::generate();
    let alice_meta = make_event(
        &alice, 0,
        &serde_json::json!({"name": "Alice", "about": "I post videos"}).to_string(),
        vec![], 1699990000,
    );
    let imeta_tag = nostr_sdk::Tag::parse([
        "imeta",
        &format!("url {}", video_url),
        "m video/mp4",
    ]).unwrap();
    let video_post = make_event(
        &alice, 22, "Check out this clip!",
        vec![imeta_tag], 1700000000,
    );

    db.cache_event(&alice_meta).await?;
    db.cache_event(&video_post).await?;

    let alice_pk = alice.public_key.to_hex();
    let events = db.get_profile_events(&alice_pk, 50).await?;
    let profile = db.get_profile_details(&alice_pk).await.ok();
    let context = build_context(&profile, &events, &None);
    println!("Context ({}) chars", context.len());

    let nostr = NostrClient::new(&config.nostr.relays, config.nostr.nsec.as_deref()).await?;
    nostr.connect().await;

    let profile_cache = ProfileCache::new(db.clone(), nostr.clone(), 7);
    let og_cache = crate::opengraph::OpenGraphCache::new(128);

    let classifier = Classifier::new(
        &config.llm,
        nostr, profile_cache, image_cache, og_cache, db,
        crate::config::load_label_taxonomy(config.labels.taxonomy_file.as_deref()),
        config.labels.min_score,
        std::time::Duration::from_secs(config.processing.tool_call_timeout_secs),
        config.chat_logs.dir.clone(),
    );

    match classifier.classify(&alice_pk, &context).await {
        Ok(result) => {
            println!("=== Classification ===");
            println!("Bio: {}", result.classification.bio);
            println!("Confidence: {:.2}", result.classification.confidence);
            let mut sorted: Vec<_> = result.classification.scores.iter().collect();
            sorted.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
            for (label, score) in sorted.iter().take(15) {
                println!("  {}: {:.2}", label, score);
            }
            assert!(!result.classification.bio.is_empty(), "bio should not be empty");
        }
        Err(e) => {
            eprintln!("Classification failed (non-critical for ASR test): {e}");
        }
    }

    Ok(())
}
