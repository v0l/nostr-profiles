use lru::LruCache;
use scraper::{Html, Selector};
use std::net::IpAddr;
use std::num::NonZeroUsize;
use std::sync::{Arc, LazyLock};
use std::time::Duration;
use tokio::sync::Mutex;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OpenGraphData {
    pub title: Option<String>,
    pub description: Option<String>,
    pub image: Option<String>,
    pub url: Option<String>,
    pub site_name: Option<String>,
    pub type_name: Option<String>,
}

static OG_SELECTOR: LazyLock<Selector> =
    LazyLock::new(|| Selector::parse("meta[property^='og:']").expect("invalid selector"));
static TITLE_SELECTOR: LazyLock<Selector> =
    LazyLock::new(|| Selector::parse("title").expect("invalid selector"));
static DESC_SELECTOR: LazyLock<Selector> =
    LazyLock::new(|| Selector::parse("meta[name='description']").expect("invalid selector"));

/// Validates a URL is safe to fetch (https scheme, public host).
pub fn validate_url(url: &str) -> Result<(), &'static str> {
    let parsed = match reqwest::Url::parse(url) {
        Ok(u) => u,
        Err(_) => return Err("invalid URL"),
    };

    if parsed.scheme() != "https" {
        return Err("only https URLs are allowed");
    }

    let host = match parsed.host_str() {
        Some(h) => h,
        None => return Err("missing host"),
    };

    // Reject numeric IPs that are private/loopback
    if let Ok(ip) = host.parse::<IpAddr>() {
        if is_private_ip(ip) {
            return Err("private/loopback IP addresses are not allowed");
        }
    }

    // Reject localhost and .local/.internal domains
    let lower = host.to_lowercase();
    if lower == "localhost" 
        || lower.ends_with(".local") 
        || lower.ends_with(".internal")
        || lower.ends_with(".lan")
        || lower.ends_with(".home")
    {
        return Err("local hostnames are not allowed");
    }

    Ok(())
}

fn is_private_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            v4.is_loopback()
                || v4.is_private()
                || v4.is_link_local()
                || v4.is_broadcast()
                || v4.is_documentation()
                || v4.is_unspecified()
        }
        IpAddr::V6(v6) => {
            v6.is_loopback()
                || v6.is_unspecified()
                || v6.is_multicast()
                || v6.is_loopback()
        }
    }
}

pub struct OpenGraphCache {
    cache: Arc<Mutex<LruCache<String, Option<OpenGraphData>>>>,
    client: reqwest::Client,
}

impl Clone for OpenGraphCache {
    fn clone(&self) -> Self {
        Self {
            cache: self.cache.clone(),
            client: self.client.clone(),
        }
    }
}

impl OpenGraphCache {
    pub fn new(max_size: usize) -> Self {
        let client = reqwest::Client::builder()
            .user_agent("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36 (OpenGraphFetcher)")
            .timeout(Duration::from_secs(10))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            cache: Arc::new(Mutex::new(LruCache::new(NonZeroUsize::new(max_size).unwrap()))),
            client,
        }
    }

    pub async fn get_preview(&self, url: &str) -> Option<OpenGraphData> {
        // Normalize URL to lowercase for cache key
        let cache_key = url.to_lowercase();

        // Check cache first
        {
            let mut cache = self.cache.lock().await;
            if let Some(cached) = cache.get(&cache_key) {
                return cached.clone();
            }
        }

        // Fetch and parse
        match self.fetch_and_parse(url).await {
            Ok(Some(data)) => {
                let mut cache = self.cache.lock().await;
                cache.put(cache_key, Some(data.clone()));
                Some(data)
            }
            Ok(None) => {
                let mut cache = self.cache.lock().await;
                cache.put(cache_key, None);
                None
            }
            Err(e) => {
                tracing::debug!("Failed to fetch preview for {}: {}", url, e);
                let mut cache = self.cache.lock().await;
                cache.put(cache_key, None);
                None
            }
        }
    }

    async fn fetch_and_parse(&self, url: &str) -> Result<Option<OpenGraphData>, reqwest::Error> {
        // Validate URL before fetching
        if let Err(_) = validate_url(url) {
            return Ok(None);
        }

        let response = self.client.get(url).send().await?;

        if !response.status().is_success() {
            return Ok(None);
        }

        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");

        // Only parse HTML pages
        if !content_type.starts_with("text/html") {
            return Ok(None);
        }

        let body = response.text().await?;
        let document = Html::parse_document(&body);

        // Parse OpenGraph tags
        let mut og_tags: std::collections::HashMap<String, String> = std::collections::HashMap::new();

        for element in document.select(&OG_SELECTOR) {
            if let (Some(property), Some(content)) = (
                element.value().attr("property"),
                element.value().attr("content"),
            ) {
                if !property.is_empty() && !content.is_empty() {
                    og_tags.insert(property.to_string(), content.to_string());
                }
            }
        }

        // Extract specific fields
        let title = og_tags
            .get("og:title")
            .cloned()
            .or_else(|| {
                document
                    .select(&TITLE_SELECTOR)
                    .next()
                    .map(|e| e.text().collect::<String>())
            });

        let description = og_tags
            .get("og:description")
            .cloned()
            .or_else(|| {
                document
                    .select(&DESC_SELECTOR)
                    .next()
                    .and_then(|e| e.value().attr("content"))
                    .map(|s| s.to_string())
            });

        let image = og_tags.get("og:image").cloned();
        let url = og_tags.get("og:url").cloned();
        let site_name = og_tags.get("og:site_name").cloned();
        let type_name = og_tags.get("og:type").cloned();

        // Only return if we have at least some data
        if title.is_none() && description.is_none() && image.is_none() {
            return Ok(None);
        }

        Ok(Some(OpenGraphData {
            title,
            description,
            image,
            url,
            site_name,
            type_name,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_url_https() {
        assert!(validate_url("https://example.com/page").is_ok());
    }

    #[test]
    fn test_validate_url_rejects_http() {
        assert!(validate_url("http://example.com/page").is_err());
    }

    #[test]
    fn test_validate_url_rejects_loopback() {
        assert!(validate_url("https://127.0.0.1/secret").is_err());
    }

    #[test]
    fn test_validate_url_rejects_private() {
        assert!(validate_url("https://192.168.0.1/").is_err());
    }

    #[test]
    fn test_validate_url_rejects_localhost() {
        assert!(validate_url("https://localhost/api").is_err());
    }

    #[test]
    fn test_validate_url_rejects_local_domain() {
        assert!(validate_url("https://myservice.local/").is_err());
        assert!(validate_url("https://db.internal/").is_err());
    }

    #[test]
    fn test_validate_url_invalid() {
        assert!(validate_url("not a url").is_err());
    }

    /// Live test: fetches real OpenGraph data from the network.
    /// Run with: cargo test opengraph_live -- --ignored
    #[tokio::test]
    #[ignore]
    async fn opengraph_live_fetch() {
        let cache = OpenGraphCache::new(64);

        // Test a well-known page that has OG tags
        let data = cache
            .get_preview("https://github.com/v0l/nostr-profiles")
            .await
            .expect("should get a result for github.com");

        // GitHub pages always have og:title and og:description
        assert!(data.title.is_some(), "GitHub page should have og:title");
        println!("Title: {:?}", data.title);
        println!("Description: {:?}", data.description);
        println!("Site: {:?}", data.site_name);
        println!("Type: {:?}", data.type_name);
        println!("Image: {:?}", data.image);

        // Test cache hit — second call should return the same data without hitting network
        let data2 = cache
            .get_preview("https://github.com/v0l/nostr-profiles")
            .await;
        assert!(data2.is_some(), "cache hit should return data");
        assert_eq!(data2.unwrap().title, data.title, "cached data should match");

        // Test a page that likely has no OG tags (e.g. a raw file)
        let none = cache.get_preview("https://example.com/nonexistent").await;
        // example.com may or may not have OG tags, just verify it doesn't panic
        println!("example.com result: {:?}", none);

        // Test SSRF rejection — should return None without hitting network
        let private_result = cache.get_preview("https://192.168.1.1/").await;
        assert!(private_result.is_none(), "should reject private IP");

        let localhost_result = cache.get_preview("https://localhost/").await;
        assert!(localhost_result.is_none(), "should reject localhost");

        // Test a non-HTML URL — should return None
        let image_result = cache.get_preview("https://avatars.githubusercontent.com/u/134780083").await;
        // May or may not have OG data depending on response content-type
        println!("Image URL result: {:?}", image_result);
    }
}
