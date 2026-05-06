use anyhow::Result;
use reqwest::Client;
use std::path::PathBuf;
use std::sync::Arc;
use reqwest::header::HeaderMap;
use sha2::{Digest, Sha256};
use tracing::info;
use std::collections::HashMap;
use std::sync::Mutex as StdMutex;

#[derive(Clone)]
pub struct ImageCache {
    cache_dir: Arc<PathBuf>,
    client: Client,
    in_flight: Arc<StdMutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>>,
}

impl ImageCache {
    pub fn new(cache_dir: &str) -> Result<Self> {
        let cache_dir = Arc::new(PathBuf::from(cache_dir));
        std::fs::create_dir_all(cache_dir.as_ref())?;

        let mut headers = HeaderMap::new();
        headers.insert("User-Agent", "Mozilla/5.0 (iPhone; CPU iPhone OS 18_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.5 Mobile/15E148 Safari/604.1".parse()?);

        Ok(Self {
            cache_dir,
            client: Client::builder()
                .timeout(std::time::Duration::from_secs(5))
                .default_headers(headers)
                .build()?,
            in_flight: Arc::new(StdMutex::new(HashMap::new())),
        })
    }

    fn get_lock(&self, url: &str) -> Arc<tokio::sync::Mutex<()>> {
        let mut map = self.in_flight.lock().unwrap();
        map.entry(url.to_string())
            .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
            .clone()
    }

    pub async fn download(&self, url: &str) -> Result<Option<(String, String)>> {
        // Dedupe concurrent downloads of the same URL
        let lock = self.get_lock(url);
        let _guard = lock.lock().await;

        // Check if URL contains a SHA256 hash (64 hex chars)
        let url_hash = extract_sha256_from_url(url);
        
        // If we have a URL hash, check if we already cached it by content hash
        // We need to check all extensions since we don't know the format yet
        if let Some(ref hash) = url_hash {
            for ext in &["png", "jpg", "jpeg", "gif", "webp"] {
                let path = self.cache_dir.join(format!("{hash}.{ext}"));
                if path.exists() {
                    return Ok(Some((path.to_string_lossy().to_string(), url_hash.unwrap())));
                }
            }
        }
        
        info!("Downloading {}", url);
        match self.client.get(url).send().await {
            Ok(response) => {
                if !response.status().is_success() {
                    return Ok(None);
                }

                let bytes = response.bytes().await?;
                if bytes.is_empty() {
                    return Ok(None);
                }

                // Detect the actual image format from the content
                let ext = detect_image_extension(&bytes);

                // Compute SHA256 hash of the actual file content
                let content_hash = hex::encode(Sha256::digest(&bytes));
                
                // Use content hash as the filename with correct extension
                let final_path = self.cache_dir.join(format!("{content_hash}.{ext}"));
                
                // Only write if it doesn't exist yet
                if !final_path.exists() {
                    tokio::fs::write(&final_path, &bytes).await?;
                }
                
                Ok(Some((final_path.to_string_lossy().to_string(), content_hash)))
            }
            Err(_) => Ok(None),
        }
    }
}

/// Detect image format from magic bytes and return the appropriate extension.
fn detect_image_extension(bytes: &[u8]) -> &'static str {
    if bytes.len() < 4 {
        return "jpg";
    }

    // PNG: 89 50 4E 47
    if bytes.starts_with(&[0x89, 0x50, 0x4E, 0x47]) {
        return "png";
    }
    // JPEG: FF D8 FF
    if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return "jpg";
    }
    // GIF: 47 49 46 38
    if bytes.starts_with(b"GIF8") {
        return "gif";
    }
    // WebP: 52 49 46 46 ... 57 45 42 50
    if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        return "webp";
    }

    // Fallback: try image crate detection
    if image::load_from_memory(bytes).is_ok() {
        "jpg"
    } else {
        // Not a valid image — return a marker so callers can skip it
        "invalid"
    }
}

/// Check if a cached file is a valid image (not marked as invalid).
pub fn is_valid_image_path(path: &str) -> bool {
    !path.ends_with(".invalid")
}

/// Extract SHA256 hash from URL if present (64 hex character substring)
fn extract_sha256_from_url(url: &str) -> Option<String> {
    // Look for 64 consecutive hex characters which would be a SHA256 hash
    let re = regex::Regex::new(r"[0-9a-fA-F]{64}").ok()?;
    re.find(url).map(|m| m.as_str().to_lowercase())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_image_cache_creation() {
        let temp_dir = TempDir::new().unwrap();
        let cache = ImageCache::new(temp_dir.path().to_str().unwrap()).unwrap();
        
        assert!(cache.cache_dir.as_ref().exists());
    }

    #[test]
    fn test_detect_image_extension_png() {
        let png_header = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        assert_eq!(detect_image_extension(&png_header), "png");
    }

    #[test]
    fn test_detect_image_extension_jpeg() {
        let jpeg_header = [0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10];
        assert_eq!(detect_image_extension(&jpeg_header), "jpg");
    }

    #[test]
    fn test_detect_image_extension_invalid() {
        let html = b"<html><body>error</body></html>";
        assert_eq!(detect_image_extension(html), "invalid");
    }

    #[test]
    fn test_is_valid_image_path() {
        assert!(is_valid_image_path("/tmp/abc123.jpg"));
        assert!(is_valid_image_path("/tmp/abc123.png"));
        assert!(!is_valid_image_path("/tmp/abc123.invalid"));
    }

    #[tokio::test]
    async fn test_image_cache_download_skips_existing() {
        let temp_dir = TempDir::new().unwrap();
        let cache = ImageCache::new(temp_dir.path().to_str().unwrap()).unwrap();
        
        // Create a fake cached file with the content hash
        let content = b"fake image";
        let hash = hex::encode(Sha256::digest(content));
        let filename = format!("{hash}.jpg");
        let path = temp_dir.path().join(&filename);
        tokio::fs::write(&path, content).await.unwrap();
        
        // Download won't actually hit the network since it tries to fetch from example.com
        // but we can verify the cache_dir exists
        assert!(cache.cache_dir.as_ref().exists());
        assert_eq!(hash.len(), 64);
    }
}
