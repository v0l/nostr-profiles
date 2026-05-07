use anyhow::{bail, Result};
use ffmpeg_rs_raw::{Decoder, Demuxer, Scaler};
use ffmpeg_rs_raw::ffmpeg_sys_the_third::AVPixelFormat;
use image::{DynamicImage, ImageBuffer, RgbImage};
use std::path::Path;

/// Number of frames to extract from a video for the collage.
const NUM_FRAMES: usize = 20;

/// Maximum dimension (width or height) for each frame in the collage.
const MAX_FRAME_DIM: usize = 256;

/// Extract key frames from a video file and stitch them into a single collage image.
///
/// The collage is a horizontal strip of `NUM_FRAMES` frames evenly spaced through
/// the video. Each frame is scaled down to fit within `MAX_FRAME_DIM` pixels on
/// its longest side while preserving aspect ratio.
pub fn extract_video_collage(video_path: &str) -> Result<String> {
    let output_path = format!("{}.collage.jpg", video_path);

    // Check if collage already exists
    if Path::new(&output_path).exists() {
        return Ok(output_path);
    }

    let frames = extract_frames(video_path)?;
    if frames.is_empty() {
        bail!("No frames could be extracted from video");
    }

    let collage = stitch_collage(&frames);
    collage.save(&output_path)?;

    // Clean up individual frame files
    for frame_path in &frames {
        let _ = std::fs::remove_file(frame_path);
    }

    Ok(output_path)
}

/// Extract `NUM_FRAMES` evenly-spaced frames from a video file.
/// Returns paths to the extracted JPEG frame files.
fn extract_frames(video_path: &str) -> Result<Vec<String>> {
    unsafe {
        let mut demuxer = Demuxer::new(video_path)?;
        let media_info = demuxer.probe_input()?;

        let video_stream = media_info
            .best_video()
            .ok_or_else(|| anyhow::anyhow!("No video stream found"))?;

        let duration = media_info.duration;
        if duration <= 0.0 {
            bail!("Video has no duration or duration is zero");
        }

        let video_stream_idx = video_stream.index as i32;
        let width = video_stream.width;
        let height = video_stream.height;
        if width == 0 || height == 0 {
            bail!("Video stream has zero dimensions");
        }

        // Set up decoder for the video stream
        let mut decoder = Decoder::new();
        decoder.setup_decoder(video_stream, None)?;

        // Calculate seek points — evenly spaced through the video
        let mut seek_timestamps: Vec<f32> = Vec::new();
        for i in 0..NUM_FRAMES {
            let t = duration * (i as f32 + 1.0) / (NUM_FRAMES as f32 + 1.0);
            seek_timestamps.push(t);
        }

        let mut extracted_frames: Vec<String> = Vec::new();
        let mut next_seek_idx: usize = 0;

        loop {
            let (pkt, stream) = demuxer.get_packet()?;
            let pkt = match pkt {
                Some(p) => p,
                None => break,
            };

            // Only decode video stream
            if (*stream).index != video_stream_idx {
                continue;
            }

            let decoded = decoder.decode_pkt(Some(&pkt))?;

            for (frame, stream_idx) in decoded {
                if stream_idx != video_stream_idx {
                    continue;
                }

                // Check if this frame is near our next seek point
                let pts_secs = frame.pts as f32
                    * video_stream.timebase.0 as f32
                    / video_stream.timebase.1 as f32;

                if next_seek_idx < seek_timestamps.len()
                    && pts_secs >= seek_timestamps[next_seek_idx]
                {
                    // Scale and save this frame
                    let mut scaler = Scaler::new();
                    let (tw, th) = fit_dimensions(width, height);
                    let scaled = scaler.process_frame(
                        &frame,
                        tw,
                        th,
                        AVPixelFormat::RGB24,
                    )?;

                    let frame_path = format!("{}_frame_{}.jpg", video_path, next_seek_idx);
                    save_frame_as_jpeg(&scaled, &frame_path)?;
                    extracted_frames.push(frame_path);
                    next_seek_idx += 1;

                    if extracted_frames.len() >= NUM_FRAMES {
                        return Ok(extracted_frames);
                    }
                }
            }
        }

        // Flush decoder
        let flushed = decoder.flush()?;
        for (frame, stream_idx) in flushed {
            if stream_idx != video_stream_idx {
                continue;
            }

            if next_seek_idx < seek_timestamps.len() {
                let pts_secs = frame.pts as f32
                    * video_stream.timebase.0 as f32
                    / video_stream.timebase.1 as f32;

                if pts_secs >= seek_timestamps[next_seek_idx] {
                    let mut scaler = Scaler::new();
                    let (tw, th) = fit_dimensions(width, height);
                    let scaled = scaler.process_frame(
                        &frame,
                        tw,
                        th,
                        AVPixelFormat::RGB24,
                    )?;

                    let frame_path = format!("{}_frame_{}.jpg", video_path, next_seek_idx);
                    save_frame_as_jpeg(&scaled, &frame_path)?;
                    extracted_frames.push(frame_path);
                    next_seek_idx += 1;
                }
            }
        }

        Ok(extracted_frames)
    }
}

/// Compute target dimensions that fit within MAX_FRAME_DIM while preserving aspect ratio.
fn fit_dimensions(width: usize, height: usize) -> (u16, u16) {
    if width <= MAX_FRAME_DIM && height <= MAX_FRAME_DIM {
        return (width as u16, height as u16);
    }
    let ratio = MAX_FRAME_DIM as f64 / width.max(height) as f64;
    let w = (width as f64 * ratio).round() as u16;
    let h = (height as f64 * ratio).round() as u16;
    (w.max(1), h.max(1))
}
fn save_frame_as_jpeg(frame: &ffmpeg_rs_raw::AvFrameRef, path: &str) -> Result<()> {
    unsafe {
        let width = frame.width as usize;
        let height = frame.height as usize;
        let linesize = frame.linesize[0] as usize;
        let data = std::slice::from_raw_parts(frame.data[0], (height - 1) * linesize + width * 3);

        let mut img: RgbImage = ImageBuffer::new(width as u32, height as u32);
        for y in 0..height {
            for x in 0..width {
                let offset = y * linesize + x * 3;
                let r = data[offset];
                let g = data[offset + 1];
                let b = data[offset + 2];
                img.put_pixel(x as u32, y as u32, image::Rgb([r, g, b]));
            }
        }

        let dynamic = DynamicImage::ImageRgb8(img);
        dynamic.save(path)?;
        Ok(())
    }
}

/// Stitch multiple frame images into a grid collage that's roughly square.
fn stitch_collage(frame_paths: &[String]) -> DynamicImage {
    let mut images: Vec<DynamicImage> = frame_paths
        .iter()
        .filter_map(|p| image::open(p).ok())
        .collect();

    if images.is_empty() {
        return DynamicImage::ImageRgb8(ImageBuffer::new(1, 1));
    }

    // Compute grid dimensions to make a roughly square collage
    let n = images.len() as u32;
    let cols = (n as f64).sqrt().ceil() as u32;
    let rows = (n + cols - 1) / cols;

    // Scale all frames to the same size — use the smallest frame's dimensions
    let cell_w = images.iter().map(|i| i.width()).min().unwrap_or(MAX_FRAME_DIM as u32);
    let cell_h = images.iter().map(|i| i.height()).min().unwrap_or(MAX_FRAME_DIM as u32);
    for img in &mut images {
        *img = img.resize_exact(cell_w, cell_h, image::imageops::FilterType::Lanczos3);
    }

    let gap = 2u32;
    let canvas_w = cols * cell_w + (cols - 1) * gap;
    let canvas_h = rows * cell_h + (rows - 1) * gap;

    let mut canvas = RgbImage::new(canvas_w, canvas_h);
    for (idx, img) in images.iter().enumerate() {
        let col = idx as u32 % cols;
        let row = idx as u32 / cols;
        let x_offset = col * (cell_w + gap);
        let y_offset = row * (cell_h + gap);
        let rgb_img = img.to_rgb8();
        for (x, y, pixel) in rgb_img.enumerate_pixels() {
            canvas.put_pixel(x_offset + x, y_offset + y, *pixel);
        }
    }

    DynamicImage::ImageRgb8(canvas)
}

/// Check if a URL looks like a video based on its extension.
pub fn is_video_url(url: &str) -> bool {
    let lower = url.to_lowercase();
    // Remove query strings before checking
    let path = lower.split('?').next().unwrap_or("");
    path.ends_with(".mp4")
        || path.ends_with(".webm")
        || path.ends_with(".mov")
        || path.ends_with(".avi")
        || path.ends_with(".mkv")
        || path.ends_with(".m4v")
        || path.ends_with(".3gp")
        || path.ends_with(".ogv")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_video_url() {
        assert!(is_video_url("https://example.com/video.mp4"));
        assert!(is_video_url("https://example.com/video.webm?token=abc"));
        assert!(is_video_url("https://example.com/clip.MOV"));
        assert!(!is_video_url("https://example.com/image.jpg"));
        assert!(!is_video_url("https://example.com/image.png"));
        assert!(!is_video_url("https://example.com/doc.pdf"));
    }

    #[tokio::test]
    #[ignore]
    async fn test_extract_video_collage() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let video_path = tmp_dir.path().join("test_video.mp4");
        let video_path_str = video_path.to_string_lossy().to_string();

        // Download the test video
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .unwrap();
        let resp = client
            .get("https://nostr.download/60b4b7ea312421f325e4a497d5636abf2428da31e1ed2081a2e4f3c5b52ddaf0.mp4")
            .send()
            .await
            .unwrap();
        assert!(resp.status().is_success(), "Failed to download test video: {}", resp.status());
        let bytes = resp.bytes().await.unwrap();
        tokio::fs::write(&video_path, &bytes).await.unwrap();

        // Extract collage (blocking operation)
        let result = tokio::task::spawn_blocking(move || {
            extract_video_collage(&video_path_str)
        }).await.unwrap();
        assert!(result.is_ok(), "extract_video_collage failed: {:?}", result.err());

        let collage_path = result.unwrap();
        assert!(Path::new(&collage_path).exists(), "Collage file not created");

        // Verify the collage is a valid image
        let img = image::open(&collage_path).expect("Collage is not a valid image");
        assert!(img.width() > 0);
        assert!(img.height() > 0);

        // The collage should be wider than tall (horizontal strip)
        assert!(img.width() > img.height(), "Collage should be a horizontal strip, got {}x{}", img.width(), img.height());

        // Copy to project dir for viewing
        let out = std::env::current_dir().unwrap().join("collage.jpg");
        std::fs::copy(&collage_path, &out).unwrap();
    }
}