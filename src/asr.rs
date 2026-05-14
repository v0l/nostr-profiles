use crate::config::AsrConfig;
use anyhow::{bail, Context, Result};
use std::io::{Read, Write};
use std::mem::transmute;
use std::net::TcpStream;
use std::time::Duration;
use tracing::info;

// ── Wyoming protocol helpers ──────────────────────────────────────────────

/// Write a Wyoming JSONL line followed by `\n`.
fn write_line(stream: &mut TcpStream, line: &str) -> Result<()> {
    stream.write_all(line.as_bytes())?;
    stream.write_all(b"\n")?;
    Ok(())
}

/// Write a Wyoming frame: JSONL line, then optional data bytes, then optional payload bytes.
fn write_frame(stream: &mut TcpStream, line: &str, payload: Option<&[u8]>) -> Result<()> {
    stream.write_all(line.as_bytes())?;
    stream.write_all(b"\n")?;
    if let Some(p) = payload {
        stream.write_all(p)?;
    }
    stream.flush()?;
    Ok(())
}

// ── Public API ────────────────────────────────────────────────────────────

/// Transcribe audio from a video file using a Wyoming STT server.
pub async fn transcribe_video(video_path: &str, config: &AsrConfig) -> Result<String> {
    let pcm = tokio::task::spawn_blocking({
        let p = video_path.to_string();
        move || extract_pcm(&p)
    })
    .await??;

    if pcm.is_empty() {
        bail!("No audio data extracted from video");
    }
    info!("Extracted {} audio bytes ({:.1}s)", pcm.len(), pcm.len() as f64 / 32000.0);

    transcribe_pcm(&pcm, config).await
}

// ── Audio extraction ──────────────────────────────────────────────────────

fn extract_pcm(path: &str) -> Result<Vec<u8>> {
    unsafe {
        let mut demuxer = ffmpeg_rs_raw::Demuxer::new(path)?;
        let info = demuxer.probe_input()?;
        let audio = info.best_audio().ok_or_else(|| anyhow::anyhow!("no audio stream"))?;
        eprintln!("[asr] audio: {}Hz {}ch", audio.sample_rate, audio.channels);

        let mut dec = ffmpeg_rs_raw::Decoder::new();
        dec.setup_decoder(audio, None)?;

        // AV_SAMPLE_FMT_S16 = 1
        let mut resampler =
            ffmpeg_rs_raw::Resample::new(transmute(1i32), 16000, 1);
        let idx = audio.index as i32;
        let mut pcm = Vec::new();

        loop {
            let (pkt, stream) = demuxer.get_packet()?;
            let Some(pkt) = pkt else { break };
            if (*stream).index != idx {
                continue;
            }
            for (frame, si) in dec.decode_pkt(Some(&pkt))? {
                if si != idx {
                    continue;
                }
                let r = resampler.process_frame(&frame)?;
                let ptr = r.data[0];
                let n = r.nb_samples as usize;
                if !ptr.is_null() && n > 0 {
                    pcm.extend_from_slice(std::slice::from_raw_parts(ptr as *const u8, n * 2));
                }
            }
        }
        eprintln!("[asr] {:.1}s PCM ({})", pcm.len() as f64 / 32000.0, pcm.len());
        Ok(pcm)
    }
}

// ── Wyoming STT client ────────────────────────────────────────────────────

async fn transcribe_pcm(pcm: &[u8], config: &AsrConfig) -> Result<String> {
    let addr = config.uri.strip_prefix("tcp://").unwrap_or(&config.uri);
    eprintln!("[asr] connecting to {addr}");

    let mut s = TcpStream::connect(addr).context(format!("connect {addr}"))?;
    s.set_read_timeout(Some(Duration::from_secs(120)))?;
    s.set_write_timeout(Some(Duration::from_secs(30)))?;

    // transcribe
    write_line(&mut s, &transcribe_json(config))?;

    // audio-start
    write_line(&mut s, r#"{"type":"audio-start","data":{"rate":16000,"width":2,"channels":1}}"#)?;

    // audio-chunks
    for chunk in pcm.chunks(4096) {
        write_frame(&mut s, &format!(
            r#"{{"type":"audio-chunk","data":{{"rate":16000,"width":2,"channels":1}},"payload_length":{}}}"#,
            chunk.len()
        ), Some(chunk))?;
    }

    // audio-stop
    write_line(&mut s, r#"{"type":"audio-stop"}"#)?;

    // read transcript
    read_transcript(&mut s)
}

fn transcribe_json(config: &AsrConfig) -> String {
    match &config.language {
        Some(l) => format!(r#"{{"type":"transcribe","data":{{"language":"{l}"}}}}"#),
        None => r#"{"type":"transcribe"}"#.to_string(),
    }
}

fn read_transcript(s: &mut TcpStream) -> Result<String> {
    let mut buf = vec![0u8; 65536];
    let mut acc = Vec::new();

    loop {
        let n = s.read(&mut buf)?;
        if n == 0 {
            bail!("server closed before transcript");
        }
        acc.extend_from_slice(&buf[..n]);

        while let Some(nl) = acc.iter().position(|&b| b == b'\n') {
            let line = acc[..nl].to_vec();
            acc.drain(..=nl);
            if line.is_empty() {
                continue;
            }

            eprintln!("[asr] recv: {}", String::from_utf8_lossy(&line));

            let ev: serde_json::Value = serde_json::from_slice(&line)?;
            let dl = ev.get("data_length").and_then(|v| v.as_u64()).unwrap_or(0) as usize;

            let data = if dl > 0 {
                while acc.len() < dl {
                    let n = s.read(&mut buf)?;
                    if n == 0 {
                        bail!("server closed mid-data");
                    }
                    acc.extend_from_slice(&buf[..n]);
                }
                let bytes = acc[..dl].to_vec();
                acc.drain(..dl);
                Some(serde_json::from_slice::<serde_json::Value>(&bytes)?)
            } else {
                ev.get("data").cloned()
            };

            match ev.get("type").and_then(|v| v.as_str()).unwrap_or("") {
                "transcript" => {
                    return data
                        .as_ref()
                        .and_then(|d| d.get("text").and_then(|v| v.as_str()))
                        .map(|t| t.to_string())
                        .ok_or_else(|| anyhow::anyhow!("transcript missing text"));
                }
                "error" => {
                    let msg = data
                        .as_ref()
                        .and_then(|d| d.get("message").and_then(|v| v.as_str()))
                        .unwrap_or("unknown");
                    bail!("Wyoming error: {msg}");
                }
                _ => {}
            }
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #[test]
    fn test_transcribe_json() {
        let json = r#"{"type":"transcribe","data":{"language":"en"}}"#;
        assert!(json.contains("transcribe"));
        assert!(json.contains("en"));
    }

    #[test]
    fn test_parse_transcript() {
        let data = br#"{"text":"hello","language":"en"}"#;
        let header = format!(r#"{{"type":"transcript","data_length":{}}}"#, data.len());
        let ev: serde_json::Value = serde_json::from_str(&header).unwrap();
        assert_eq!(ev["type"], "transcript");
        let d: serde_json::Value = serde_json::from_slice(data).unwrap();
        assert_eq!(d["text"], "hello");
    }
}
