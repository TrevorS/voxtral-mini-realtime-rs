//! CLI for Voxtral transcription.

use anyhow::Result;
use clap::Parser;

#[derive(Parser)]
#[command(name = "voxtral-transcribe")]
#[command(about = "Transcribe audio using Voxtral Mini 4B Realtime")]
struct Cli {
    /// Path to audio file (WAV format)
    #[arg(short, long)]
    audio: String,

    /// Path to model directory
    #[arg(short, long, default_value = "models/voxtral-mini-4b")]
    model: String,

    /// Delay in tokens (1 token = 80ms)
    #[arg(short, long, default_value = "6")]
    delay: usize,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    println!("Loading model from: {}", cli.model);
    println!("Processing audio: {}", cli.audio);
    println!("Delay: {} tokens ({}ms)", cli.delay, cli.delay * 80);

    // TODO: Implement actual transcription
    println!("\nTranscription not yet implemented - model loading in progress");

    Ok(())
}
