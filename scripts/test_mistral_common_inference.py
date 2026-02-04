#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "mistral-common[soxr]>=1.9.0",
#     "torch",
#     "soundfile",
#     "numpy",
#     "safetensors",
# ]
# ///
"""
Test inference using mistral-common's tokenizer to see how streaming mode actually works.
"""

import sys
import torch
import torch.nn.functional as F
import soundfile as sf
import numpy as np
from safetensors import safe_open
from pathlib import Path

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.transcription.request import TranscriptionRequest, StreamingMode
from mistral_common.protocol.instruct.chunk import RawAudio
from mistral_common.audio import Audio
import base64

MODEL_PATH = "models/voxtral/consolidated.safetensors"
TOKENIZER_PATH = "models/voxtral/tekken.json"

def main():
    if len(sys.argv) < 2:
        print("Usage: test_mistral_common_inference.py <audio.wav>")
        sys.exit(1)

    audio_path = sys.argv[1]

    # Load audio
    print(f"Loading audio: {audio_path}")
    audio, sr = sf.read(audio_path)
    print(f"  Duration: {len(audio)/sr:.2f}s at {sr}Hz")

    # Convert to WAV bytes for mistral-common
    import io
    import struct
    audio_bytes_io = io.BytesIO()
    audio_bytes_io.write(struct.pack('<4s', b'RIFF'))
    audio_bytes_io.write(struct.pack('<I', 36 + len(audio) * 2))
    audio_bytes_io.write(struct.pack('<4s', b'WAVE'))
    audio_bytes_io.write(struct.pack('<4s', b'fmt '))
    audio_bytes_io.write(struct.pack('<I', 16))
    audio_bytes_io.write(struct.pack('<H', 1))  # PCM
    audio_bytes_io.write(struct.pack('<H', 1))  # mono
    audio_bytes_io.write(struct.pack('<I', sr))
    audio_bytes_io.write(struct.pack('<I', sr * 2))
    audio_bytes_io.write(struct.pack('<H', 2))
    audio_bytes_io.write(struct.pack('<H', 16))
    audio_bytes_io.write(struct.pack('<4s', b'data'))
    audio_bytes_io.write(struct.pack('<I', len(audio) * 2))
    audio_int16 = (audio * 32767).astype(np.int16)
    audio_bytes_io.write(audio_int16.tobytes())
    audio_bytes_data = audio_bytes_io.getvalue()

    # Create Audio object
    audio_obj = Audio.from_bytes(audio_bytes_data)
    print(f"  Audio object: {audio_obj.duration:.2f}s")

    # Load tokenizer
    print("\nLoading mistral tokenizer...")
    tokenizer = MistralTokenizer.from_file(TOKENIZER_PATH)

    # Create transcription request
    print("\nEncoding transcription request...")
    raw_audio = RawAudio.from_audio(audio_obj)
    request = TranscriptionRequest(
        model="voxtral",
        audio=raw_audio,
        language="en",
        streaming=StreamingMode.ONLINE,
    )

    # Encode the request
    try:
        tokenized = tokenizer.encode_transcription(request)
        print(f"\n=== Tokenized Output ===")
        print(f"Tokens: {tokenized.tokens[:50]}... ({len(tokenized.tokens)} total)")
        print(f"Token count: {len(tokenized.tokens)}")

        # Print all available attributes
        print(f"\nTokenized attributes: {[a for a in dir(tokenized) if not a.startswith('_')]}")

        # Check audio content
        if hasattr(tokenized, 'audios') and tokenized.audios:
            print(f"\nAudio embeddings ({len(tokenized.audios)}):")
            for i, audio in enumerate(tokenized.audios):
                if hasattr(audio, 'shape'):
                    print(f"  Audio {i}: shape={audio.shape}, dtype={audio.dtype}")
                elif hasattr(audio, 'signal'):
                    sig = audio.signal
                    print(f"  Audio {i}: signal len={len(sig) if hasattr(sig, '__len__') else 'N/A'}")
                else:
                    print(f"  Audio {i}: {type(audio)}")

        # Check for encoder outputs
        if hasattr(tokenized, 'audio_signals') and tokenized.audio_signals:
            print(f"\nAudio signals ({len(tokenized.audio_signals)}):")
            for i, sig in enumerate(tokenized.audio_signals):
                print(f"  Signal {i}: shape={sig.shape if hasattr(sig, 'shape') else len(sig)}")

        # Print token breakdown
        print(f"\nToken values (first 50): {tokenized.tokens[:50]}")
        unique_tokens = set(tokenized.tokens)
        print(f"Unique tokens: {unique_tokens}")

        # Check prefix_ids
        if hasattr(tokenized, 'prefix_ids'):
            print(f"\nPrefix IDs: {tokenized.prefix_ids}")

        # Check audio object in detail
        if tokenized.audios:
            audio = tokenized.audios[0]
            print(f"\nAudio object details:")
            print(f"  Type: {type(audio)}")
            print(f"  Attributes: {[a for a in dir(audio) if not a.startswith('_')]}")
            if hasattr(audio, 'signal'):
                sig = audio.signal
                print(f"  Signal shape: {sig.shape if hasattr(sig, 'shape') else len(sig)}")
                print(f"  Signal type: {type(sig)}")
            if hasattr(audio, 'duration'):
                print(f"  Duration: {audio.duration}s")
            if hasattr(audio, 'sample_rate'):
                print(f"  Sample rate: {audio.sample_rate}")

        # The key question: how many positions does the audio produce after encoder?
        # Audio duration * 12.5 Hz = expected positions
        print(f"\n=== Expected sequence lengths ===")
        audio_duration = 6.6
        mel_frames = audio_duration * 100  # 100 mel frames/second (16kHz, hop=160)
        encoder_pos = mel_frames / 4  # After 2x stride conv layers
        adapter_pos = encoder_pos / 4  # After 4x reshape
        print(f"  Audio duration: {audio_duration:.2f}s")
        print(f"  Mel frames: {mel_frames:.0f}")
        print(f"  After encoder (4x): {encoder_pos:.0f}")
        print(f"  After adapter (4x): {adapter_pos:.0f}")
        print(f"  Text tokens (prefix): {len(tokenized.tokens)}")
        print(f"\n  MISMATCH: {adapter_pos:.0f} audio positions vs {len(tokenized.tokens)} text tokens!")

    except Exception as e:
        print(f"Error encoding: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
