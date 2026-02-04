#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "mistral-common",
# ]
# ///
"""
Check what the actual token IDs are for special tokens in Voxtral.
"""

from mistral_common.tokens.tokenizers.tekken import Tekkenizer

def main():
    tokenizer = Tekkenizer.from_file("models/voxtral/tekken.json")

    # Find special tokens
    print("=== Special Token IDs ===")
    print(f"[STREAMING_PAD]: {tokenizer.get_special_token('[STREAMING_PAD]')}")
    print(f"[STREAMING_WORD]: {tokenizer.get_special_token('[STREAMING_WORD]')}")
    print(f"[AUDIO]: {tokenizer.get_special_token('[AUDIO]')}")
    print(f"[BEGIN_AUDIO]: {tokenizer.get_special_token('[BEGIN_AUDIO]')}")
    print(f"<unk>: {tokenizer.get_special_token('<unk>')}")
    print(f"<s>: {tokenizer.get_special_token('<s>')}")
    print(f"</s>: {tokenizer.get_special_token('</s>')}")

    print(f"\n=== Vocab Info ===")
    print(f"Vocab size: {tokenizer.n_words}")

    # Check what tokens 32 and 33 actually are
    print(f"\n=== Token 32 and 33 ===")
    print(f"Token 32 decodes to: {repr(tokenizer.decode([32]))}")
    print(f"Token 33 decodes to: {repr(tokenizer.decode([33]))}")

    # Check STREAMING_PAD token decoding
    streaming_pad_id = tokenizer.get_special_token('[STREAMING_PAD]')
    print(f"\n=== STREAMING_PAD Token ({streaming_pad_id}) ===")
    # Special tokens don't decode to text, but let's check
    try:
        print(f"Decodes to: {repr(tokenizer.decode([streaming_pad_id]))}")
    except Exception as e:
        print(f"Cannot decode: {e}")

if __name__ == "__main__":
    main()
