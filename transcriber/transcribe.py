"""
Audio transcription with speaker diarization.

Transcribes audio files using Whisper and identifies speakers using pyannote.audio.
Outputs timestamped transcript with speaker labels.

The output format is compatible with the analyzer pipeline:
    [HH:MM:SS] SPEAKER_NAME:
    Transcribed text...
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import timedelta

from faster_whisper import WhisperModel
from pyannote.audio import Pipeline


# =========================
# CONFIG
# =========================

# Default model sizes
DEFAULT_WHISPER_MODEL = "medium"  # Options: tiny, base, small, medium, large-v2, large-v3
DEFAULT_DEVICE = "cpu"            # Options: cpu, cuda
DEFAULT_COMPUTE_TYPE = "int8"     # Options: int8, float16, float32


# =========================
# HELPERS
# =========================

def _format_timestamp(seconds):
    """
    Convert seconds to HH:MM:SS format.

    Args:
        seconds: Time in seconds (float or int)

    Returns:
        str: Formatted timestamp (e.g., "1:23:45")
    """
    return str(timedelta(seconds=int(seconds)))


def _find_speaker(start_time, diarization):
    """
    Find speaker label for a given timestamp.

    Args:
        start_time: Timestamp in seconds (float)
        diarization: pyannote diarization result

    Returns:
        str: Speaker label (e.g., "SPEAKER_00") or "UNKNOWN"
    """
    for turn, _, speaker_label in diarization.itertracks(yield_label=True):
        if turn.start <= start_time <= turn.end:
            return speaker_label
    return "UNKNOWN"


# =========================
# MAIN PIPELINE
# =========================

def transcribe_audio(audio_path, output_path, whisper_model, hf_token, device="cpu", compute_type="int8"):
    """
    Transcribe audio file with speaker diarization.

    Process:
    1. Load Whisper model for transcription
    2. Transcribe audio to segments
    3. Load pyannote pipeline for speaker diarization
    4. Match each transcript segment to a speaker
    5. Write formatted output

    Args:
        audio_path: Path to input audio file (Path or str)
        output_path: Path to output transcript file (Path or str)
        whisper_model: Whisper model size (str)
        hf_token: HuggingFace token for pyannote models (str)
        device: Device to use - "cpu" or "cuda" (str)
        compute_type: Compute type - "int8", "float16", "float32" (str)

    Raises:
        FileNotFoundError: If audio file doesn't exist
        RuntimeError: If models fail to load
        ValueError: If HF_TOKEN is missing

    Output format:
        [HH:MM:SS] SPEAKER_NAME:
        Transcribed text...

        [HH:MM:SS] SPEAKER_NAME:
        More text...
    """
    audio_path = Path(audio_path)
    output_path = Path(output_path)

    # Validate inputs
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is required for speaker diarization")

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load Whisper model
    print(f"▶ Loading Whisper model ({whisper_model})...")
    try:
        model = WhisperModel(whisper_model, device=device, compute_type=compute_type)
    except Exception as e:
        raise RuntimeError(f"Failed to load Whisper model: {e}") from e

    # Transcribe audio
    print("▶ Transcribing audio...")
    try:
        segments, info = model.transcribe(str(audio_path), language="cs")
        # Convert generator to list to allow multiple iterations
        segments = list(segments)
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {e}") from e

    print(f"  Detected language: {info.language} (probability: {info.language_probability:.2f})")

    # Load diarization pipeline
    print("▶ Loading speaker diarization model...")
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            use_auth_token=hf_token
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load diarization model: {e}") from e

    # Perform diarization
    print("▶ Identifying speakers...")
    try:
        diarization = pipeline(str(audio_path))
    except Exception as e:
        raise RuntimeError(f"Speaker diarization failed: {e}") from e

    # Write output
    print(f"▶ Writing transcript to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for segment in segments:
            start_time = segment.start
            text = segment.text.strip()

            # Skip empty segments
            if not text:
                continue

            # Find speaker for this segment
            speaker = _find_speaker(start_time, diarization)

            # Write formatted output
            timestamp = _format_timestamp(start_time)
            f.write(f"[{timestamp}] {speaker}:\n{text}\n\n")

    print("✅ Transcription complete!")


def main():
    """
    Main entry point for audio transcriber.

    Transcribes audio files using Whisper and identifies speakers using pyannote.

    Command line arguments:
        --audio, -i: Path to input audio file (required)
        --output, -o: Path to output transcript file (required)
        --model: Whisper model size (default: medium)
        --device: Device to use - cpu or cuda (default: cpu)
        --compute-type: Compute type (default: int8)

    Environment variables:
        HF_TOKEN: HuggingFace token (required for speaker diarization)
        WHISPER_MODEL: Override default Whisper model size
    """
    parser = argparse.ArgumentParser(
        description="Transcribe audio with speaker diarization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python transcribe.py --audio input.mp3 --output transcript.txt

  # With custom model
  python transcribe.py -i audio.opus -o output.txt --model large-v3

  # Using GPU
  python transcribe.py -i audio.wav -o output.txt --device cuda --compute-type float16

Environment variables:
  HF_TOKEN         HuggingFace API token (required)
  WHISPER_MODEL    Default Whisper model size
        """
    )

    parser.add_argument(
        "--audio", "-i",
        type=Path,
        required=True,
        help="Input audio file path"
    )

    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output transcript file path"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("WHISPER_MODEL", DEFAULT_WHISPER_MODEL),
        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        help=f"Whisper model size (default: {DEFAULT_WHISPER_MODEL})"
    )

    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        choices=["cpu", "cuda"],
        help=f"Device to use (default: {DEFAULT_DEVICE})"
    )

    parser.add_argument(
        "--compute-type",
        type=str,
        default=DEFAULT_COMPUTE_TYPE,
        choices=["int8", "float16", "float32"],
        help=f"Compute type (default: {DEFAULT_COMPUTE_TYPE})"
    )

    args = parser.parse_args()

    # Get HuggingFace token from environment
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        parser.error("HF_TOKEN environment variable is required. Get your token at https://huggingface.co/settings/tokens")

    # Run transcription
    try:
        transcribe_audio(
            audio_path=args.audio,
            output_path=args.output,
            whisper_model=args.model,
            hf_token=hf_token,
            device=args.device,
            compute_type=args.compute_type
        )
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
