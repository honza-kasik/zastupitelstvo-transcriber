"""
Complete meeting processing pipeline.

Processes audio recordings through the entire pipeline:
1. Transcription (audio → text with speaker diarization)
2. Analysis (text → structured topics)
3. Article generation (topics → LLM prompt + Jekyll draft)

This is the main entry point for processing meeting recordings.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime


# =========================
# CONFIG
# =========================

# Script paths
TRANSCRIBER_SCRIPT = Path("transcriber/transcribe.py")
ANALYZER_SCRIPT = Path("analyzer/analyze_meeting_topics.py")
ARTICLE_GENERATOR_SCRIPT = Path("article-generator/generate_meeting_article.py")

# Default output directory
DEFAULT_OUTPUT_DIR = Path("output")


# =========================
# PIPELINE STEPS
# =========================

def run_transcription(audio_file, output_dir, whisper_model="medium", device="cpu", compute_type="int8"):
    """
    Run transcription step.

    Args:
        audio_file: Path to input audio file
        output_dir: Output directory path
        whisper_model: Whisper model size
        device: Device to use (cpu/cuda)
        compute_type: Compute type (int8/float16/float32)

    Returns:
        Path: Path to generated transcript file

    Raises:
        RuntimeError: If transcription fails
    """
    transcript_file = output_dir / "transcript.txt"

    print("\n" + "="*60)
    print("STEP 1/3: TRANSCRIPTION")
    print("="*60)
    print(f"Input: {audio_file}")
    print(f"Output: {transcript_file}")
    print(f"Model: {whisper_model}")
    print(f"Device: {device}")
    print()

    cmd = [
        sys.executable,
        str(TRANSCRIBER_SCRIPT),
        "--audio", str(audio_file),
        "--output", str(transcript_file),
        "--model", whisper_model,
        "--device", device,
        "--compute-type", compute_type
    ]

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Transcription failed with exit code {result.returncode}")

    if not transcript_file.exists():
        raise RuntimeError(f"Transcript file not created: {transcript_file}")

    print(f"\n✅ Transcription complete: {transcript_file}")
    return transcript_file


def run_analysis(transcript_file, output_dir):
    """
    Run analysis step.

    Args:
        transcript_file: Path to transcript file
        output_dir: Output directory path

    Returns:
        Path: Path to generated llm_input.json file

    Raises:
        RuntimeError: If analysis fails
    """
    llm_input_file = output_dir / "llm_input.json"

    print("\n" + "="*60)
    print("STEP 2/3: ANALYSIS")
    print("="*60)
    print(f"Input: {transcript_file}")
    print(f"Output: {output_dir}")
    print()

    cmd = [
        sys.executable,
        str(ANALYZER_SCRIPT),
        "--file", str(transcript_file),
        "--outdir", str(output_dir)
    ]

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Analysis failed with exit code {result.returncode}")

    if not llm_input_file.exists():
        raise RuntimeError(f"LLM input file not created: {llm_input_file}")

    print(f"\n✅ Analysis complete: {llm_input_file}")
    return llm_input_file


def run_article_generation(llm_input_file, output_dir, meeting_date, meeting_number, layout="meeting"):
    """
    Run article generation step.

    Args:
        llm_input_file: Path to llm_input.json
        output_dir: Output directory path
        meeting_date: Meeting date (YYYY-MM-DD)
        meeting_number: Meeting sequence number
        layout: Jekyll layout name

    Returns:
        dict: Paths to generated files (prompt_path, jekyll_path)

    Raises:
        RuntimeError: If article generation fails
    """
    print("\n" + "="*60)
    print("STEP 3/3: ARTICLE GENERATION")
    print("="*60)
    print(f"Input: {llm_input_file}")
    print(f"Date: {meeting_date}")
    print(f"Number: {meeting_number}")
    print(f"Output: {output_dir}")
    print()

    cmd = [
        sys.executable,
        str(ARTICLE_GENERATOR_SCRIPT),
        "--topics", str(llm_input_file),
        "--date", meeting_date,
        "--number", str(meeting_number),
        "--outdir", str(output_dir),
        "--layout", layout
    ]

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Article generation failed with exit code {result.returncode}")

    prompt_file = output_dir / "llm_prompt.txt"
    jekyll_file = output_dir / "jekyll_draft.md"

    if not prompt_file.exists():
        raise RuntimeError(f"LLM prompt file not created: {prompt_file}")
    if not jekyll_file.exists():
        raise RuntimeError(f"Jekyll draft file not created: {jekyll_file}")

    print(f"\n✅ Article generation complete")
    return {
        "prompt_path": prompt_file,
        "jekyll_path": jekyll_file
    }


# =========================
# MAIN PIPELINE
# =========================

def process_meeting(
    audio_file,
    meeting_date,
    meeting_number,
    output_dir=None,
    whisper_model="medium",
    device="cpu",
    compute_type="int8",
    layout="meeting",
    skip_transcription=False,
    skip_analysis=False
):
    """
    Run complete meeting processing pipeline.

    Args:
        audio_file: Path to input audio file
        meeting_date: Meeting date in YYYY-MM-DD format
        meeting_number: Meeting sequence number
        output_dir: Output directory (default: ./output)
        whisper_model: Whisper model size
        device: Device for transcription (cpu/cuda)
        compute_type: Compute type (int8/float16/float32)
        layout: Jekyll layout name
        skip_transcription: Skip transcription step (use existing transcript)
        skip_analysis: Skip analysis step (use existing analysis)

    Returns:
        dict: Paths to all generated files

    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If parameters are invalid
        RuntimeError: If any pipeline step fails
    """
    audio_file = Path(audio_file)
    output_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR

    # Validate inputs
    if not skip_transcription and not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    try:
        datetime.strptime(meeting_date, "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"Invalid date format: {meeting_date}. Use YYYY-MM-DD.")

    if meeting_number < 1:
        raise ValueError(f"Meeting number must be positive: {meeting_number}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("MEETING PROCESSING PIPELINE")
    print("="*60)
    print(f"Audio: {audio_file}")
    print(f"Date: {meeting_date}")
    print(f"Meeting: #{meeting_number}")
    print(f"Output: {output_dir}")
    print("="*60)

    results = {}

    # Step 1: Transcription
    if skip_transcription:
        transcript_file = output_dir / "transcript.txt"
        print(f"\n⏭️  Skipping transcription, using: {transcript_file}")
        if not transcript_file.exists():
            raise FileNotFoundError(f"Transcript file not found: {transcript_file}")
    else:
        transcript_file = run_transcription(
            audio_file=audio_file,
            output_dir=output_dir,
            whisper_model=whisper_model,
            device=device,
            compute_type=compute_type
        )
    results["transcript"] = str(transcript_file)

    # Step 2: Analysis
    if skip_analysis:
        llm_input_file = output_dir / "llm_input.json"
        print(f"\n⏭️  Skipping analysis, using: {llm_input_file}")
        if not llm_input_file.exists():
            raise FileNotFoundError(f"LLM input file not found: {llm_input_file}")
    else:
        llm_input_file = run_analysis(
            transcript_file=transcript_file,
            output_dir=output_dir
        )
    results["llm_input"] = str(llm_input_file)

    # Step 3: Article Generation
    article_files = run_article_generation(
        llm_input_file=llm_input_file,
        output_dir=output_dir,
        meeting_date=meeting_date,
        meeting_number=meeting_number,
        layout=layout
    )
    results.update(article_files)

    # Final summary
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETE")
    print("="*60)
    print(f"\n📁 Output directory: {output_dir}")
    print(f"\n📄 Generated files:")
    print(f"  - Transcript:    {results['transcript']}")
    print(f"  - LLM Input:     {results['llm_input']}")
    print(f"  - LLM Prompt:    {results['prompt_path']}")
    print(f"  - Jekyll Draft:  {results['jekyll_path']}")

    print(f"\n⚡ Next steps:")
    print(f"  1. Send {output_dir}/llm_prompt.txt to your preferred LLM")
    print(f"  2. Copy LLM output into {output_dir}/jekyll_draft.md")
    print(f"  3. Review and publish")

    return results


def main():
    """
    Main entry point for meeting processing pipeline.

    Processes meeting recordings from audio to article draft in a single command.

    Command line arguments:
        --audio, -i: Input audio file (required)
        --date, -d: Meeting date YYYY-MM-DD (required)
        --number, -n: Meeting sequence number (required)
        --outdir, -o: Output directory (default: ./output)
        --model: Whisper model size (default: medium)
        --device: Device cpu/cuda (default: cpu)
        --compute-type: Compute precision (default: int8)
        --layout: Jekyll layout (default: meeting)
        --skip-transcription: Skip transcription step
        --skip-analysis: Skip analysis step

    Pipeline steps:
        1. Transcription: audio → timestamped transcript with speakers
        2. Analysis: transcript → topics with metadata
        3. Article generation: topics → LLM prompt + Jekyll draft
    """
    parser = argparse.ArgumentParser(
        description="Process meeting audio through complete pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - process complete meeting
  python process_meeting.py \\
    --audio input/meeting.opus \\
    --date 2025-01-15 \\
    --number 23

  # Use GPU for faster transcription
  python process_meeting.py \\
    -i input/meeting.opus \\
    -d 2025-01-15 \\
    -n 23 \\
    --device cuda

  # Skip transcription (if already done)
  python process_meeting.py \\
    -i input/meeting.opus \\
    -d 2025-01-15 \\
    -n 23 \\
    --skip-transcription

  # Custom output directory
  python process_meeting.py \\
    -i input/meeting.opus \\
    -d 2025-01-15 \\
    -n 23 \\
    -o meetings/meeting-23/

Pipeline:
  [Audio] → Transcriber → [Transcript] → Analyzer → [Topics] → Article Gen → [Draft]

Output files:
  - transcript.txt: Timestamped transcript with speaker labels
  - topics.json: Full topic analysis with metadata
  - llm_input.json: Filtered topics for LLM
  - llm_prompt.txt: Structured prompt for LLM
  - jekyll_draft.md: Jekyll page template

Time estimates (2 hour audio on CPU):
  - Transcription: 10-12 hours
  - Analysis: 2-5 minutes
  - Article generation: < 1 second
        """
    )

    parser.add_argument(
        "--audio", "-i",
        type=Path,
        required=True,
        help="Input audio file"
    )

    parser.add_argument(
        "--date", "-d",
        type=str,
        required=True,
        help="Meeting date (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--number", "-n",
        type=int,
        required=True,
        help="Meeting sequence number"
    )

    parser.add_argument(
        "--outdir", "-o",
        type=Path,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="medium",
        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        help="Whisper model size (default: medium)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for transcription (default: cpu)"
    )

    parser.add_argument(
        "--compute-type",
        type=str,
        default="int8",
        choices=["int8", "float16", "float32"],
        help="Compute precision (default: int8)"
    )

    parser.add_argument(
        "--layout",
        type=str,
        default="meeting",
        help="Jekyll layout name (default: meeting)"
    )

    parser.add_argument(
        "--skip-transcription",
        action="store_true",
        help="Skip transcription step (use existing transcript.txt)"
    )

    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip analysis step (use existing llm_input.json)"
    )

    args = parser.parse_args()

    # Check that required scripts exist
    missing_scripts = []
    for script in [TRANSCRIBER_SCRIPT, ANALYZER_SCRIPT, ARTICLE_GENERATOR_SCRIPT]:
        if not script.exists():
            missing_scripts.append(str(script))

    if missing_scripts:
        print("❌ Error: Required scripts not found:", file=sys.stderr)
        for script in missing_scripts:
            print(f"  - {script}", file=sys.stderr)
        sys.exit(1)

    # Run pipeline
    try:
        process_meeting(
            audio_file=args.audio,
            meeting_date=args.date,
            meeting_number=args.number,
            output_dir=args.outdir,
            whisper_model=args.model,
            device=args.device,
            compute_type=args.compute_type,
            layout=args.layout,
            skip_transcription=args.skip_transcription,
            skip_analysis=args.skip_analysis
        )
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
