from pathlib import Path
import re

TIMESTAMP_PATTERN = re.compile(r"^\d{2}:\d{2}:\d{2}\.\d{3}\s+-->\s+\d{2}:\d{2}:\d{2}\.\d{3}$")


def clean_transcript_text(transcript: str) -> str:
    """Remove timestamp lines from a transcript and return only the spoken text."""
    lines = transcript.splitlines()
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if TIMESTAMP_PATTERN.match(stripped):
            continue
        cleaned_lines.append(stripped)

    return "\n".join(cleaned_lines)


def load_and_clean_transcripts(original_dir: Path | str,
                               new_dir: Path | str):
    original_path = Path(original_dir)
    new_path = Path(new_dir)

    new_path.mkdir(parents=True, exist_ok=True)

    transcripts = sorted(original_path.glob("*.txt"))
    for transcript_path in transcripts:
        file_content = transcript_path.read_text(encoding="utf-8")
        cleaned_content = clean_transcript_text(file_content)
        output_path = new_path / transcript_path.name
        output_path.write_text(cleaned_content, encoding="utf-8")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent

    load_and_clean_transcripts(
        repo_root / "data" / "raw",
        repo_root / "data" / "processed",
    )