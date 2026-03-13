from pathlib import Path
import json
import re
import os

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import WebshareProxyConfig
from youtube_transcript_api._errors import TranscriptsDisabled, VideoUnavailable


def extract_video_id(url_or_id: str) -> str:
    """Accept either a full YouTube URL or a raw video ID."""
    url_or_id = url_or_id.strip()

    if "youtube.com/watch?v=" in url_or_id:
        return url_or_id.split("v=")[1].split("&")[0]

    if "youtu.be/" in url_or_id:
        return url_or_id.split("youtu.be/")[1].split("?")[0]

    return url_or_id


def clean_snippet_text(text: str) -> str:
    """Remove subtitle formatting noise."""
    text = re.sub(r"\{\\an\d+\}", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_transcript_segments(video_id: str):
    """Fetch transcript and keep per-snippet timestamps."""
    proxy_config = WebshareProxyConfig(
        proxy_username=os.environ["WEBSHARE_USERNAME"],
        proxy_password=os.environ["WEBSHARE_PASSWORD"],
    )
    ytt_api = YouTubeTranscriptApi(proxy_config=proxy_config)
    transcript = ytt_api.fetch(video_id)

    segments = []
    for snippet in transcript:
        cleaned_text = clean_snippet_text(snippet.text)

        if cleaned_text:
            segments.append(
                {
                    "text": cleaned_text,
                    "start": float(snippet.start),
                    "duration": float(snippet.duration),
                }
            )

    return segments


def save_transcript_json(video_id: str, segments: list[dict]):
    """Save structured transcript with timestamps."""
    output_dir = Path("transcripts")
    output_dir.mkdir(exist_ok=True)

    file_path = output_dir / f"{video_id}.json"

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)

    print(f"Structured transcript saved to {file_path}")


if __name__ == "__main__":
    video_input = "nC4jOfpWV1E"  # can also be a full YouTube URL
    video_id = extract_video_id(video_input)

    try:
        segments = fetch_transcript_segments(video_id)
        save_transcript_json(video_id, segments)
        print(f"Total segments saved: {len(segments)}")

        if segments:
            print("\nFirst segment:\n")
            print(segments[0])

    except TranscriptsDisabled:
        print("This video does not have accessible subtitles. Pick another video with CC enabled.")
    except VideoUnavailable:
        print("This video is unavailable. Pick another public video.")
    except Exception as e:
        print(f"Unexpected error: {e}")