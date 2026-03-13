import json
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_transcript(file_path):
    """Load transcript segments from JSON."""
    with open(file_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    return segments


def split_transcript_segments(segments, video_id):
    """
    Combine transcript subtitle segments into larger semantic chunks
    while preserving the timestamp of the first segment in each chunk.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    full_text = ""
    segment_map = []

    for segment in segments:
        text = segment["text"].strip()
        if not text:
            continue

        start_time = segment["start"]

        if full_text:
            full_text += " "

        start_index = len(full_text)
        full_text += text
        end_index = len(full_text)

        segment_map.append({
            "start_index": start_index,
            "end_index": end_index,
            "timestamp": start_time
        })

    split_texts = splitter.split_text(full_text)

    chunks = []
    cursor = 0

    for chunk in split_texts:
        chunk_start_index = full_text.find(chunk, cursor)
        if chunk_start_index == -1:
            chunk_start_index = cursor

        chunk_end_index = chunk_start_index + len(chunk)
        cursor = chunk_end_index

        chunk_timestamp = 0.0
        for seg in segment_map:
            if seg["start_index"] <= chunk_start_index < seg["end_index"]:
                chunk_timestamp = seg["timestamp"]
                break

        chunks.append({
            "text": chunk,
            "metadata": {
                "video_id": video_id,
                "timestamp": chunk_timestamp
            }
        })

    return chunks


if __name__ == "__main__":
    video_id = "nC4jOfpWV1E"
    file_path = f"transcripts/{video_id}.json"

    segments = load_transcript(file_path)
    chunks = split_transcript_segments(segments, video_id)

    print(f"\nTotal chunks created: {len(chunks)}")

    if chunks:
        print("\nFirst chunk:\n")
        print(chunks[0])