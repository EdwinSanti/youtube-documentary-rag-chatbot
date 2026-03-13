from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from src.transcript_loader import extract_video_id, fetch_transcript_segments
from src.text_splitter import split_transcript_segments
import requests


def get_vectorstore():
    embeddings = OpenAIEmbeddings()
    return Chroma(
        persist_directory="vectorstore",
        embedding_function=embeddings
    )


def get_video_metadata(video_id):
    """Get video title and thumbnail."""
    
    oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
    
    response = requests.get(oembed_url)
    data = response.json()

    title = data["title"]
    thumbnail = f"https://img.youtube.com/vi/{video_id}/0.jpg"

    return title, thumbnail


def load_video(video_url_or_id: str):

    video_id = extract_video_id(video_url_or_id)

    title, thumbnail = get_video_metadata(video_id)

    segments = fetch_transcript_segments(video_id)

    chunks = split_transcript_segments(segments, video_id)

    texts = [chunk["text"] for chunk in chunks]

    metadatas = []

    for chunk in chunks:
        meta = chunk["metadata"]

        meta["video_title"] = title
        meta["thumbnail"] = thumbnail

        metadatas.append(meta)

    vectorstore = get_vectorstore()

    vectorstore.add_texts(
        texts=texts,
        metadatas=metadatas
    )

    print(f"Video loaded successfully: {title}")
    print(f"Chunks added: {len(chunks)}")


def list_videos():
    vectorstore = get_vectorstore()

    data = vectorstore.get(include=["metadatas"])
    metadatas = data.get("metadatas", [])

    seen = {}

    for meta in metadatas:
        if not meta:
            continue

        video_id = meta.get("video_id")
        title = meta.get("video_title")

        if video_id not in seen:
            seen[video_id] = title

    print("Loaded videos:")

    for video_id, title in seen.items():
        print(f"- {title} ({video_id})")


def remove_video(video_url_or_id: str):
    video_id = extract_video_id(video_url_or_id)
    vectorstore = get_vectorstore()

    data = vectorstore.get(include=["metadatas"])
    ids = data.get("ids", [])
    metadatas = data.get("metadatas", [])

    ids_to_delete = [
        doc_id
        for doc_id, meta in zip(ids, metadatas)
        if meta and meta.get("video_id") == video_id
    ]

    if not ids_to_delete:
        print(f"No chunks found for video: {video_id}")
        return

    vectorstore.delete(ids=ids_to_delete)

    print(f"Removed video: {video_id}")
    print(f"Chunks deleted: {len(ids_to_delete)}")


def clear_all_videos():
    vectorstore = get_vectorstore()

    data = vectorstore.get()
    ids = data.get("ids", [])

    if not ids:
        print("Vector database already empty.")
        return

    vectorstore.delete(ids=ids)

    print(f"Cleared vector database. Removed {len(ids)} chunks.")


if __name__ == "__main__":
    list_videos()