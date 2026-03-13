from dotenv import load_dotenv
load_dotenv()

import os
import json
from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from text_splitter import split_transcript_segments


def load_transcript_json(file_path):
    """Load transcript JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_vector_db():

    embeddings = OpenAIEmbeddings()

    transcript_folder = Path("transcripts")

    all_chunks = []

    for file in transcript_folder.glob("*.json"):

        video_id = file.stem

        segments = load_transcript_json(file)

        chunks = split_transcript_segments(segments, video_id)

        all_chunks.extend(chunks)

    texts = [chunk["text"] for chunk in all_chunks]
    metadatas = [chunk["metadata"] for chunk in all_chunks]

    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory="vectorstore"
    )

    print(f"\nVector database created with {len(texts)} chunks.")


if __name__ == "__main__":
    create_vector_db()