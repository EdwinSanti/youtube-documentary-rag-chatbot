from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

from src.text_splitter import load_transcript, split_transcript_segments


def get_vectorstore():
    embeddings = OpenAIEmbeddings()
    return Chroma(
        persist_directory="vectorstore",
        embedding_function=embeddings
    )


def format_timestamp(seconds: float) -> str:
    total_seconds = int(seconds)

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    if hours > 0:
        return f"{hours:02}:{minutes:02}:{secs:02}"

    return f"{minutes:02}:{secs:02}"


def build_youtube_link(video_id: str, timestamp: float) -> str:
    return f"https://www.youtube.com/watch?v={video_id}&t={int(timestamp)}s"


def get_source_docs(question: str, k: int = 5):
    vectorstore = get_vectorstore()

    results = vectorstore.similarity_search_with_score(question, k=k)

    # FILTER weak matches (this is the improvement)
    filtered = [(doc, score) for doc, score in results if score < 0.55]

    docs = [doc for doc, score in filtered]
    scores = [score for doc, score in filtered]

    return docs, scores


@tool
def rag_answer(question: str) -> str:
    """Answer a question using only the loaded documentary transcripts."""

    lower_q = question.lower()

    is_comparison = any(word in lower_q for word in [
        "compare", "difference", "differences", "similar", "similarities", "both"
    ])

    retrieval_k = 10 if is_comparison else 6
    docs, scores = get_source_docs(question, k=retrieval_k)

    avg_score = sum(scores) / len(scores) if scores else 1.0

    # Chroma returns distance-like scores: lower is better
    if avg_score < 0.3:
        confidence = "High"
    elif avg_score < 0.5:
        confidence = "Medium"
    else:
        confidence = "Low"

    if not docs:
        return (
            "I don’t have enough evidence from the loaded documentaries to answer that reliably."
            f"\n\nConfidence: {confidence}"
        )

    context = "\n\n".join(
        f"[Source {i+1}]\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = f"""
You are a documentary question-answering assistant.

Your job is to answer using ONLY the retrieved documentary context below.

Rules:
1. Use only the retrieved context.
2. Do not use outside knowledge.
3. You may summarize, synthesize, and compare information across the retrieved context.
4. Do not invent facts that are not supported by the context.
5. If the context contains partial information, provide the best grounded answer possible from what is available.
6. If there is truly not enough information, say:
   "I don’t have enough evidence from the loaded documentaries to answer that reliably."
7. For "key facts" questions, extract the main grounded points from the retrieved context instead of refusing unless the context is clearly insufficient.
8. For summary questions, summarize the documentary or documentaries using the retrieved context, even if the summary is partial.
9. For comparison questions, compare the themes, animals, behaviors, habitats, events, or subjects described in the retrieved context for each documentary.
10. If a comparison is possible only at a high level, provide a high-level grounded comparison instead of refusing outright.
11. Be explicit when some details are missing or only partially supported by the retrieved context.
12. Be concise, specific, and trustworthy.
13. Do not claim a documentary says something unless that idea is supported by the retrieved context.

Response format:
- Start with a direct answer in 1 to 2 sentences.
- Then give 2 to 5 short bullet points if helpful.
- If the answer is based on partial context, add a brief limitation note at the end.

Retrieved documentary context:
{context}

User question:
{question}

Answer:
"""

    response = llm.invoke(prompt)
    answer = response.content.strip()

    answer += f"\n\nConfidence: {confidence}"

    return answer


@tool
def retrieve_sources(question: str) -> str:
    """Return the top retrieved documentary sources with titles, timestamps, previews, and YouTube links."""

    docs, _ = get_source_docs(question, k=4)

    if not docs:
        return "No sources found."

    parts = []

    for i, doc in enumerate(docs, start=1):
        video_id = doc.metadata.get("video_id", "unknown")
        video_title = doc.metadata.get("video_title", video_id)
        timestamp = doc.metadata.get("timestamp", 0)
        timestamp_str = format_timestamp(timestamp)
        youtube_link = build_youtube_link(video_id, timestamp)

        preview = doc.page_content[:400].strip()

        source_block = (
            f"Source {i}\n"
            f"Title: {video_title}\n"
            f"Video ID: {video_id}\n"
            f"Timestamp: {timestamp_str}\n"
            f"Link: {youtube_link}\n"
            f"Preview: {preview}"
        )

        parts.append(source_block)

    return "\n\n---\n\n".join(parts)


@tool
def video_info(_: str = "") -> str:
    """Return basic information about the currently loaded documentaries."""

    vectorstore = get_vectorstore()
    data = vectorstore.get(include=["metadatas"])
    metadatas = data.get("metadatas", [])

    videos = {}
    for meta in metadatas:
        if not meta:
            continue

        video_id = meta.get("video_id")
        if not video_id:
            continue

        if video_id not in videos:
            videos[video_id] = meta.get("video_title", video_id)

    if not videos:
        return "No videos are currently loaded."

    lines = ["Loaded documentaries:"]
    for video_id, title in videos.items():
        lines.append(f"- {title} ({video_id})")

    return "\n".join(lines)


agent = create_agent(
    model="gpt-4o-mini",
    tools=[rag_answer, retrieve_sources, video_info],
    system_prompt=(
    "You are a trustworthy documentary assistant.\n"
    "\n"
    "You help users answer questions about the currently loaded documentaries.\n"
    "\n"
    "Rules:\n"
    "1. For documentary content questions, use rag_answer.\n"
    "2. For questions about sources, timestamps, evidence, or links, use retrieve_sources.\n"
    "3. For questions about what documentaries are loaded, use video_info.\n"
    "4. Do not answer documentary-content questions from general knowledge when rag_answer should be used.\n"
    "5. If the loaded documentary evidence is insufficient, say so clearly instead of guessing.\n"
    "6. Do not invent scenes or unsupported facts.\n"
    "7. Summaries, key facts, and comparisons are allowed when they are grounded in retrieved documentary context.\n"
    "8. If a question is ambiguous, ask a short clarifying question.\n"
    "9. Be concise, specific, and trustworthy.\n"
    "10. If rag_answer returns a final line in the format 'Confidence: High', 'Confidence: Medium', or 'Confidence: Low', preserve that line verbatim at the end of your final answer.\n"
)
)


if __name__ == "__main__":
    print("\nDocumentary Agent Chatbot")
    print("Type 'exit' to quit.\n")

    messages = []

    while True:
        user_input = input("Ask a question: ")

        if user_input.lower() == "exit":
            break

        messages.append({"role": "user", "content": user_input})

        result = agent.invoke({"messages": messages})
        messages = result["messages"]

        final_message = messages[-1]

        print("\nAnswer:\n")
        print(final_message.content)

        print("\nSources:\n")
        print(retrieve_sources.invoke(user_input))

        print("\n" + "=" * 60 + "\n")