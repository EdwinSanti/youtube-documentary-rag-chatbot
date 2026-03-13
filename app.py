import io
import re
from pathlib import Path

import streamlit as st
from openai import OpenAI
from PIL import Image, UnidentifiedImageError

from src.agent_chatbot import agent, retrieve_sources
from src.video_tools import (
    get_vectorstore,
    load_video,
    remove_video,
    clear_all_videos,
)

# -----------------------------
# Paths + logo loader
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
LOGO_PATH = BASE_DIR / "assets" / "doculens_logo.png"

logo_image = None
logo_error = None

try:
    if LOGO_PATH.exists():
        logo_image = Image.open(LOGO_PATH).convert("RGBA")

        # Remove white background
        datas = logo_image.getdata()
        new_data = []
        for item in datas:
            if item[0] > 240 and item[1] > 240 and item[2] > 240:
                new_data.append((255, 255, 255, 0))
            else:
                new_data.append(item)
        logo_image.putdata(new_data)

        # Crop center area to remove extra whitespace
        logo_image = logo_image.crop((600, 600, 1400, 1400))
    else:
        logo_error = f"Logo file not found: {LOGO_PATH}"
except UnidentifiedImageError:
    logo_error = f"Pillow could not identify the image file: {LOGO_PATH}"
except Exception as e:
    logo_error = f"Error loading logo: {e}"

# -----------------------------
# Config
# -----------------------------
st.set_page_config(
    page_title="DocuLens AI",
    page_icon=logo_image if logo_image is not None else "🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Header
# -----------------------------
if logo_image is not None:
    import base64
    from io import BytesIO

    buffered = BytesIO()
    logo_image.save(buffered, format="PNG")
    logo_base64 = base64.b64encode(buffered.getvalue()).decode()

    st.markdown(
        f"""
        <div style="text-align:center; margin-top:10px;">
            <img src="data:image/png;base64,{logo_base64}" width="95"/>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div style="text-align:center;">
        <div class="header-title">DocuLens AI</div>
        <div class="header-subtitle">
            Ask grounded questions about your loaded documentaries.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if logo_error:
    st.warning(logo_error)

# -----------------------------
# Global CSS
# -----------------------------
st.markdown(
    """
    <style>
    .main .block-container {
        padding-bottom: 170px;
        padding-top: 2rem;
    }

    textarea, input {
        font-size: 16px !important;
    }

    div[data-testid="stTextInput"] input {
        height: 50px;
        border-radius: 14px;
    }

    div[data-testid="stButton"] > button {
        min-height: 50px;
        border-radius: 14px;
        white-space: normal !important;
        line-height: 1.35 !important;
        padding-top: 0.55rem !important;
        padding-bottom: 0.55rem !important;
    }

    [data-testid="stAudioInput"] {
        width: 100%;
        border-radius: 14px;
        overflow: hidden;
    }

    .fixed-chat-dock {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: #0e1117;
        border-top: 1px solid rgba(250, 250, 250, 0.12);
        padding: 0.85rem 1rem 0.95rem 1rem;
        z-index: 9999;
    }

    .fixed-chat-inner {
        max-width: 980px;
        margin: 0 auto;
    }

    .composer-note {
        max-width: 980px;
        margin: 0 auto 0.35rem auto;
        color: #c9d1d9;
        font-size: 0.9rem;
    }

    .header-shell {
        max-width: 980px;
        margin: 0 auto 2.2rem auto;
    }

    .header-row {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 0.85rem;
        text-align: center;
    }

    .header-logo-wrap {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 0.5rem;
    }

    .header-text-wrap {
        text-align: center;
    }

    .header-title {
        margin: 0;
        font-size: 2.85rem;
        font-weight: 800;
        line-height: 1.05;
        color: #f5f7fb;
    }

    .header-subtitle {
        margin-top: 0.45rem;
        color: #aeb8c5;
        font-size: 1rem;
    }

    .welcome-box {
        text-align: center;
        margin: 0 auto;
        margin-top: 40px;
        max-width: 760px;
        opacity: 0.95;
    }

    .welcome-box h3 {
        margin-bottom: 0.8rem;
        font-size: 2.2rem;
        line-height: 1.15;
        color: #f5f7fb;
    }

    .welcome-box p {
        margin-bottom: 0.45rem;
        color: #c9d1d9;
        font-size: 1rem;
        line-height: 1.6;
    }

    .suggestion-title {
        text-align: center;
        margin-top: 1.7rem;
        margin-bottom: 1rem;
        color: #c9d1d9;
        font-size: 0.98rem;
        font-weight: 600;
    }

    .hero-container {
        max-width: 720px;
        margin: 0 auto;
    }

    .confidence-pill {
        margin-top: 0.85rem;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 0.85rem;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.03);
        font-size: 0.92rem;
        font-weight: 600;
    }

    .confidence-label {
        color: #c9d1d9;
        font-weight: 600;
    }

    .confidence-high {
        color: #57d38c;
    }

    .confidence-medium {
        color: #f2c94c;
    }

    .confidence-low {
        color: #ff5c7a;
    }

    @media (max-width: 1100px) {
        .fixed-chat-inner,
        .composer-note {
            max-width: 100%;
        }

        .main .block-container {
            padding-bottom: 190px;
        }
    }

    @media (max-width: 768px) {
        .header-title {
            font-size: 2.2rem;
        }

        .header-subtitle {
            font-size: 0.94rem;
        }

        .welcome-box h3 {
            font-size: 1.8rem;
        }

        .welcome-box p {
            font-size: 0.95rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

client = OpenAI()

# -----------------------------
# Session state
# -----------------------------
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []

if "draft_prompt" not in st.session_state:
    st.session_state.draft_prompt = ""

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

if "last_voice_bytes" not in st.session_state:
    st.session_state.last_voice_bytes = None

if "voice_status" not in st.session_state:
    st.session_state.voice_status = ""

if "video_input_counter" not in st.session_state:
    st.session_state.video_input_counter = 0

# -----------------------------
# Helpers
# -----------------------------
def get_video_library():
    """Read unique loaded videos from Chroma metadata."""
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
            videos[video_id] = {
                "video_id": video_id,
                "title": meta.get("video_title", video_id),
                "thumbnail": meta.get(
                    "thumbnail",
                    f"https://img.youtube.com/vi/{video_id}/0.jpg",
                ),
            }

    return list(videos.values())


def parse_source_blocks(sources_raw: str):
    if not sources_raw:
        return []
    return [block.strip() for block in sources_raw.split("\n\n---\n\n") if block.strip()]


def format_confidence_badge(text: str) -> str:
    """
    Replace a plain 'Confidence: X' line with a styled pill.
    """
    pattern = r"Confidence:\s*(High|Medium|Low)"
    match = re.search(pattern, text)

    if not match:
        return text

    level = match.group(1)

    css_class_map = {
        "High": "confidence-high",
        "Medium": "confidence-medium",
        "Low": "confidence-low",
    }

    icon_map = {
        "High": "●",
        "Medium": "●",
        "Low": "●",
    }

    badge_html = f"""
<div class="confidence-pill">
    <span class="confidence-label">Answer Confidence</span>
    <span class="{css_class_map[level]}">{icon_map[level]} {level}</span>
</div>
"""

    text_without_confidence = re.sub(pattern, "", text).strip()
    return text_without_confidence + badge_html


def ask_agent_with_memory(user_prompt: str):
    """Call the agent with conversation history, then fetch supporting sources."""
    st.session_state.agent_messages.append(
        {"role": "user", "content": user_prompt}
    )

    result = agent.invoke({"messages": st.session_state.agent_messages})
    final_message = result["messages"][-1]
    answer = final_message.content if hasattr(final_message, "content") else str(final_message)

    st.session_state.agent_messages.append(
        {"role": "assistant", "content": answer}
    )

    sources_raw = retrieve_sources.invoke(user_prompt)
    source_blocks = parse_source_blocks(sources_raw)

    return answer, source_blocks


def render_sources(source_blocks: list[str]):
    if source_blocks:
        with st.expander(f"Sources ({min(len(source_blocks), 4)})", expanded=False):
            tabs = st.tabs([f"Source {i+1}" for i in range(min(len(source_blocks), 4))])
            for i, tab in enumerate(tabs):
                with tab:
                    st.markdown(source_blocks[i])


def handle_user_prompt(user_prompt: str):
    """Single path for submitted prompts."""
    st.session_state.chat_messages.append(
        {"role": "user", "content": user_prompt}
    )

    try:
        with st.spinner("Analyzing documentaries..."):
            answer, source_blocks = ask_agent_with_memory(user_prompt)

        st.session_state.chat_messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": source_blocks,
            }
        )

        st.session_state.pending_prompt = ""
        st.session_state.voice_status = ""
        st.session_state.last_voice_bytes = None

    except Exception as e:
        st.session_state.chat_messages.append(
            {
                "role": "assistant",
                "content": f"Error: {e}",
                "sources": [],
            }
        )


def transcribe_audio(audio_file):
    """Transcribe microphone audio to text using OpenAI speech-to-text."""
    audio_bytes = audio_file.getvalue()
    buffer = io.BytesIO(audio_bytes)
    buffer.name = "voice_question.wav"

    transcript = client.audio.transcriptions.create(
        model="gpt-4o-mini-transcribe",
        file=buffer,
    )

    return transcript.text.strip()


def shorten_title_for_tab(title: str, max_len: int = 18) -> str:
    """Make tab labels shorter and cleaner."""
    clean = title.strip()
    if len(clean) <= max_len:
        return clean
    return clean[:max_len].rstrip() + "..."


def build_general_suggested_prompts() -> list[str]:
    """General prompts that work across almost any documentary set."""
    return [
        "Summarize all loaded documentaries",
        "What are the main topics discussed across the documentaries?",
        "Compare the loaded documentaries based only on the retrieved context",
        "What major challenges or themes appear across the documentaries?",
    ]


def build_documentary_tab_prompts(title: str) -> list[str]:
    """Fixed templates that adapt to the documentary title."""
    return [
        f"What is '{title}' mainly about?",
        f"What are the key facts from '{title}'?",
        f"What important challenges or themes are discussed in '{title}'?",
        f"What evidence or examples stand out in '{title}'?",
    ]


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("🎥 Loaded Documentaries")

    current_video_input_key = f"video_input_{st.session_state.video_input_counter}"

    video_value = st.text_input(
        "Add YouTube video",
        key=current_video_input_key,
        placeholder="Paste a YouTube link or video ID",
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Load video", use_container_width=True):
            clean_value = video_value.strip()

            if not clean_value:
                st.warning("Paste a YouTube link or video ID first.")
            else:
                try:
                    load_video(clean_value)
                    st.session_state.video_input_counter += 1
                    st.success("Video loaded successfully.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to load video: {e}")

    with col2:
        if st.button("Clear all", use_container_width=True):
            try:
                clear_all_videos()
                st.success("Library cleared.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to clear library: {e}")

    st.divider()

    library = get_video_library()

    if not library:
        st.info("No documentaries loaded yet.")
    else:
        for item in library:
            with st.container():
                st.image(item["thumbnail"], use_container_width=True)
                st.write(f"**{item['title']}**")
                st.caption(item["video_id"])

                if st.button(f"Remove {item['video_id']}", key=f"remove_{item['video_id']}"):
                    try:
                        remove_video(item["video_id"])
                        st.success(f"Removed {item['title']}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to remove video: {e}")

                st.divider()

# Refresh library for suggestions outside sidebar
library = get_video_library()

# -----------------------------
# Empty welcome + suggested prompts
# -----------------------------
if not st.session_state.chat_messages:
    st.markdown(
        """
        <div class="welcome-box hero-container">
            <h3>Explore your documentaries with AI 🎬</h3>
            <p>Ask questions across multiple documentaries, with grounded answers and sources.</p>
            <p>You can also record your question, review the transcription, and send it manually.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if library:
        st.markdown(
            '<div class="suggestion-title">💡 Suggested Questions</div>',
            unsafe_allow_html=True,
        )

        general_prompts = build_general_suggested_prompts()
        general_cols = st.columns(2, gap="small")

        for i, prompt in enumerate(general_prompts):
            with general_cols[i % 2]:
                if st.button(prompt, key=f"general_suggestion_{i}", use_container_width=True):
                    st.session_state.pending_prompt = prompt
                    st.rerun()

        with st.expander("🎯 Suggestions by Documentary", expanded=False):
            tab_labels = [shorten_title_for_tab(item["title"]) for item in library]
            doc_tabs = st.tabs(tab_labels)

            for idx, tab in enumerate(doc_tabs):
                item = library[idx]
                doc_title = item["title"]
                tab_prompts = build_documentary_tab_prompts(doc_title)

                with tab:
                    tab_cols = st.columns(2, gap="small")

                    for j, prompt in enumerate(tab_prompts):
                        with tab_cols[j % 2]:
                            if st.button(
                                prompt,
                                key=f"doc_prompt_{idx}_{j}",
                                use_container_width=True,
                            ):
                                st.session_state.pending_prompt = prompt
                                st.rerun()

# -----------------------------
# Chat history
# -----------------------------
for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            formatted_content = format_confidence_badge(msg["content"])
            st.markdown(formatted_content, unsafe_allow_html=True)
        else:
            st.markdown(msg["content"])

        if msg["role"] == "assistant" and msg.get("sources"):
            render_sources(msg["sources"])

# -----------------------------
# Apply pending prompt BEFORE rendering input widget
# -----------------------------
if st.session_state.pending_prompt is not None:
    st.session_state.draft_prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

# -----------------------------
# Fixed composer
# -----------------------------
composer_placeholder = st.empty()

with composer_placeholder.container():
    st.markdown('<div class="fixed-chat-dock"><div class="fixed-chat-inner">', unsafe_allow_html=True)

    if st.session_state.voice_status:
        st.markdown(
            f'<div class="composer-note">{st.session_state.voice_status}</div>',
            unsafe_allow_html=True,
        )

    input_col, mic_col, send_col = st.columns([8.5, 1.7, 1.2], gap="small")

    with input_col:
        st.text_input(
            "Message",
            key="draft_prompt",
            placeholder="Type your question, or record and edit it before sending...",
            label_visibility="collapsed",
        )

    with mic_col:
        st.audio_input(
            "Record",
            key="voice_input",
            label_visibility="collapsed",
        )

    with send_col:
        send_clicked = st.button(
            "Send",
            use_container_width=True,
            type="primary",
        )

    st.markdown("</div></div>", unsafe_allow_html=True)

# -----------------------------
# Voice transcription
# -----------------------------
voice_audio_obj = st.session_state.get("voice_input", None)

if voice_audio_obj is not None:
    current_bytes = voice_audio_obj.getvalue()

    if st.session_state.last_voice_bytes != current_bytes:
        st.session_state.last_voice_bytes = current_bytes

        try:
            with st.spinner("Transcribing..."):
                voice_text = transcribe_audio(voice_audio_obj)

            if voice_text:
                st.session_state.pending_prompt = voice_text
                st.session_state.voice_status = "Voice transcribed. Review or edit it, then press Send."
                st.rerun()
            else:
                st.session_state.voice_status = "No speech detected."
                st.rerun()

        except Exception as e:
            st.session_state.voice_status = f"Voice transcription failed: {e}"
            st.rerun()

# -----------------------------
# Send flow
# -----------------------------
if send_clicked:
    prompt = st.session_state.draft_prompt.strip()
    if prompt:
        handle_user_prompt(prompt)
        st.rerun()
    else:
        st.warning("Type or record a question first.")

# -----------------------------
# Auto-scroll after reply
# -----------------------------
st.markdown(
    """
    <script>
        window.scrollTo(0, document.body.scrollHeight);
    </script>
    """,
    unsafe_allow_html=True,
)