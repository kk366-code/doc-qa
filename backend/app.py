"""Enterprise Document Q&A – RAG demo with Langfuse v4 observability."""

import os
import uuid

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from google.genai import errors as genai_errors  # noqa: E402
from langfuse_client import LangfuseClient  # noqa: E402
from rag import PROVIDERS, AgentPipeline, RAGPipeline  # noqa: E402

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="DocQA – RAG Demo",
    page_icon="📄",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session-level singletons
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Initialising RAG pipeline…")
def get_rag(provider: str) -> RAGPipeline:
    return RAGPipeline(provider=provider)


@st.cache_resource(show_spinner=False)
def get_agent_pipeline(provider: str) -> AgentPipeline:
    return AgentPipeline(get_rag(provider))


@st.cache_resource(show_spinner=False)
def get_lf() -> LangfuseClient:
    return LangfuseClient()


lf = get_lf()

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []  # {role, content, chunks, trace_id}
if "feedback_sent" not in st.session_state:
    st.session_state.feedback_sent = set()
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = set()  # (name, size) already processed
if "llm_provider" not in st.session_state:
    st.session_state.llm_provider = os.environ.get("LLM_PROVIDER", "groq")
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "rag"

# ---------------------------------------------------------------------------
# Sidebar – document management
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("📄 DocQA")
    st.caption("Enterprise document Q&A powered by pgvector")

    st.divider()
    st.subheader("モード")
    app_mode = st.radio(
        "モード",
        options=["rag", "agent"],
        format_func=lambda m: (
            "RAGモード（マルチプロバイダー）" if m == "rag" else "エージェントモード（Claude専用）"
        ),
        key="app_mode",  # Streamlit が session_state.app_mode を自動同期
        label_visibility="collapsed",
    )

    st.divider()
    st.subheader("LLM プロバイダー")
    if app_mode == "agent":
        st.info("エージェントモードは Claude (tool_use) 固定です")
        selected_provider = "claude"
    else:
        provider_keys = list(PROVIDERS.keys())
        selected_provider = st.selectbox(
            "LLM プロバイダー",
            options=provider_keys,
            format_func=lambda k: PROVIDERS[k],
            index=provider_keys.index(st.session_state.llm_provider),
            label_visibility="collapsed",
        )
        st.session_state.llm_provider = selected_provider

    rag = get_rag(selected_provider)

    st.divider()
    st.subheader("Upload documents")

    uploaded = st.file_uploader(
        "PDF or plain text",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded:
        for f in uploaded:
            key = (f.name, f.size)
            if key not in st.session_state.ingested_files:
                with st.spinner(f"{f.name} を処理中（スキャンPDFの場合 OCR が実行されます）…"):
                    try:
                        n = rag.ingest(f.name, f.read())
                    except genai_errors.ServerError:
                        st.error(
                            f"{f.name}: Gemini API が混雑しています。"
                            "しばらく待ってから再度お試しください。"
                        )
                        continue
                if n == 0:
                    st.error(f"{f.name}: テキストを抽出できませんでした。")
                else:
                    st.session_state.ingested_files.add(key)
                    st.success(f"{f.name}: {n} chunks indexed")

    st.divider()
    st.subheader("Indexed documents")

    docs = rag.list_documents()
    if docs:
        for name, n_chunks in docs:
            col1, col2 = st.columns([4, 1])
            col1.write(f"**{name}** ({n_chunks} chunks)")
            if col2.button("🗑", key=f"del_{name}", help="Remove"):
                rag.delete_document(name)
                st.rerun()
    else:
        st.info("No documents yet. Upload a PDF or .txt file above.")

    st.divider()
    if lf.enabled:
        st.success("Langfuse monitoring active")
    else:
        st.warning("Langfuse not configured – set LANGFUSE_* env vars to enable")

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.session_state.feedback_sent = set()
        st.rerun()

# ---------------------------------------------------------------------------
# Main – chat interface
# ---------------------------------------------------------------------------

st.header("Ask a question about your documents")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant":
            if msg.get("chunks"):
                with st.expander("📎 Sources used", expanded=False):
                    for filename, content, sim in msg["chunks"]:
                        st.markdown(f"**{filename}** – relevance `{sim:.0%}`\n\n> {content[:300]}…")

            trace_id = msg.get("trace_id", "")
            if trace_id and trace_id not in st.session_state.feedback_sent:
                col_up, col_dn, _ = st.columns([1, 1, 8])
                if col_up.button("👍", key=f"up_{trace_id}"):
                    lf.send_feedback(trace_id, 1.0, "thumbs up")
                    st.session_state.feedback_sent.add(trace_id)
                    st.toast("Thanks!", icon="👍")
                if col_dn.button("👎", key=f"dn_{trace_id}"):
                    lf.send_feedback(trace_id, 0.0, "thumbs down")
                    st.session_state.feedback_sent.add(trace_id)
                    st.toast("Thanks for the feedback.", icon="📝")

# ---------------------------------------------------------------------------
# Handle new user input
# ---------------------------------------------------------------------------

if query := st.chat_input("Type your question here…"):
    if not rag.list_documents():
        st.warning("Please upload at least one document first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    full_response = ""
    chunks: list = []
    trace_id = "no-trace"
    usage_out: list[dict] = []

    with lf.trace(query=query, session_id=st.session_state.session_id) as trace:
        if app_mode == "agent":
            # ----------------------------------------------------------------
            # Agent mode: Claude tool_use loop
            # ----------------------------------------------------------------
            agent = get_agent_pipeline(selected_provider)
            with st.chat_message("assistant"):
                steps_container = st.container()
                text_placeholder = st.empty()
                current_expander = None
                step_idx = 0

                for event in agent.run_stream(query, usage_out=usage_out):
                    if event.type == "tool_call":
                        step_idx += 1
                        tool = event.data["tool"]
                        inp = event.data["input"]
                        label = (
                            f"🔍 検索: {inp.get('query', '')}"
                            if tool == "search_documents"
                            else "📋 ドキュメント一覧を取得"
                        )
                        with steps_container:
                            current_expander = st.expander(label, expanded=False)

                    elif event.type == "tool_result":
                        result_chunks: list = event.data.get("chunks", [])
                        chunks.extend(result_chunks)
                        trace.log_agent_step(
                            step=step_idx,
                            tool_name=event.data["tool"],
                            tool_input={},
                            result_chunks=len(result_chunks),
                        )
                        if current_expander is not None:
                            with current_expander:
                                if result_chunks:
                                    for fname, content, sim in result_chunks[:3]:
                                        st.markdown(
                                            f"**{fname}** – `{sim:.0%}`\n\n> {content[:200]}…"
                                        )
                                else:
                                    st.text(event.data.get("result", "")[:300])

                    elif event.type == "text_delta":
                        full_response += event.data["delta"]
                        text_placeholder.markdown(full_response + "▌")

                text_placeholder.markdown(full_response)

                if chunks:
                    with st.expander("📎 参照したソース", expanded=False):
                        for fname, content, sim in chunks:
                            st.markdown(
                                f"**{fname}** – relevance `{sim:.0%}`\n\n> {content[:300]}…"
                            )

                col_up, col_dn, _ = st.columns([1, 1, 8])
                if col_up.button("👍", key="up_new"):
                    lf.send_feedback(trace.id, 1.0, "thumbs up")
                    st.session_state.feedback_sent.add(trace.id)
                    st.toast("Thanks!", icon="👍")
                if col_dn.button("👎", key="dn_new"):
                    lf.send_feedback(trace.id, 0.0, "thumbs down")
                    st.session_state.feedback_sent.add(trace.id)
                    st.toast("Thanks for the feedback.", icon="📝")

            usage = usage_out[0] if usage_out else None
            trace.log_generation(full_response, usage)

        else:
            # ----------------------------------------------------------------
            # RAG mode: single-shot retrieve → generate
            # ----------------------------------------------------------------
            with st.spinner("Searching documents…"):
                chunks = rag.retrieve(query, top_k=5)
                trace.log_retrieval(query, chunks)

            with st.chat_message("assistant"):
                placeholder = st.empty()
                for token in rag.generate_stream(query, chunks, usage_out=usage_out):
                    full_response += token
                    placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)

                if chunks:
                    with st.expander("📎 Sources used", expanded=True):
                        for filename, content, sim in chunks:
                            st.markdown(
                                f"**{filename}** – relevance `{sim:.0%}`\n\n> {content[:300]}…"
                            )

                col_up, col_dn, _ = st.columns([1, 1, 8])
                if col_up.button("👍", key="up_new"):
                    lf.send_feedback(trace.id, 1.0, "thumbs up")
                    st.session_state.feedback_sent.add(trace.id)
                    st.toast("Thanks!", icon="👍")
                if col_dn.button("👎", key="dn_new"):
                    lf.send_feedback(trace.id, 0.0, "thumbs down")
                    st.session_state.feedback_sent.add(trace.id)
                    st.toast("Thanks for the feedback.", icon="📝")

            usage = usage_out[0] if usage_out else None
            trace.log_generation(full_response, usage)

        trace_id = trace.id

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_response,
            "chunks": chunks,
            "trace_id": trace_id,
        }
    )
    lf.flush()
