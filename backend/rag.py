import argparse
import io
import os
import tempfile
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Literal, Protocol, cast

import anthropic
import psycopg2
import pypdf
from fastembed import TextEmbedding
from google import genai
from google.genai import types as genai_types
from groq import Groq
from groq.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from pgvector.psycopg2 import register_vector
from psycopg2 import sql as pgsql
from psycopg2.extras import execute_values

PROVIDERS: dict[str, str] = {
    "groq": "Groq (Llama 3.3 70B)",
    "claude": "Claude (claude-sonnet-4-6)",
    "gemini": "Gemini (gemini-2.5-flash)",
}

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384
CHUNK_SIZE = 400
CHUNK_OVERLAP = 40

SYSTEM_PROMPT = """You are a helpful document Q&A assistant for enterprise knowledge bases.
Answer questions using only the provided context. If the context is insufficient, say so explicitly.
Keep answers concise. Always reference which document(s) support your answer."""


# ---------------------------------------------------------------------------
# LLM provider abstraction
# ---------------------------------------------------------------------------


class LLMProvider(Protocol):
    def generate_stream(
        self,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        usage_out: list[dict] | None,
    ) -> Generator[str, None, None]: ...


class GroqProvider:
    MODEL = "llama-3.3-70b-versatile"

    def __init__(self) -> None:
        self._client = Groq(api_key=os.environ["GROQ_API_KEY"])

    def generate_stream(
        self,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        usage_out: list[dict] | None,
    ) -> Generator[str, None, None]:
        sys_msg: ChatCompletionSystemMessageParam = {"role": "system", "content": system}
        user_msgs: list[ChatCompletionUserMessageParam] = [
            {"role": "user", "content": m["content"]} for m in messages
        ]
        typed_messages: list[ChatCompletionMessageParam] = [sys_msg, *user_msgs]
        stream = self._client.chat.completions.create(
            model=self.MODEL,
            max_tokens=max_tokens,
            messages=typed_messages,
            stream=True,
        )
        input_tokens = output_tokens = 0
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
            if chunk.usage:
                input_tokens = chunk.usage.prompt_tokens
                output_tokens = chunk.usage.completion_tokens
        if usage_out is not None:
            usage_out.append({"input_tokens": input_tokens, "output_tokens": output_tokens})


class ClaudeProvider:
    MODEL = "claude-sonnet-4-6"

    def __init__(self) -> None:
        self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def generate_stream(
        self,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        usage_out: list[dict] | None,
    ) -> Generator[str, None, None]:
        typed_messages: list[anthropic.types.MessageParam] = [
            {"role": cast(Literal["user", "assistant"], m["role"]), "content": m["content"]}
            for m in messages
        ]
        with self._client.messages.stream(
            model=self.MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=typed_messages,
        ) as stream:
            yield from stream.text_stream
            if usage_out is not None:
                final = stream.get_final_message()
                usage_out.append(
                    {
                        "input_tokens": final.usage.input_tokens,
                        "output_tokens": final.usage.output_tokens,
                    }
                )


class GeminiProvider:
    MODEL = "gemini-2.5-flash"

    def __init__(self) -> None:
        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if project:
            # Vertex AI モード: Application Default Credentials で認証
            self._client = genai.Client(
                vertexai=True,
                project=project,
                location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
            )
        else:
            # AI Studio モード: GEMINI_API_KEY で認証
            self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def generate_stream(
        self,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        usage_out: list[dict] | None,
    ) -> Generator[str, None, None]:
        # Gemini のロール名は "user"/"model"（"assistant" ではない）
        contents = [
            genai_types.Content(
                role="user" if m["role"] == "user" else "model",
                parts=[genai_types.Part.from_text(text=m["content"])],
            )
            for m in messages
        ]
        config = genai_types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
        )
        last_chunk = None
        for chunk in self._client.models.generate_content_stream(
            model=self.MODEL,
            contents=contents,
            config=config,
        ):
            if chunk.text:
                yield chunk.text
            last_chunk = chunk
        # 最終チャンクに全体の usage_metadata が含まれる
        if usage_out is not None and last_chunk and last_chunk.usage_metadata:
            usage_out.append(
                {
                    "input_tokens": last_chunk.usage_metadata.prompt_token_count or 0,
                    "output_tokens": last_chunk.usage_metadata.candidates_token_count or 0,
                }
            )


def make_llm_provider(provider: str | None = None) -> LLMProvider:
    name = (provider or os.environ.get("LLM_PROVIDER", "groq")).lower()
    if name == "claude":
        return ClaudeProvider()
    if name == "groq":
        return GroqProvider()
    if name == "gemini":
        return GeminiProvider()
    raise ValueError(f"Unknown LLM_PROVIDER={name!r}. Valid: {list(PROVIDERS)}")


# ---------------------------------------------------------------------------
# RAG pipeline
# ---------------------------------------------------------------------------


class RAGPipeline:
    def __init__(self, provider: str | None = None) -> None:
        self.embedder = TextEmbedding(EMBEDDING_MODEL)
        self.llm: LLMProvider = make_llm_provider(provider)
        self.conn = self._connect_db()
        self._init_schema()

    def _connect_db(self) -> psycopg2.extensions.connection:
        import time

        url = os.environ["DATABASE_URL"]
        parsed = urllib.parse.urlparse(url)
        db_name = parsed.path.lstrip("/")

        last_exc: Exception | None = None
        for attempt in range(5):
            try:
                conn = psycopg2.connect(url)
                conn.autocommit = False
                return conn
            except psycopg2.OperationalError as exc:
                last_exc = exc
                # On the first attempt, if the database doesn't exist, create it.
                # pgcode 3D000 = invalid_catalog_name (database does not exist).
                if attempt == 0 and getattr(exc, "pgcode", None) == "3D000":
                    try:
                        fallback_url = urllib.parse.urlunparse(parsed._replace(path="/postgres"))
                        init_conn = psycopg2.connect(fallback_url)
                        init_conn.autocommit = True
                        with init_conn.cursor() as cur:
                            cur.execute(
                                pgsql.SQL("CREATE DATABASE {}").format(pgsql.Identifier(db_name))
                            )
                        init_conn.close()
                        continue  # Retry immediately, skip sleep
                    except Exception:
                        pass
                time.sleep(2**attempt)
        raise last_exc  # type: ignore[misc]

    def _init_schema(self) -> None:
        # Step 1: enable the extension, then register the vector type
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self.conn.commit()
        register_vector(self.conn)

        # Step 2: create table and index now that the vector type is known
        with self.conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS documents (
                    id         SERIAL PRIMARY KEY,
                    filename   TEXT        NOT NULL,
                    content    TEXT        NOT NULL,
                    embedding  vector({EMBEDDING_DIM}) NOT NULL,
                    chunk_idx  INTEGER     NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            # HNSW: no minimum row requirement, better recall than IVFFlat for small datasets
            cur.execute("""
                CREATE INDEX IF NOT EXISTS documents_hnsw_idx
                ON documents USING hnsw (embedding vector_cosine_ops)
            """)
        self.conn.commit()

    # -------------------------------------------------------------------------
    # Ingestion
    # -------------------------------------------------------------------------

    def _ocr_pdf(self, file_bytes: bytes) -> str:
        """pypdf でテキスト抽出できなかった画像 PDF を ndlocr-lite で OCR 処理する。"""
        import ocr as ndlocr
        import pypdfium2 as pdfium

        base_dir = Path(ndlocr.__file__).resolve().parent
        args_template = argparse.Namespace(
            sourcedir=None,
            viz=False,
            device="cpu",
            det_weights=str(base_dir / "model" / "deim-s-1024x1024.onnx"),
            det_classes=str(base_dir / "config" / "ndl.yaml"),
            det_score_threshold=0.2,
            det_conf_threshold=0.25,
            det_iou_threshold=0.2,
            simple_mode=False,
            rec_weights30=str(
                base_dir / "model" / "parseq-ndl-24x256-30-tiny-189epoch-tegaki3-r8data-202604.onnx"
            ),
            rec_weights50=str(
                base_dir / "model" / "parseq-ndl-24x384-50-tiny-300epoch-tegaki3-r8data-202604.onnx"
            ),
            rec_weights=str(
                base_dir
                / "model"
                / "parseq-ndl-24x768-100-tiny-153epoch-tegaki3-r8data-202604.onnx"
            ),
            rec_classes=str(base_dir / "config" / "NDLmoji.yaml"),
            enable_tcy=False,
            json_only=False,
        )

        pdf = pdfium.PdfDocument(file_bytes)
        texts: list[str] = []

        with tempfile.TemporaryDirectory() as tmpdir:
            img_dir = Path(tmpdir) / "imgs"
            out_dir = Path(tmpdir) / "out"
            img_dir.mkdir()
            out_dir.mkdir()

            for i, page in enumerate(pdf):
                bitmap = page.render(scale=2.0)
                img_path = img_dir / f"page_{i:04d}.png"
                bitmap.to_pil().save(str(img_path))

                page_out = out_dir / f"page_{i:04d}"
                page_out.mkdir()

                page_args = argparse.Namespace(
                    **vars(args_template),
                    sourceimg=str(img_path),
                    output=str(page_out),
                )
                ndlocr.process(page_args)

                txt_path = page_out / f"page_{i:04d}.txt"
                if txt_path.exists():
                    texts.append(txt_path.read_text(encoding="utf-8"))

        return "\n".join(texts)

    def extract_text(self, file_bytes: bytes, filename: str) -> str:
        if filename.lower().endswith(".pdf"):
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            if len(text.strip()) < 50:
                text = self._ocr_pdf(file_bytes)
            return text
        return file_bytes.decode("utf-8", errors="replace")

    def _chunk(self, text: str) -> list[str]:
        words = text.split()
        step = CHUNK_SIZE - CHUNK_OVERLAP
        return [
            " ".join(words[i : i + CHUNK_SIZE])
            for i in range(0, len(words), step)
            if words[i : i + CHUNK_SIZE]
        ]

    def ingest(self, filename: str, file_bytes: bytes) -> int:
        text = self.extract_text(file_bytes, filename)
        chunks = self._chunk(text)
        if not chunks:
            return 0
        embeddings = [emb.tolist() for emb in self.embedder.embed(chunks)]

        with self.conn.cursor() as cur:
            execute_values(
                cur,
                "INSERT INTO documents (filename, content, embedding, chunk_idx) VALUES %s",
                [
                    (filename, chunk, emb, idx)
                    for idx, (chunk, emb) in enumerate(zip(chunks, embeddings))
                ],
                template="(%s, %s, %s::vector, %s)",
            )
        self.conn.commit()
        return len(chunks)

    def list_documents(self) -> list[tuple[str, int]]:
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT filename, COUNT(*) AS chunks
                FROM documents
                GROUP BY filename
                ORDER BY filename
            """)
            return cur.fetchall()

    def delete_document(self, filename: str) -> None:
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM documents WHERE filename = %s", (filename,))
        self.conn.commit()

    # -------------------------------------------------------------------------
    # Retrieval
    # -------------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[str, str, float]]:
        (query_embedding,) = self.embedder.embed([query])
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT filename, content,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM documents
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query_embedding.tolist(), query_embedding.tolist(), top_k),
            )
            return cur.fetchall()

    # -------------------------------------------------------------------------
    # Generation (streaming)
    # -------------------------------------------------------------------------

    def generate_stream(
        self,
        query: str,
        chunks: list[tuple[str, str, float]],
        usage_out: list[dict] | None = None,
    ) -> Generator[str, None, None]:
        context = "\n\n---\n\n".join(
            f"[{filename} | relevance: {sim:.0%}]\n{content}" for filename, content, sim in chunks
        )
        messages = [
            {
                "role": "user",
                "content": f"Context documents:\n\n{context}\n\n---\n\nQuestion: {query}",
            }
        ]
        yield from self.llm.generate_stream(
            system=SYSTEM_PROMPT,
            messages=messages,
            max_tokens=1024,
            usage_out=usage_out,
        )


# ---------------------------------------------------------------------------
# Agent pipeline (Claude tool_use)
# ---------------------------------------------------------------------------

AGENT_SYSTEM_PROMPT = """You are an intelligent document Q&A assistant with tool access.

Use the available tools to answer user questions accurately:
- Call search_documents with a focused query to retrieve relevant document chunks
- Call list_documents to see what documents are indexed
- You may search multiple times with different queries to gather comprehensive information

After gathering sufficient context, provide a clear, accurate answer citing source documents.
If the retrieved context is insufficient, say so explicitly."""

AGENT_TOOLS: list[dict] = [
    {
        "name": "search_documents",
        "description": "インデックス済みドキュメントを意味的に検索し、関連するチャンクを返す。",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "検索クエリ"},
                "top_k": {"type": "integer", "description": "返す最大チャンク数（デフォルト: 5）"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "list_documents",
        "description": "インデックス済みドキュメントの一覧（ファイル名・チャンク数）を返す。",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
]


@dataclass
class AgentEvent:
    type: Literal["tool_call", "tool_result", "text_delta", "done"]
    data: dict = field(default_factory=dict)


class AgentPipeline:
    """Multi-step reasoning agent using Claude tool_use. Uses RAGPipeline for DB access."""

    _MAX_STEPS = 8

    def __init__(self, rag: RAGPipeline) -> None:
        self._rag = rag
        self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def _execute_tool(self, name: str, inputs: dict) -> dict:
        if name == "search_documents":
            query = inputs.get("query", "")
            top_k = int(inputs.get("top_k", 5))
            chunks = self._rag.retrieve(query, top_k=top_k)
            text = (
                "\n\n".join(
                    f"[{fname} | relevance: {sim:.0%}]\n{content}" for fname, content, sim in chunks
                )
                or "No relevant documents found."
            )
            return {"text": text, "chunks": chunks}
        if name == "list_documents":
            docs = self._rag.list_documents()
            text = (
                "\n".join(f"- {fname} ({n} chunks)" for fname, n in docs) or "No documents indexed."
            )
            return {"text": text, "chunks": []}
        return {"text": f"Unknown tool: {name}", "chunks": []}

    def run_stream(
        self,
        query: str,
        usage_out: list[dict] | None = None,
    ) -> Generator[AgentEvent, None, None]:
        messages: list[dict] = [{"role": "user", "content": query}]
        total_input = total_output = 0

        for _ in range(self._MAX_STEPS):
            response = self._client.messages.create(
                model=ClaudeProvider.MODEL,
                max_tokens=4096,
                system=AGENT_SYSTEM_PROMPT,
                tools=AGENT_TOOLS,  # type: ignore[arg-type]
                messages=messages,  # type: ignore[arg-type]
            )
            total_input += response.usage.input_tokens
            total_output += response.usage.output_tokens
            messages.append({"role": "assistant", "content": response.content})  # type: ignore[arg-type]

            if response.stop_reason == "end_turn":
                text = "".join(
                    block.text  # type: ignore[union-attr]
                    for block in response.content
                    if block.type == "text"
                )
                # Yield word-by-word for a streaming feel
                words = text.split(" ")
                for i, word in enumerate(words):
                    yield AgentEvent(
                        type="text_delta",
                        data={"delta": word if i == len(words) - 1 else word + " "},
                    )
                break

            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    yield AgentEvent(
                        type="tool_call",
                        data={"tool": block.name, "input": block.input, "id": block.id},
                    )
                    result = self._execute_tool(block.name, block.input)  # type: ignore[arg-type]
                    yield AgentEvent(
                        type="tool_result",
                        data={
                            "tool": block.name,
                            "id": block.id,
                            "result": result["text"],
                            "chunks": result["chunks"],
                        },
                    )
                    tool_results.append(
                        {"type": "tool_result", "tool_use_id": block.id, "content": result["text"]}
                    )
                messages.append({"role": "user", "content": tool_results})  # type: ignore[arg-type]

        if usage_out is not None:
            usage_out.append({"input_tokens": total_input, "output_tokens": total_output})
        yield AgentEvent(type="done", data={})
