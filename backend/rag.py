import io
import os
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
from psycopg2.extras import execute_values

PROVIDERS: dict[str, str] = {
    "groq": "Groq (Llama 3.3 70B)",
    "claude": "Claude (claude-sonnet-4-6)",
    "gemini": "Gemini (gemini-2.0-flash)",
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
    MODEL = "gemini-2.0-flash"

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

        last_exc: Exception | None = None
        for attempt in range(5):
            try:
                conn = psycopg2.connect(os.environ["DATABASE_URL"])
                conn.autocommit = False
                return conn
            except psycopg2.OperationalError as exc:
                last_exc = exc
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

    def extract_text(self, file_bytes: bytes, filename: str) -> str:
        if filename.lower().endswith(".pdf"):
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
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
