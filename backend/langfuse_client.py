"""Langfuse v4 (OTel-based) monitoring wrapper.

Langfuse v4 uses OpenTelemetry under the hood. The primary API is:
  - langfuse.start_as_current_observation(as_type="span"|"generation", name=...)
  - langfuse.get_current_trace_id()
  - langfuse.propagate_attributes(session_id=...)
  - langfuse.create_score(trace_id=..., name=..., value=..., data_type="NUMERIC")

Ref: https://langfuse.com/docs/sdk/python
"""

import os
from contextlib import contextmanager
from typing import Generator


def _langfuse_enabled() -> bool:
    return bool(os.environ.get("LANGFUSE_SECRET_KEY"))


class LangfuseClient:
    """Thin, optional wrapper around Langfuse v4.

    When LANGFUSE_SECRET_KEY is unset, all methods are no-ops so the app
    runs fine without monitoring credentials.
    """

    def __init__(self) -> None:
        self._lf = None
        if _langfuse_enabled():
            from langfuse import get_client
            self._lf = get_client()

    @property
    def enabled(self) -> bool:
        return self._lf is not None

    # -------------------------------------------------------------------------
    # Trace context manager
    # -------------------------------------------------------------------------

    @contextmanager
    def trace(
        self, query: str, session_id: str | None = None
    ) -> Generator["_TraceHandle", None, None]:
        """Open a root span for one RAG query. Yields a _TraceHandle."""
        if not self._lf:
            yield _NullTrace()
            return

        with self._lf.start_as_current_observation(
            as_type="span",
            name="rag-query",
            input={"query": query},
            # session_id has no dedicated param in v4 OTel API; pass via metadata
            metadata={"session_id": session_id} if session_id else None,
        ):
            trace_id = self._lf.get_current_trace_id() or "no-trace"
            yield _TraceHandle(lf=self._lf, trace_id=trace_id)

    # -------------------------------------------------------------------------
    # Scoring (called after the trace context has closed)
    # -------------------------------------------------------------------------

    def send_feedback(self, trace_id: str, value: float, comment: str = "") -> None:
        """value: 1.0 = thumbs up, 0.0 = thumbs down"""
        if not self._lf or trace_id in ("no-trace", ""):
            return
        self._lf.create_score(
            trace_id=trace_id,
            name="user-feedback",
            value=value,
            data_type="NUMERIC",
            comment=comment,
        )

    def flush(self) -> None:
        if self._lf:
            self._lf.flush()


class _TraceHandle:
    """Wraps an active trace, providing helpers for child observations."""

    def __init__(self, lf: object, trace_id: str) -> None:
        self._lf = lf
        self.id = trace_id

    def log_retrieval(self, query: str, chunks: list[tuple[str, str, float]]) -> None:
        with self._lf.start_as_current_observation(
            as_type="span",
            name="retrieval",
            input={"query": query},
            output={
                "n_chunks": len(chunks),
                "chunks": [
                    {"filename": f, "similarity": round(s, 4), "preview": c[:120]}
                    for f, c, s in chunks
                ],
            },
        ):
            pass

    def log_generation(
        self,
        response: str,
        usage: dict[str, int] | None = None,
    ) -> None:
        kwargs: dict = {
            "as_type": "generation",
            "name": "llm-generate",
            "model": "llama-3.3-70b-versatile",
            "output": response,
        }
        if usage:
            kwargs["usage_details"] = {
                "input": usage.get("input_tokens", 0),
                "output": usage.get("output_tokens", 0),
            }
        with self._lf.start_as_current_observation(**kwargs):
            pass


class _NullTrace:
    """Returned when Langfuse is disabled; silently absorbs all calls."""

    id = "no-trace"

    def log_retrieval(self, *_: object, **__: object) -> None:
        pass

    def log_generation(self, *_: object, **__: object) -> None:
        pass
