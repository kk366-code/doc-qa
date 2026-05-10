"""Integration tests — require a running PostgreSQL + pgvector instance.

Run inside the app container:
    docker compose run --rm app uv run pytest tests/ -v
"""

import os

import psycopg2
import pytest
from pgvector.psycopg2 import register_vector


@pytest.fixture(scope="module")
def db_conn():
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    register_vector(conn)
    yield conn
    conn.close()


def test_db_connection(db_conn):
    with db_conn.cursor() as cur:
        cur.execute("SELECT 1")
        assert cur.fetchone() == (1,)


def test_pgvector_extension(db_conn):
    with db_conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector'")
        assert cur.fetchone() is not None
    db_conn.rollback()


def test_schema_init():
    """RAGPipeline.__init__ が例外なく完了することを確認する。"""
    from rag import RAGPipeline

    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.conn = psycopg2.connect(os.environ["DATABASE_URL"])
    register_vector(pipeline.conn)
    pipeline._init_schema()
    pipeline.conn.close()
