# DocQA RAG Demo — 開発ガイド

## パッケージ管理
- **uv** を使用する（`pip install` は書かない）
- 依存追加: `uv add <package>`
- 依存同期: `uv sync`
- スクリプト実行: `uv run streamlit run backend/app.py`
- ロックファイル `uv.lock` はコミットすること

## Python バージョン
- ターゲット: **Python 3.13**
- `pyproject.toml` に `requires-python = ">=3.13"` を明記

## フォーマット・リント
- **Ruff** を使用（black / flake8 / isort は使わない）
- フォーマット: `uv run ruff format .`
- リント: `uv run ruff check .`
- 設定は `pyproject.toml` の `[tool.ruff]` セクションで管理

## モダン Python の書き方
- `Optional[X]` ではなく `X | None` を使う
- `List[T]`, `Dict[K, V]` ではなく `list[T]`, `dict[K, V]` を使う
- 前方参照には `TYPE_CHECKING` ブロックを使う

## コード追加・変更時のルール
- 新しいファイルを作成するときは設計の意図（何をするファイルか・なぜこの構造にしたか）を説明する
- 既存コードへの変更は「なぜ変えたか」を1〜2行で添える
- 1行以下の軽微な修正（typo修正・import追加など）は説明不要
- **コード変更後は必ず `README.md` を最新の状態に更新すること**（アーキテクチャ図・技術選定表・使い方・スケールアップ路表など関連箇所を確認する）

## コマンド実行のルール
- **コマンド実行の許可を仰ぐ際は、必ずそのコマンドが「何をするものか」という意図と意味を日本語で説明すること。**
- `uv run` や `docker` コマンドなど、システムに変更を加える可能性のある操作については、副作用（ファイルの生成、DBの初期化、ポートの専有など）があれば併記すること。
- 複数の手順がある場合は、ステップごとに意味を説明すること。

## 外部 SDK の実装ルール
- **初めて使うライブラリは必ず WebFetch で公式ドキュメントを確認してからコードを書く**
- 記憶の API を使わない（特に監視・インフラ系はメジャーバージョンで API が別物になる）
- 主要ライブラリの公式ドキュメント:
  - Langfuse: https://langfuse.com/docs/sdk/python
  - Anthropic: https://docs.anthropic.com/
  - pgvector: https://github.com/pgvector/pgvector-python
  - google-genai (Gemini): https://googleapis.github.io/python-genai/

## Docker ルール
- **Embedding には `fastembed` を使う**（ONNX Runtime ベース、PyTorch 不要、軽量）
  - sentence-transformers は CUDA 付き PyTorch（1GB+）を引き込むため使わない
- `docker-compose.yml` には `platform: linux/arm64` を明示（M2 Mac 向け）
- Dockerfile では uv を公式イメージからコピーして使う:
  ```dockerfile
  COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
  ```
- `uv sync --frozen` でロックファイルに基づいた再現性あるインストールを行う

## 技術スタック
- LLM: マルチプロバイダー（`LLM_PROVIDER` 環境変数で切り替え、デフォルト: `groq`）
  - `groq`: `llama-3.3-70b-versatile` via Groq SDK（無料枠あり）
  - `claude`: `claude-sonnet-4-6` via Anthropic Python SDK
  - `gemini`: `gemini-2.0-flash` via google-genai SDK（`GEMINI_API_KEY` 必要）
  - 新プロバイダー追加: `rag.py` に `XxxProvider` クラスを追加 → `make_llm_provider()` に分岐を追加
- Embedding: `fastembed` (BAAI/bge-small-en-v1.5, 384 次元)
- Vector DB: pgvector (PostgreSQL) — HNSW インデックス
- 監視: Langfuse v4（最新 API を使うこと）
- UI: Streamlit

## 📝 Git & PR Convention

- **タイトル（1行目）**: 英語で記述する。例: `feat: add agent UI`, `fix: handle empty URL response`
- **本文（3行目以降）**: 日本語で記述する。何を・なぜ変えたかを説明する。
- **PRの作成**: 実装完了後は必ずプルリクエストを作成する。既存のPRがマージ済みの場合は、新しいPRを作成する（マージ済みPRには追加しない）。PR作成前に `git fetch origin main` で最新の main を取得し、`git log origin/main..HEAD` でそのPRに含まれるコミットを確認すること。
- **PR作成後のセッション引き継ぎ**: PRを作成したら、次のセッションへの引き継ぎテキストを必ず出力する。以下の形式で記述すること：

  ```
  ## 次のセッションへの引き継ぎ

  ### 作成したPR
  - PR URL: <URL>
  - 概要: <何を実装したか1〜2文>

  ### 現在のブランチ状態
  - ブランチ名: <branch>
  - 対象ファイル: <主な変更ファイル>

  ### 次にやること（あれば）
  - <残タスクや懸念点>
  ```

## 開発コマンド

```bash
# 初回セットアップ
cp .env.example .env        # APIキーを記入
uv sync                     # 依存インストール
uv run pre-commit install   # gitleaks pre-commit hook を登録

# ローカル実行（PostgreSQL + pgvector が別途必要）
uv run streamlit run backend/app.py

# Docker 起動（推奨）
docker compose up --build

# フォーマット・リント
uv run ruff format .
uv run ruff check --fix .
```

## ディレクトリ構成

```
.
├── CLAUDE.md
├── docker-compose.yml
├── pyproject.toml           # uv で管理
├── uv.lock                  # コミットすること
├── .env.example
├── backend/
│   ├── app.py               # Streamlit UI
│   ├── rag.py               # RAG パイプライン
│   ├── langfuse_client.py   # Langfuse v4 監視
│   └── Dockerfile
└── docs/
    └── adr-001-vector-db.md
```
