# DocQA – Enterprise RAG Demo

社内ドキュメント Q&A システムのデモ。
PDF / テキストをアップロードして自然言語で質問できる。
回答には参照元チャンクが表示され、👍/👎 フィードバックは Langfuse に記録される。

---

## アーキテクチャ

```
ユーザー → Streamlit UI（サイドバーでプロバイダー切り替え）
              ↓
          RAGPipeline
          ├── Embed query (fastembed / BAAI/bge-small-en-v1.5)
          ├── pgvector HNSW 近似近傍探索 (PostgreSQL)
          └── LLM 生成 (Groq / Claude, streaming)
              ↓
          Langfuse トレース (レイテンシ・トークン数・ユーザー評価)
```

### 技術選定の理由

| レイヤー | 選択 | 理由 |
|----------|------|------|
| LLM | Groq (Llama 3.3 70B) / Claude (claude-sonnet-4-6) | UI で動的に切り替え可能。Groq は無料枠あり |
| ベクトルDB | pgvector (HNSW) | 既存PostgreSQLに追加するだけで運用コスト最小 |
| Embedding | fastembed / BAAI/bge-small-en-v1.5 | ONNX ベースで PyTorch 不要・軽量・384次元 |
| 監視 | Langfuse v4 | トレース・スコア・ダッシュボード |
| UI | Streamlit | 1〜2日で動くデモが作れる Python ネイティブ UI |

詳細な決定理由は [docs/adr-001-vector-db.md](docs/adr-001-vector-db.md) を参照。

---

## クイックスタート

### 1. 前提

- Docker / Docker Compose
- Groq API キー（無料: [console.groq.com](https://console.groq.com)）または Anthropic API キー
- （任意）Langfuse アカウント（[cloud.langfuse.com](https://cloud.langfuse.com) 無料枠あり）

### 2. 環境変数の設定

```bash
cp .env.example .env
# .env を編集して使用するプロバイダーの API キーを設定:
#   Groq 使用時  → GROQ_API_KEY
#   Claude 使用時 → ANTHROPIC_API_KEY
# LLM_PROVIDER=groq または claude（デフォルト: groq）
# 任意: LANGFUSE_* を設定するとモニタリングが有効になる
```

### 3. 起動

```bash
docker compose up --build
```

ブラウザで http://localhost:8501 を開く。

### ローカル実行（Docker なし）

```bash
# PostgreSQL + pgvector が必要
# homebrew: brew install postgresql
# pgvector: https://github.com/pgvector/pgvector

cd backend
uv sync
DATABASE_URL=postgresql://localhost/ragdemo uv run streamlit run app.py
```

---

## 使い方

1. **左サイドバー**の **LLM プロバイダー** ドロップダウンで Groq / Claude を選択
2. サイドバーから PDF または `.txt` ファイルをアップロード
3. チャット欄に質問を入力して Enter
4. 回答の下の **「📎 Sources used」** で参照元チャンクを確認
5. **👍/👎** で回答品質を評価 → Langfuse に記録される

---

## Langfuse ダッシュボードで確認できること

- リクエストごとのトレース（検索・生成・合計レイテンシ）
- Claude のトークン使用量（コスト見積もり）
- ユーザーフィードバックスコアの集計
- セッション単位でのコンテキスト追跡

---

## テスト

統合テストはコンテナ内で実行します（`db` サービスが起動している必要があります）。

```bash
docker compose run --rm app uv run pytest tests/ -v
```

CI（GitHub Actions）は PR ごとに自動で `docker compose up` → テスト → teardown を実行します。

---

## プロジェクト構成

```
.
├── .github/
│   └── workflows/
│       └── ci.yml           # CI: docker compose lint + 統合テスト
├── docker-compose.yml       # PostgreSQL + pgvector + Streamlit（app_net で接続）
├── .env.example
├── backend/
│   ├── app.py               # Streamlit UI + セッション管理
│   ├── rag.py               # RAG パイプライン（ingest / retrieve / generate）
│   ├── langfuse_client.py   # Langfuse ラッパー（無効化可能）
│   ├── tests/
│   │   └── test_integration.py  # DB 接続・スキーマ初期化の統合テスト
│   └── Dockerfile
└── docs/
    └── adr-001-vector-db.md # Architecture Decision Record
```

---

## 設計上のポイント（テックリード視点）

### RAG パイプライン

- **チャンク戦略**: 400 語・40 語オーバーラップ。文脈を失わずトークン効率を保つ
- **HNSW インデックス**: IVFFlat と異なりデータ挿入後すぐに使える。デモで詰まらない
- **コサイン類似度**: 0〜1 のスコアでフロントエンドに表示しやすい

### 品質・運用

- **Langfuse をオプショナルに**: 環境変数がなければ `_NullTrace` にフォールバックし、
  開発環境でもエラーなく動く
- **streaming**: 各プロバイダーの streaming API + `st.empty()` で UX を損なわない
- **`@st.cache_resource(provider)`**: プロバイダーごとに RAGPipeline をキャッシュ。切り替えても再初期化コスト最小

### スケールアップ時の移行パス

| 課題 | 対策 |
|------|------|
| ベクトル数 1000万超 | pgvector → Qdrant（インタフェースを `retrieve()` の裏側だけ変える） |
| Embedding の質向上 | `BAAI/bge-small-en-v1.5` → `BAAI/bge-large-en-v1.5` に差し替え |
| LLM の追加 | `rag.py` に `XxxProvider` クラスを追加 → `PROVIDERS` と `make_llm_provider()` に分岐を追加 |
| マルチユーザー認証 | Streamlit + FastAPI バックエンド分離 |

---

## ライセンス

MIT
