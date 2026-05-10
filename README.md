# DocQA – Enterprise RAG Demo

社内ドキュメント Q&A システムのデモ。
PDF / テキストをアップロードして自然言語で質問できる。
回答には参照元チャンクが表示され、👍/👎 フィードバックは Langfuse に記録される。

---

## アーキテクチャ

```
ユーザー → Streamlit UI
              ↓
          RAGPipeline
          ├── Embed query (sentence-transformers / all-MiniLM-L6-v2)
          ├── pgvector HNSW 近似近傍探索 (PostgreSQL)
          └── Claude API 生成 (claude-sonnet-4-6, streaming)
              ↓
          Langfuse トレース (レイテンシ・トークン数・ユーザー評価)
```

### 技術選定の理由

| レイヤー | 選択 | 理由 |
|----------|------|------|
| LLM | `claude-sonnet-4-6` | 高品質・低コスト・ストリーミング対応 |
| ベクトルDB | pgvector (HNSW) | 既存PostgreSQLに追加するだけで運用コスト最小 |
| Embedding | all-MiniLM-L6-v2 | OSS・API コスト不要・384次元で RAG に十分 |
| 監視 | Langfuse | LangCore スタックに記載。トレース・スコア・ダッシュボード |
| UI | Streamlit | 1〜2日で動くデモが作れる Python ネイティブ UI |

詳細な決定理由は [docs/adr-001-vector-db.md](docs/adr-001-vector-db.md) を参照。

---

## クイックスタート

### 1. 前提

- Docker / Docker Compose
- Anthropic API キー
- （任意）Langfuse アカウント（[cloud.langfuse.com](https://cloud.langfuse.com) 無料枠あり）

### 2. 環境変数の設定

```bash
cp .env.example .env
# .env を編集して ANTHROPIC_API_KEY を設定
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
pip install -r requirements.txt
DATABASE_URL=postgresql://localhost/ragdemo streamlit run app.py
```

---

## 使い方

1. **左サイドバー**から PDF または `.txt` ファイルをアップロード
2. チャット欄に質問を入力して Enter
3. 回答の下の **「📎 Sources used」** で参照元チャンクを確認
4. **👍/👎** で回答品質を評価 → Langfuse に記録される

---

## Langfuse ダッシュボードで確認できること

- リクエストごとのトレース（検索・生成・合計レイテンシ）
- Claude のトークン使用量（コスト見積もり）
- ユーザーフィードバックスコアの集計
- セッション単位でのコンテキスト追跡

---

## プロジェクト構成

```
.
├── docker-compose.yml       # PostgreSQL + pgvector + Streamlit
├── .env.example
├── backend/
│   ├── app.py               # Streamlit UI + セッション管理
│   ├── rag.py               # RAG パイプライン（ingest / retrieve / generate）
│   ├── langfuse_client.py   # Langfuse ラッパー（無効化可能）
│   ├── requirements.txt
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
- **streaming**: `client.messages.stream` + `st.write_stream` で UX を損なわない
- **`@st.cache_resource`**: RAGPipeline・LangfuseClient を 1 回だけ初期化してメモリを節約

### スケールアップ時の移行パス

| 課題 | 対策 |
|------|------|
| ベクトル数 1000万超 | pgvector → Qdrant（インタフェースを `retrieve()` の裏側だけ変える） |
| Embedding の質向上 | `all-MiniLM-L6-v2` → `text-embedding-3-large` に差し替え |
| マルチユーザー認証 | Streamlit + FastAPI バックエンド分離 |

---

## ライセンス

MIT
