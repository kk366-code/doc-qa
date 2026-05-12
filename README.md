# DocQA – Enterprise RAG Demo

社内ドキュメント Q&A システムのデモ。
PDF / テキストをアップロードして自然言語で質問できる。
スキャン済み日本語 PDF は ndlocr-lite（国立国会図書館製）で自動 OCR 処理される。
回答には参照元チャンクが表示され、👍/👎 フィードバックは Langfuse に記録される。

---

## アーキテクチャ

```
ユーザー → Streamlit UI（モード切替 + プロバイダー切り替え）
              ↓
  ┌─── RAGモード ────────────────────────────────┐
  │    RAGPipeline                              │
  │    ├── Embed query (fastembed)              │
  │    ├── pgvector HNSW 近似近傍探索             │
  │    └── LLM 生成 (Groq / Claude / Gemini)     │
  └─────────────────────────────────────────────┘
  ┌─── エージェントモード ─────────────────────────┐
  │    AgentPipeline (Claude tool_use)          │
  │    ├── Tool: search_documents               │
  │    │     └── pgvector HNSW 検索（複数回）     │
  │    ├── Tool: list_documents                 │
  │    └── Claude が推論・ツール呼び出しを決定       │
  └─────────────────────────────────────────────┘
              ↓
          Langfuse トレース (レイテンシ・トークン数・ユーザー評価)
```

### 技術選定の理由

| レイヤー | 選択 | 理由 |
|----------|------|------|
| LLM | Groq (Llama 3.3 70B) / Claude (claude-sonnet-4-6) / Gemini (gemini-2.5-flash) | UI で動的に切り替え可能。Groq は無料枠あり |
| ベクトルDB | pgvector (HNSW) | 既存PostgreSQLに追加するだけで運用コスト最小 |
| Embedding | fastembed / BAAI/bge-small-en-v1.5 | ONNX ベースで PyTorch 不要・軽量・384次元 |
| OCR | ndlocr-lite（国立国会図書館） | 日本語スキャン PDF に特化・ONNX ベース・GPU 不要 |
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
#   Groq 使用時   → GROQ_API_KEY
#   Claude 使用時 → ANTHROPIC_API_KEY
#   Gemini 使用時 → 下記のどちらかを選択
# LLM_PROVIDER=groq または claude または gemini（デフォルト: groq）
# 任意: LANGFUSE_* を設定するとモニタリングが有効になる
```

#### Gemini の認証方式

**AI Studio（無料枠）**
```bash
# GEMINI_API_KEY のみ設定（GOOGLE_CLOUD_PROJECT は不要）
GEMINI_API_KEY=AIza...
```
> 無料枠のクォータが枯渇すると `429 RESOURCE_EXHAUSTED` が発生します。その場合は Vertex AI に切り替えてください。

**Vertex AI（Google Cloud 課金）**
```bash
# 1. Google Cloud で認証
gcloud auth application-default login

# 2. .env に Google Cloud プロジェクトを設定（GEMINI_API_KEY は不要）
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1   # 省略時は us-central1

# 3. Docker 内で使う場合は docker-compose.yml の ADC マウント行のコメントを外す
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

### RAGモード（デフォルト）

1. **左サイドバー**の **モード** で「RAGモード」を選択
2. **LLM プロバイダー** ドロップダウンで Groq / Claude / Gemini を選択
3. サイドバーから PDF または `.txt` ファイルをアップロード（スキャン済み日本語 PDF は自動で OCR 処理）
4. チャット欄に質問を入力して Enter
5. 回答の下の **「📎 Sources used」** で参照元チャンクを確認
6. **👍/👎** で回答品質を評価 → Langfuse に記録される

### エージェントモード（Claude tool_use）

1. **左サイドバー**の **モード** で「エージェントモード」を選択（`ANTHROPIC_API_KEY` が必要）
2. 質問を入力すると Claude が自律的に `search_documents` / `list_documents` ツールを呼び出す
3. ツール呼び出しの過程が **🔍 検索: ...** expander にリアルタイム表示される
4. 必要に応じて複数回検索した後、最終回答を生成する

---

## Langfuse ダッシュボードで確認できること

- リクエストごとのトレース（検索・生成・合計レイテンシ）
- Claude のトークン使用量（コスト見積もり）
- ユーザーフィードバックスコアの集計
- セッション単位でのコンテキスト追跡

---

## セキュリティ

[gitleaks](https://github.com/gitleaks/gitleaks) による pre-commit hook でシークレットの誤コミットを防止している。

```bash
# hook 登録（初回のみ）
uv run pre-commit install

# 手動スキャン
uv run pre-commit run --all-files
```

`.gitleaks.toml` で `.env.example` 内のダミー値を除外設定済み。

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

### エージェントパイプライン

- **Claude tool_use**: `search_documents` / `list_documents` の 2 ツールを定義。Claude が「何をいつ検索するか」を自律的に判断する
- **多段推論**: 最大 8 ステップのループ。異なるクエリで複数回検索することで、単一検索より高品質な回答を生成
- **Langfuse 連携**: `log_agent_step()` でステップごとのツール呼び出しをトレースに記録
- **UI の可視化**: ツール呼び出しを `st.expander` でリアルタイム表示し、思考過程を透明化

### スケールアップ時の移行パス

| 課題 | 対策 |
|------|------|
| ベクトル数 1000万超 | pgvector → Qdrant（インタフェースを `retrieve()` の裏側だけ変える） |
| Embedding の質向上 | `BAAI/bge-small-en-v1.5` → `BAAI/bge-large-en-v1.5` に差し替え |
| LLM の追加 | `rag.py` に `XxxProvider` クラスを追加 → `PROVIDERS` と `make_llm_provider()` に分岐を追加 |
| マルチユーザー認証 | Streamlit + FastAPI バックエンド分離 |

---

## Google Cloud へのデプロイ

### アーキテクチャマッピング

| ローカル | Google Cloud |
|---------|-------------|
| PostgreSQL (pgvector) | Cloud SQL for PostgreSQL 16 + pgvector 拡張 |
| Streamlit コンテナ | Cloud Run |
| `.env` ファイル | Secret Manager |
| Docker イメージ | Artifact Registry |

コードの変更は不要。`DATABASE_URL` や API キーはすでに環境変数経由のため、Cloud Run の環境変数・Secret Manager に差し替えるだけで動作する。

### 前提

- Google Cloud プロジェクト作成済み
- `gcloud` CLI インストール済み・認証済み (`gcloud auth login`)

### Step 0: 必要な API をまとめて有効化

```bash
export PROJECT_ID=your-project-id
export REGION=asia-northeast1

gcloud config set project $PROJECT_ID

gcloud services enable \
  sqladmin.googleapis.com \
  sql-component.googleapis.com \
  secretmanager.googleapis.com \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com
```

### Step 1: Artifact Registry にイメージをプッシュ

```bash
gcloud artifacts repositories create rag-demo \
  --repository-format=docker \
  --location=$REGION

gcloud builds submit ./backend \
  --tag $REGION-docker.pkg.dev/$PROJECT_ID/rag-demo/app:latest
```

### Step 2: Cloud SQL インスタンスを作成

`--edition=ENTERPRISE` を指定することで `db-f1-micro`（最小・低コスト）が使用できる。

```bash
gcloud sql instances create ragdemo \
  --database-version=POSTGRES_16 \
  --tier=db-f1-micro \
  --edition=ENTERPRISE \
  --region=$REGION

gcloud sql databases create ragdemo --instance=ragdemo

# postgres ユーザーのパスワードを設定
gcloud sql users set-password postgres --instance=ragdemo --password=<STRONG_PW>
```

> pgvector 拡張は Cloud SQL PostgreSQL 16 に同梱済み。アプリ初回起動時に `CREATE EXTENSION IF NOT EXISTS vector` が自動実行される。

### Step 3: Secret Manager にシークレットを登録

```bash
# DATABASE_URL: Cloud SQL Auth Proxy (Unix ソケット) 経由の接続文字列
echo -n "postgresql://postgres:<PW>@/ragdemo?host=/cloudsql/$PROJECT_ID:$REGION:ragdemo" | \
  gcloud secrets create DATABASE_URL --data-file=-

echo -n "gsk_..."    | gcloud secrets create GROQ_API_KEY      --data-file=-
echo -n "sk-ant-..." | gcloud secrets create ANTHROPIC_API_KEY --data-file=-
echo -n "AIza..."    | gcloud secrets create GEMINI_API_KEY    --data-file=-
echo -n "sk-lf-..."  | gcloud secrets create LANGFUSE_SECRET   --data-file=-
echo -n "pk-lf-..."  | gcloud secrets create LANGFUSE_PUBLIC   --data-file=-
```

### Step 4: IAM 権限を付与

Cloud Run のサービスアカウントに Secret Manager と Cloud SQL へのアクセス権を付与する。**この手順を先に実行しないとデプロイが失敗する。**

```bash
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')
SA=$PROJECT_NUMBER-compute@developer.gserviceaccount.com

# Secret Manager アクセス権
for SECRET in DATABASE_URL GROQ_API_KEY ANTHROPIC_API_KEY GEMINI_API_KEY LANGFUSE_SECRET LANGFUSE_PUBLIC; do
  gcloud secrets add-iam-policy-binding $SECRET \
    --member="serviceAccount:$SA" --role=roles/secretmanager.secretAccessor
done

# Cloud SQL アクセス権
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$SA" --role=roles/cloudsql.client
```

### Step 5: Cloud Run にデプロイ

```bash
gcloud run deploy rag-demo \
  --image=$REGION-docker.pkg.dev/$PROJECT_ID/rag-demo/app:latest \
  --region=$REGION \
  --port=8501 \
  --allow-unauthenticated \
  --add-cloudsql-instances=$PROJECT_ID:$REGION:ragdemo \
  --set-secrets="DATABASE_URL=DATABASE_URL:latest,ANTHROPIC_API_KEY=ANTHROPIC_API_KEY:latest,GROQ_API_KEY=GROQ_API_KEY:latest,LANGFUSE_SECRET_KEY=LANGFUSE_SECRET:latest,LANGFUSE_PUBLIC_KEY=LANGFUSE_PUBLIC:latest" \
  --set-env-vars="LLM_PROVIDER=groq,LANGFUSE_BASE_URL=https://jp.cloud.langfuse.com" \
  --memory=1Gi \
  --cpu=1 \
  --min-instances=0 \
  --max-instances=3
```

デプロイ完了後に表示される URL (`https://rag-demo-xxxx.run.app`) にブラウザでアクセスする。

### Vertex AI (Gemini) を使う場合

Cloud Run はサービスアカウント経由で Vertex AI に直接認証できる（ADC ファイル不要）。`--region` を必ず指定すること。

```bash
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$SA" --role=roles/aiplatform.user

gcloud run services update rag-demo \
  --region=$REGION \
  --update-env-vars="GOOGLE_CLOUD_PROJECT=$PROJECT_ID,LLM_PROVIDER=gemini"
```

### コスト概算（最小構成）

| サービス | 月額目安 |
|---------|---------|
| Cloud SQL db-f1-micro | ~$10 |
| Cloud Run（軽量） | ~$0–5（無料枠あり） |
| Artifact Registry | ~$0–1 |
| Secret Manager | ほぼ無料 |

---

## ライセンス

MIT
