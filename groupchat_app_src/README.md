# Group Chat Web App + LLM Bot

FastAPI + MySQL + vanilla HTML/JS group chat with LLM bot (OpenAI-compatible server such as self-hosted Llama 3 via llama.cpp).

## Quick Start (Dev)

```bash
# 1) MySQL
# create DB & user, or run sql/schema.sql

# 2) Backend
cd backend
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# copy env and edit values
cp ../.env.example ../.env

# 3) Run app
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000

LLM endpoint defaults to http://localhost:8001/v1 (llama.cpp server). Set `LLM_API_BASE`, `LLM_MODEL`, `LLM_API_KEY` if needed in `.env`.

# Part 4 — HeartWise: Heart Health RAG Chatbot

HeartWise is a domain-specific, Retrieval-Augmented Generation (RAG) chatbot designed to provide educational information about cardiovascular health. It combines a curated medical knowledge base, semantic search, and a locally hosted Llama 3.1 language model to generate context-grounded responses with source references.

The application includes a responsive web interface and an Android WebView application, both connected to the same FastAPI backend.

## Features

- **Curated knowledge base:** Heart-health information collected from CDC and NHLBI resources, covering cardiovascular risk factors, prevention, symptoms, and related topics.
- **Semantic retrieval:** Uses `BAAI/bge-small-en-v1.5` sentence-transformer embeddings (384 dimensions) and FAISS similarity search to retrieve relevant passages.
- **RAG question answering:** Retrieves the top four matching passages and supplies them as context to a locally hosted Llama 3.1 8B Instruct model through an OpenAI-compatible API.
- **Source references:** Displays supporting sources alongside generated responses.
- **User authentication:** FastAPI login/signup endpoints integrated with the existing MySQL user database.
- **Web and Android support:** Responsive HTML/CSS/JavaScript frontend, with a Kotlin-based Android WebView client.
- **Safety messaging:** Includes educational-use disclaimers and basic emergency-keyword detection. The application is not intended for diagnosis or clinical decision-making.

## Project Structure

The following paths are relative to `groupchat_app_src/`:

```text
heartwise_app/
├── backend/
│   └── app.py                 # HeartWise FastAPI backend
├── frontend/
│   ├── index.html             # Chat interface
│   ├── styles.css             # Responsive styling
│   └── app.js                 # Authentication and chat requests
└── android/                   # Android Studio WebView project

embedding/
└── heartwise/
    ├── ingest_heartwise.py    # Knowledge-base ingestion
    ├── embed_heartwise.py     # Embedding and FAISS indexing
    ├── chat_heartwise.py      # RAG retrieval and generation
    ├── test_retrieval.py      # Retrieval testing
    └── faiss_index/           # Generated vector index and metadata
```

## Quick Start (Development)

Run these commands from the `groupchat_app_src/` directory unless otherwise indicated.

### 1. Start MySQL

Use the existing GroupChat MySQL database and user schema. Configure database credentials and application settings in the root `.env` file, following the GroupChat setup instructions.

### 2. Install Python dependencies

```bash
cd embedding
python3 -m venv .venv
source .venv/bin/activate
pip install -r ../backend/requirements.txt
pip install faiss-cpu sentence-transformers
cd ..
```

Additional dependencies may be needed depending on the existing environment.

### 3. Start the local LLM server

Start an OpenAI-compatible `llama.cpp` server using the locally downloaded Llama 3.1 8B Instruct GGUF model:

```bash
./llama.cpp/build/bin/llama-server \
  -m ./models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  --host 127.0.0.1 \
  --port 8001 \
  -c 4096
```

Run this from the directory containing the `llama.cpp/` and `models/` folders, adjusting paths as necessary.

### 4. Prepare the knowledge base

If the FAISS index and metadata are already included in the repository, this step can be skipped.

Otherwise, run the ingestion and embedding scripts from the project root:

```bash
python embedding/heartwise/ingest_heartwise.py
python embedding/heartwise/embed_heartwise.py
```

These scripts prepare the curated heart-health content and generate the vector index used for semantic retrieval.

### 5. Start the HeartWise backend

From `groupchat_app_src/`:

```bash
./embedding/.venv/bin/python -m uvicorn heartwise_app.backend.app:app \
  --host 127.0.0.1 --port 8002
```

Open:

- Web application: `http://127.0.0.1:8002/`
- Backend health check: `http://127.0.0.1:8002/api/health`

Sign up or log in to access the chatbot.

### 6. Run the Android application

1. Open `heartwise_app/android/` in Android Studio.
2. Sync Gradle and select an Android emulator.
3. Keep MySQL, the Llama server, and the HeartWise FastAPI backend running.
4. Build and run the `app` configuration.

The Android WebView loads `http://10.0.2.2:8002/`, where `10.0.2.2` is the Android emulator's special address for accessing the host computer.


## Implementation Overview

The RAG pipeline follows these steps:

1. **Data preparation:** Collect and preprocess cardiovascular-health information from CDC and NHLBI resources, then split it into text chunks.
2. **Embedding and indexing:** Generate normalized sentence-transformer embeddings and store them in a FAISS vector index.
3. **Retrieval:** Embed the user's question and retrieve the four most semantically relevant chunks.
4. **Generation:** Pass the retrieved context and user question to the locally hosted Llama 3.1 model.
5. **Response:** Display the generated answer and supporting source references in the web or Android chat interface.

The prototype knowledge base was built from 11 source webpages, producing 78 indexed text chunks.

## Technology Stack

| Component | Technology |
|---|---|
| Backend | Python, FastAPI |
| Authentication/database | MySQL, JWT |
| Embeddings | Sentence Transformers, BAAI/bge-small-en-v1.5 |
| Vector search | FAISS |
| Language model | Llama 3.1 8B Instruct, llama.cpp |
| Web frontend | HTML, CSS, JavaScript |
| Android frontend | Kotlin, Android WebView |
| Development environment | Android Studio, Android Emulator |

