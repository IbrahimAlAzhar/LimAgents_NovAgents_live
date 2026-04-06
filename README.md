# LimAgents with NovAgents – Multi-Agent Limitation & Novelty Analysis

Upload a research paper and analyze its limitations using 13+ AI agents across two pipelines, then merge the results.

## Three Pipelines

| Pipeline | What it does | Agents |
|----------|-------------|--------|
| **LimAgents** | General limitations from the paper itself | 6 specialists + Leader + Master |
| **NovAgents** | Novelty limitations vs similar published papers | 6 specialists + Leader + Master |
| **Merge** | Combines both outputs, removes duplicates | 1 merge agent |

## Setup (Two Steps)

### Step 1: Upload your 140K papers to Pinecone (one time)

1. Create a free account at https://www.pinecone.io
2. Get your API key from the Pinecone dashboard
3. On your local machine (NOT on Render):

```bash
pip install openai pinecone-client pandas tqdm
```

4. Edit `upload_to_pinecone.py` — set your API keys and CSV path
5. Run it:

```bash
python upload_to_pinecone.py
```

This takes a few hours for 140K papers. You only do it once.

### Step 2: Deploy the web app on Render (free)

1. Push these files to GitHub:
   - `app.py`
   - `index.html`
   - `requirements.txt`
   - `Procfile`
   - `.python-version`

   (Do NOT upload `upload_to_pinecone.py` — it's a local-only script)

2. Go to https://render.com → New + → Web Service → connect your repo
3. Settings:
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - **Instance Type:** Free
4. Deploy and wait 2-3 minutes

### Step 3: Use it

1. Visit your Render URL
2. Enter your OpenAI API key
3. **LimAgents tab:** Upload PDF → get general limitations
4. **NovAgents tab:** Enter Pinecone key + index name → upload PDF → get novelty limitations
5. **Merge tab:** Both outputs auto-fill → click merge → get final report

## How Retrieval Works

When you run NovAgents:
1. First 500 words of your PDF are embedded using `text-embedding-3-small`
2. Top 20 similar papers fetched from Pinecone (dense cosine similarity)
3. BM25 keyword scoring applied to those 20 candidates
4. Hybrid ranking: **70% dense + 30% BM25**
5. Top 3 papers selected and summarized
6. Novelty agents compare your paper against these 3

## File Overview

| File | Purpose | Upload to GitHub? |
|------|---------|:-:|
| `app.py` | Backend (all 3 pipelines + Pinecone retrieval) | ✅ |
| `index.html` | Frontend (3-tab UI) | ✅ |
| `requirements.txt` | Python dependencies | ✅ |
| `Procfile` | Render start command | ✅ |
| `.python-version` | Pin Python 3.11 | ✅ |
| `upload_to_pinecone.py` | One-time data upload script | ❌ (local only) |
| `README.md` | This file | Optional |

## Notes

- **Free tier:** Render sleeps after 15 min idle. First request takes ~30 sec.
- **Pinecone free tier:** 5GB storage, handles 140K papers easily.
- **Cost:** $0 hosting. Users provide their own OpenAI + Pinecone keys.
