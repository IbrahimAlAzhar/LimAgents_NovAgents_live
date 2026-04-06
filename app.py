"""
LimGen v2 – Multi-Agent Limitation + Novelty Generator
Three pipelines:
  1. LimAgents  – 6 specialist agents analyze limitations from the paper itself
  2. NovAgents  – 7 specialist agents analyze novelty vs top-3 retrieved papers (Pinecone)
  3. Merge      – combines both outputs into a polished final report
"""

import asyncio
import json
import math
import os
import re
import traceback
from typing import AsyncGenerator

import fitz  # PyMuPDF
import numpy as np
import tiktoken
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from openai import OpenAI
from pinecone import Pinecone
from rank_bm25 import BM25Okapi

# ═══════════════════════════════════════════════════════════════
# App Setup
# ═══════════════════════════════════════════════════════════════
app = FastAPI(title="LimGen API v2")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

SAFE_TOKEN_LIMIT = 44_000
EMBED_MODEL = "text-embedding-3-small"
FIRST_N_WORDS = 500  # words used for retrieval matching


# ═══════════════════════════════════════════════════════════════
# Serve Frontend
# ═══════════════════════════════════════════════════════════════
@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"), media_type="text/html")


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════
def _enc():
    try:
        return tiktoken.get_encoding("o200k_base")
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def truncate(text: str, max_tokens: int = SAFE_TOKEN_LIMIT) -> str:
    if not text:
        return ""
    enc = _enc()
    toks = enc.encode(text)
    if len(toks) <= max_tokens:
        return text
    return enc.decode(toks[:max_tokens]) + "\n... [TRUNCATED]"


def first_n_words(text: str, n: int = FIRST_N_WORDS) -> str:
    words = text.split()
    return " ".join(words[:n])


def extract_pdf_text(pdf_bytes: bytes) -> str:
    parts = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            parts.append(page.get_text())
    return "\n".join(parts)


def call_openai(client: OpenAI, model: str, system: str, user: str,
                temperature: float = 0.2, max_tokens: int = 1500) -> str:
    resp = client.chat.completions.create(
        model=model, temperature=temperature, max_tokens=max_tokens,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return resp.choices[0].message.content.strip()


def event(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


# ═══════════════════════════════════════════════════════════════
# Pinecone Retrieval  (70% dense + 30% BM25 rerank)
# ═══════════════════════════════════════════════════════════════
def retrieve_top3(openai_client: OpenAI, pinecone_key: str, index_name: str,
                  query_text: str) -> list[dict]:
    """
    1. Embed query (first 500 words)
    2. Get top-20 from Pinecone (dense cosine)
    3. BM25 rerank on returned texts
    4. Combine 0.7*dense + 0.3*bm25 → return top 3
    """
    # Dense embedding
    embed_resp = openai_client.embeddings.create(model=EMBED_MODEL, input=query_text)
    query_vec = embed_resp.data[0].embedding

    # Pinecone query
    pc = Pinecone(api_key=pinecone_key)
    idx = pc.Index(index_name)
    results = idx.query(vector=query_vec, top_k=20, include_metadata=True)

    if not results.matches:
        return []

    # Extract candidates
    candidates = []
    for m in results.matches:
        text = m.metadata.get("text", "")
        candidates.append({
            "id": m.id,
            "text": text,
            "dense_score": float(m.score),
        })

    if not candidates:
        return []

    # BM25 rerank on candidate texts
    tokenized_docs = [c["text"].lower().split() for c in candidates]
    query_tokens = query_text.lower().split()

    bm25 = BM25Okapi(tokenized_docs)
    bm25_scores = bm25.get_scores(query_tokens)

    # Normalize scores to [0, 1]
    dense_scores = np.array([c["dense_score"] for c in candidates])
    d_min, d_max = dense_scores.min(), dense_scores.max()
    if d_max > d_min:
        dense_norm = (dense_scores - d_min) / (d_max - d_min)
    else:
        dense_norm = np.ones_like(dense_scores)

    b_min, b_max = bm25_scores.min(), bm25_scores.max()
    if b_max > b_min:
        bm25_norm = (bm25_scores - b_min) / (b_max - b_min)
    else:
        bm25_norm = np.ones_like(bm25_scores)

    # Hybrid: 70% dense + 30% BM25
    hybrid = 0.7 * dense_norm + 0.3 * bm25_norm

    # Sort and pick top 3
    ranked_idx = np.argsort(-hybrid)[:3]
    top3 = []
    for ri in ranked_idx:
        c = candidates[ri]
        c["hybrid_score"] = float(hybrid[ri])
        top3.append(c)

    return top3


# ═══════════════════════════════════════════════════════════════
# PIPELINE 1: LimAgents (Limitation Analysis)
# ═══════════════════════════════════════════════════════════════
LIM_PROMPTS = {
    "Novelty & Significance": (
        "You are a highly skeptical expert focused on limitations related to novelty and significance. "
        "Scrutinize whether contributions are truly novel or merely incremental, whether claims of importance are overstated, "
        "whether the problem is impactful, and whether motivations or real-world relevance are weakly justified.\n"
        "Look for: rebranding existing ideas, lack of differentiation from prior work, exaggerated claims, narrow scope, "
        "or failure to articulate why the work matters.\n"
        "Provide a concise bullet list of novelty/significance limitations with evidence from the paper.\n\nPAPER:\n{paper}"
    ),
    "Theoretical & Methodological": (
        "You are an expert in theoretical and methodological soundness. "
        "Scrutinize the core method, theoretical claims, and component breakdowns for flaws, unrealistic assumptions, "
        "missing proofs, logical gaps, oversimplifications, or failure to explain why the method works.\n"
        "Provide a bullet list of theoretical/methodological/ablation limitations with evidence.\n\nPAPER:\n{paper}"
    ),
    "Experimental Evaluation": (
        "You specialize in experimental evaluation, validation, rigor, comparisons, baselines, and metrics. "
        "Find weaknesses: insufficient runs, no statistical significance, cherry-picked results, narrow conditions, "
        "inappropriate baselines, incomplete comparisons, misleading metrics.\n"
        "Provide a bullet list of experimental limitations.\n\nPAPER:\n{paper}"
    ),
    "Generalization & Robustness": (
        "Your expertise covers generalization, robustness, computational efficiency, and real-world applicability. "
        "Evaluate: overfitting to benchmarks, no OOD testing, hyperparameter sensitivity, excessive resource demands, "
        "reliance on synthetic data, or over-optimistic deployment assumptions.\n"
        "Provide a bullet list of generalization/robustness/efficiency limitations.\n\nPAPER:\n{paper}"
    ),
    "Clarity & Reproducibility": (
        "You focus on clarity, interpretability, and reproducibility. "
        "Scrutinize: unclear explanations, lack of explainability, insufficient replication details, "
        "missing code/data/hyperparameters, black-box behavior without insight.\n"
        "Provide a bullet list of clarity/reproducibility limitations.\n\nPAPER:\n{paper}"
    ),
    "Data & Ethics": (
        "You specialize in data integrity, bias, fairness, and ethics. "
        "Scrutinize: dataset quality, labeling, representativeness, data leakage, biased outcomes, "
        "lack of fairness metrics, unreported subgroup performance, misuse potential.\n"
        "Provide a bullet list of data/bias/ethics limitations.\n\nPAPER:\n{paper}"
    ),
}


async def pipeline_lim(api_key: str, pdf_bytes: bytes, model: str) -> AsyncGenerator[str, None]:
    try:
        yield event({"type": "status", "message": "Extracting text from PDF..."})
        raw = await asyncio.to_thread(extract_pdf_text, pdf_bytes)
        if len(raw.strip()) < 100:
            yield event({"type": "error", "message": "PDF has too little text."})
            return
        paper = truncate(raw, SAFE_TOKEN_LIMIT)
        yield event({"type": "status", "message": f"Extracted ~{len(paper.split()):,} words."})

        client = OpenAI(api_key=api_key)
        specialist_outputs = {}

        for idx, (name, tmpl) in enumerate(LIM_PROMPTS.items(), 1):
            yield event({"type": "status", "message": f"Agent {idx}/6: {name}...", "agent": name})
            out = await asyncio.to_thread(
                call_openai, client, model, tmpl.format(paper=paper),
                "Analyze the paper and list limitations as bullet points.", 0.2, 1500
            )
            specialist_outputs[name] = out
            yield event({"type": "agent", "agent": name, "content": out})

        # Leader
        yield event({"type": "status", "message": "Leader Agent compiling..."})
        handoff = "\n\n".join(f"[{n}]\n{o}" for n, o in specialist_outputs.items())
        leader_out = await asyncio.to_thread(
            call_openai, client, model,
            "You are the Leader Agent. Review specialist outputs. De-duplicate, strengthen vague points with paper evidence. Preserve agent sections.",
            f"Specialist outputs:\n\n{handoff}", 0.1, 2500
        )
        yield event({"type": "agent", "agent": "Leader Agent", "content": leader_out})

        # Master
        yield event({"type": "status", "message": "Master Agent finalizing..."})
        final = await asyncio.to_thread(
            call_openai, client, model,
            "You are the Master Agent. Merge into one non-redundant list grouped by category. Use ONLY what appears in the input. Format:\n- **Category:** Description",
            f"Compiled analyses:\n\n{leader_out}", 0.0, 2000
        )
        yield event({"type": "result", "content": final})

    except Exception as e:
        traceback.print_exc()
        yield event({"type": "error", "message": str(e)})


# ═══════════════════════════════════════════════════════════════
# PIPELINE 2: NovAgents (Novelty Analysis vs Retrieved Papers)
# ═══════════════════════════════════════════════════════════════
NOV_PROMPTS = {
    "Technical Contributions": (
        "Identify limitations in the paper's technical contributions that undermine claims of novelty.\n"
        "Focus on: rebranded existing methods, minor tweaks, combinations of known components, "
        "lack of substantive advancement beyond prior work.\n"
        "You MUST compare against the Retrieved Papers (Paper B) summaries.\n"
        "Output format:\n"
        "Limitations in Technical Contributions (A vs B):\n"
        "- <limitation with explanation; compare against Paper B>\n"
        "Evidence:\n- A: <pointer from main paper>\n- B: <pointer from retrieved papers>\n\n"
        "=== MAIN PAPER (A) ===\n{paper}\n\n=== RETRIEVED PAPERS (B) ===\n{retrieved}"
    ),
    "Experimental Validation": (
        "Identify limitations in experimental design, benchmarking, or comparative analysis "
        "that weaken the paper's novelty (missing baselines, inadequate datasets, no ablation, overstated improvements).\n"
        "You MUST compare against the Retrieved Papers (Paper B).\n"
        "Output format:\n"
        "Limitations in Experimental Validation (A vs B):\n"
        "- <limitation; compare against Paper B>\n"
        "Evidence:\n- A: <pointer>\n- B: <pointer>\n\n"
        "=== MAIN PAPER (A) ===\n{paper}\n\n=== RETRIEVED PAPERS (B) ===\n{retrieved}"
    ),
    "Literature Review": (
        "Identify limitations in the literature review or positioning that undermine perceived novelty "
        "(overlooking key prior work, vague differentiation, failure to explain why the gap matters).\n"
        "You MUST compare against the Retrieved Papers (Paper B).\n"
        "Output format:\n"
        "Limitations in Literature Review (A vs B):\n"
        "- <limitation; compare against Paper B>\n"
        "Evidence:\n- A: <pointer>\n- B: <pointer>\n\n"
        "=== MAIN PAPER (A) ===\n{paper}\n\n=== RETRIEVED PAPERS (B) ===\n{retrieved}"
    ),
    "Scope & Generalizability": (
        "Identify limitations in scope, datasets, tasks, or implications that restrict broader significance "
        "(narrow domain, toy settings, ignored real-world constraints).\n"
        "You MUST compare against the Retrieved Papers (Paper B).\n"
        "Output format:\n"
        "Limitations in Scope & Generalizability (A vs B):\n"
        "- <limitation; compare against Paper B>\n"
        "Evidence:\n- A: <pointer>\n- B: <pointer>\n\n"
        "=== MAIN PAPER (A) ===\n{paper}\n\n=== RETRIEVED PAPERS (B) ===\n{retrieved}"
    ),
    "Claims & Overclaiming": (
        "Identify limitations from overstated novelty, impact, or effectiveness claims "
        "that lack supporting evidence or ignore caveats.\n"
        "You MUST compare against the Retrieved Papers (Paper B).\n"
        "Output format:\n"
        "Limitations in Claims & Overclaiming (A vs B):\n"
        "- <limitation; compare against Paper B>\n"
        "Evidence:\n- A: <pointer>\n- B: <pointer>\n\n"
        "=== MAIN PAPER (A) ===\n{paper}\n\n=== RETRIEVED PAPERS (B) ===\n{retrieved}"
    ),
    "Methodological Rigor": (
        "Identify limitations in methodological description, reproducibility, or rigor "
        "that erode confidence in claimed novelty (missing details, ambiguous setups, unverifiable experiments).\n"
        "You MUST compare against the Retrieved Papers (Paper B).\n"
        "Output format:\n"
        "Limitations in Methodological Rigor (A vs B):\n"
        "- <limitation; compare against Paper B>\n"
        "Evidence:\n- A: <pointer>\n- B: <pointer>\n\n"
        "=== MAIN PAPER (A) ===\n{paper}\n\n=== RETRIEVED PAPERS (B) ===\n{retrieved}"
    ),
}

NOV_MASTER_PROMPT = """You are the Master Agent for novelty analysis.
Synthesize specialist outputs into a final novelty limitations report.

CRITICAL RULES:
1. State limitations directly about the evaluated paper only.
2. NEVER use "Paper A", "Paper B", "the main paper", or "the retrieved papers".
3. Transform comparative statements into objective weaknesses.
   INSTEAD OF: "Paper A lacks robust baselines compared to Paper B."
   WRITE: "The experimental validation lacks robust baselines, failing to account for contemporary state-of-the-art standards."
4. Remove redundancies. Be professional and objective.

OUTPUT FORMAT:
**Technical Contributions:**
- <limitation>

**Experimental Validation:**
- <limitation>

**Literature Review & Contextualization:**
- <limitation>

**Scope & Generalizability:**
- <limitation>

**Claims & Overclaiming:**
- <limitation>

**Methodological Clarity & Rigor:**
- <limitation>
"""


async def pipeline_nov(api_key: str, pdf_bytes: bytes, model: str,
                       pinecone_key: str, index_name: str) -> AsyncGenerator[str, None]:
    try:
        yield event({"type": "status", "message": "Extracting text from PDF..."})
        raw = await asyncio.to_thread(extract_pdf_text, pdf_bytes)
        if len(raw.strip()) < 100:
            yield event({"type": "error", "message": "PDF has too little text."})
            return
        paper = truncate(raw, SAFE_TOKEN_LIMIT)
        query = first_n_words(raw, FIRST_N_WORDS)
        yield event({"type": "status", "message": f"Extracted ~{len(paper.split()):,} words."})

        client = OpenAI(api_key=api_key)

        # Retrieve top 3 papers
        yield event({"type": "status", "message": "Searching 140K papers in Pinecone (70% dense + 30% BM25)..."})
        try:
            top3 = await asyncio.to_thread(retrieve_top3, client, pinecone_key, index_name, query)
        except Exception as e:
            yield event({"type": "error", "message": f"Pinecone retrieval failed: {e}"})
            return

        if not top3:
            yield event({"type": "error", "message": "No similar papers found in Pinecone index."})
            return

        yield event({"type": "status", "message": f"Retrieved {len(top3)} similar papers."})

        # Summarize retrieved papers
        retrieved_summaries = []
        for i, paper_b in enumerate(top3):
            yield event({"type": "status", "message": f"Summarizing retrieved paper {i+1}/3..."})
            text_b = truncate(paper_b["text"], max_tokens=3000)
            summary = await asyncio.to_thread(
                call_openai, client, model,
                "Summarize this paper focusing on its methods, contributions, and key findings. Be concise but thorough.",
                f"Paper text:\n{text_b}", 0.2, 700
            )
            retrieved_summaries.append(f"--- Retrieved Paper #{i+1} (score: {paper_b['hybrid_score']:.3f}) ---\n{summary}")

        retrieved_text = "\n\n".join(retrieved_summaries)
        yield event({
            "type": "agent", "agent": "Retrieval",
            "content": f"Top 3 papers retrieved and summarized:\n\n{retrieved_text}"
        })

        # Run novelty specialists
        specialist_outputs = {}
        for idx, (name, tmpl) in enumerate(NOV_PROMPTS.items(), 1):
            yield event({"type": "status", "message": f"Novelty Agent {idx}/6: {name}...", "agent": name})
            prompt = tmpl.format(paper=paper, retrieved=retrieved_text)
            out = await asyncio.to_thread(
                call_openai, client, model, prompt,
                "Analyze and provide novelty-related limitations as specified.", 0.2, 1500
            )
            specialist_outputs[name] = out
            yield event({"type": "agent", "agent": name, "content": out})

        # Leader
        yield event({"type": "status", "message": "Leader Agent compiling novelty findings..."})
        handoff = "\n\n".join(f"[{n}]\n{o}" for n, o in specialist_outputs.items())
        leader_out = await asyncio.to_thread(
            call_openai, client, model,
            "You are the Leader Agent for novelty analysis. Review all specialist outputs. "
            "De-duplicate, strengthen vague points, ensure all claims reference evidence from the paper or retrieved papers. "
            "Preserve agent sections.",
            f"Specialist outputs:\n\n{handoff}", 0.1, 2500
        )
        yield event({"type": "agent", "agent": "Leader Agent", "content": leader_out})

        # Master
        yield event({"type": "status", "message": "Master Agent producing final novelty report..."})
        final = await asyncio.to_thread(
            call_openai, client, model, NOV_MASTER_PROMPT,
            f"Leader compiled analyses:\n\n{leader_out}", 0.0, 2000
        )
        yield event({"type": "result", "content": final})

    except Exception as e:
        traceback.print_exc()
        yield event({"type": "error", "message": str(e)})


# ═══════════════════════════════════════════════════════════════
# PIPELINE 3: Merge (LimAgents + NovAgents)
# ═══════════════════════════════════════════════════════════════
MERGE_PROMPT = """You are the **Final Merge Agent**. You receive two sets of limitation analyses for the same research paper:

1. **LimAgents Output** — General limitations identified from analyzing the paper itself (covering methodology, experiments, generalization, clarity, data quality, and ethics).

2. **NovAgents Output** — Novelty-related limitations identified by comparing the paper against similar published works (covering technical contributions, experimental validation, literature gaps, scope, overclaiming, and methodological rigor).

YOUR TASK:
Produce a single, polished, comprehensive, non-redundant set of limitations.

RULES:
1. **Remove duplicates**: If both outputs mention the same limitation (even worded differently), keep the more specific and evidence-rich version.
2. **Remove vague/generic points**: Drop limitations that are too generic (e.g., "could be improved") without specific evidence.
3. **Remove redundant/repetitive items**: If two points overlap substantially, merge into one stronger point.
4. **Preserve all unique findings**: Every distinct, evidence-backed limitation from either source must appear.
5. **Be objective**: Write in professional academic review language. Never reference "Paper A", "Paper B", or "agents".

OUTPUT FORMAT:

## General Limitations (from paper analysis)
- **[Category]:** Specific limitation with evidence.
- **[Category]:** ...

## Novelty & Comparative Limitations (from comparison with prior work)
- **[Category]:** Specific limitation with evidence.
- **[Category]:** ...

## Consolidated Key Limitations
(Top 8-12 most impactful limitations from both sources, ranked by severity)
1. **[Short title]:** Detailed limitation.
2. ...
"""


async def pipeline_merge(api_key: str, model: str,
                         lim_output: str, nov_output: str) -> AsyncGenerator[str, None]:
    try:
        yield event({"type": "status", "message": "Merge Agent combining outputs..."})
        client = OpenAI(api_key=api_key)

        user_msg = (
            "=== LIMAGENTS OUTPUT ===\n"
            f"{lim_output}\n\n"
            "=== NOVAGENTS OUTPUT ===\n"
            f"{nov_output}"
        )

        final = await asyncio.to_thread(
            call_openai, client, model, MERGE_PROMPT, user_msg, 0.1, 3000
        )
        yield event({"type": "result", "content": final})

    except Exception as e:
        traceback.print_exc()
        yield event({"type": "error", "message": str(e)})


# ═══════════════════════════════════════════════════════════════
# API Endpoints
# ═══════════════════════════════════════════════════════════════
@app.post("/api/lim-agents")
async def endpoint_lim(
    file: UploadFile = File(...),
    api_key: str = Form(...),
    model: str = Form("gpt-4o-mini"),
):
    pdf_bytes = await file.read()
    return StreamingResponse(
        pipeline_lim(api_key, pdf_bytes, model),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/nov-agents")
async def endpoint_nov(
    file: UploadFile = File(...),
    api_key: str = Form(...),
    model: str = Form("gpt-4o-mini"),
    pinecone_key: str = Form(...),
    index_name: str = Form(...),
):
    pdf_bytes = await file.read()
    return StreamingResponse(
        pipeline_nov(api_key, pdf_bytes, model, pinecone_key, index_name),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/merge")
async def endpoint_merge(
    api_key: str = Form(...),
    model: str = Form("gpt-4o-mini"),
    lim_output: str = Form(...),
    nov_output: str = Form(...),
):
    return StreamingResponse(
        pipeline_merge(api_key, model, lim_output, nov_output),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/health")
def health():
    return {"status": "ok", "version": "2.0"}
