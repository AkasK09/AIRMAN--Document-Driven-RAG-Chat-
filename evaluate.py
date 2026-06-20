"""
AIRMAN RAG Evaluation Framework
================================
Runs 50 predefined aviation questions through both:
  - Level 1: Vector Search Only (FAISS semantic)
  - Level 2: Hybrid Search + Reranker (FAISS + BM25 + CrossEncoder)

Evaluation uses deterministic heuristics (keyword matching, similarity
scores) so it can run entirely WITHOUT LLM API calls.  This makes
the benchmark 100 % reproducible and immune to rate limits.

Generates a professional evaluation report (report.md).

Usage:
    python evaluate.py
"""

import json
import re
import time
import os
from datetime import datetime
from typing import List, Dict, Any, Tuple
import numpy as np
import faiss

from app.config import settings
from app.ingest import get_embedding_model, load_or_create_vector_store
from app.rag import (
    hybrid_search,
    rerank_chunks,
    call_llm,
    load_stores,
    REFUSAL_RESPONSE,
)
from app.utils import setup_logger

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# 50 Evaluation Questions with expected keywords for deterministic judging
# ---------------------------------------------------------------------------

EVAL_QUESTIONS: List[Dict[str, Any]] = [
    # --- Simple Factual (1-20) ---
    {"id": 1, "category": "Simple Factual", "question": "What is the definition of meteorology?",
     "keywords": ["meteorology", "atmosphere", "weather", "science", "study"]},
    {"id": 2, "category": "Simple Factual", "question": "What percentage of the atmosphere is composed of nitrogen?",
     "keywords": ["nitrogen", "78", "percent", "atmosphere"]},
    {"id": 3, "category": "Simple Factual", "question": "What is the average height of the tropopause over the Equator?",
     "keywords": ["tropopause", "equator", "16", "17", "18", "km", "feet", "height"]},
    {"id": 4, "category": "Simple Factual", "question": "What is the standard sea level pressure in ISA?",
     "keywords": ["1013", "hpa", "millibar", "29.92", "pressure", "sea level", "isa"]},
    {"id": 5, "category": "Simple Factual", "question": "What is the ISA temperature at mean sea level?",
     "keywords": ["15", "celsius", "temperature", "isa", "sea level"]},
    {"id": 6, "category": "Simple Factual", "question": "What does ADF stand for in radio navigation?",
     "keywords": ["automatic", "direction", "finder", "adf"]},
    {"id": 7, "category": "Simple Factual", "question": "What is the primary purpose of a VOR system?",
     "keywords": ["vor", "omnidirectional", "range", "bearing", "navigation", "radial"]},
    {"id": 8, "category": "Simple Factual", "question": "What does DME measure?",
     "keywords": ["dme", "distance", "measure", "slant", "range", "nautical"]},
    {"id": 9, "category": "Simple Factual", "question": "What is the function of an Instrument Landing System (ILS)?",
     "keywords": ["ils", "instrument", "landing", "localizer", "glide", "approach"]},
    {"id": 10, "category": "Simple Factual", "question": "What is a rhumb line in navigation?",
     "keywords": ["rhumb", "loxodrome", "constant", "bearing", "meridian", "track"]},
    {"id": 11, "category": "Simple Factual", "question": "What is the purpose of a Mercator chart?",
     "keywords": ["mercator", "chart", "projection", "navigation", "rhumb", "straight"]},
    {"id": 12, "category": "Simple Factual", "question": "What is the 1-in-60 rule used for?",
     "keywords": ["1-in-60", "one in sixty", "track", "error", "distance", "degree", "correction"]},
    {"id": 13, "category": "Simple Factual", "question": "What is the role of an Air Information Publication (AIP)?",
     "keywords": ["aip", "air", "information", "publication", "aeronautical"]},
    {"id": 14, "category": "Simple Factual", "question": "What does NOTAM stand for?",
     "keywords": ["notam", "notice", "airmen", "air", "mission"]},
    {"id": 15, "category": "Simple Factual", "question": "What is the purpose of a Point of Equal Time (PET)?",
     "keywords": ["pet", "point", "equal", "time", "critical", "diversion"]},
    {"id": 16, "category": "Simple Factual", "question": "What is the purpose of a Point of Safe Return (PSR)?",
     "keywords": ["psr", "point", "safe", "return", "no return", "fuel"]},
    {"id": 17, "category": "Simple Factual", "question": "What is the function of an airspeed indicator?",
     "keywords": ["airspeed", "indicator", "speed", "pitot", "static", "knots"]},
    {"id": 18, "category": "Simple Factual", "question": "What does a vertical speed indicator display?",
     "keywords": ["vertical", "speed", "indicator", "climb", "descent", "rate", "vsi"]},
    {"id": 19, "category": "Simple Factual", "question": "What is the purpose of an autopilot system?",
     "keywords": ["autopilot", "automatic", "flight", "control", "pilot", "workload"]},
    {"id": 20, "category": "Simple Factual", "question": "What is the function of a Ground Proximity Warning System (GPWS)?",
     "keywords": ["gpws", "ground", "proximity", "warning", "terrain", "alert"]},
    # --- Applied (21-40) ---
    {"id": 21, "category": "Applied", "question": "A pilot flying near the Equator wants to estimate the expected tropopause height. What value should be used?",
     "keywords": ["tropopause", "equator", "16", "17", "18", "km", "feet"]},
    {"id": 22, "category": "Applied", "question": "An aircraft is operating at 18,000 ft. How can the pilot determine the ISA temperature at that altitude?",
     "keywords": ["isa", "temperature", "18000", "lapse", "rate", "2", "degree", "celsius"]},
    {"id": 23, "category": "Applied", "question": "During an approach, the aircraft receives localizer and glide slope signals. Which navigation system is being used?",
     "keywords": ["ils", "instrument", "landing", "localizer", "glide", "slope"]},
    {"id": 24, "category": "Applied", "question": "A pilot needs the distance to a navigation beacon. Which radio navigation aid should be used?",
     "keywords": ["dme", "distance", "measuring", "beacon"]},
    {"id": 25, "category": "Applied", "question": "An aircraft drifts to the right of its planned track. How can the 1-in-60 rule be applied to regain track?",
     "keywords": ["1-in-60", "track", "drift", "correction", "angle", "degree"]},
    {"id": 26, "category": "Applied", "question": "A pilot wants to follow the shortest route between two distant points on Earth. Which type of route should be used?",
     "keywords": ["great", "circle", "shortest", "route", "orthodrome"]},
    {"id": 27, "category": "Applied", "question": "While planning a flight, where should a pilot look for official aeronautical information and procedures?",
     "keywords": ["aip", "air", "information", "publication", "notam"]},
    {"id": 28, "category": "Applied", "question": "A NOTAM reports a runway closure. How should this information affect flight planning?",
     "keywords": ["notam", "runway", "closure", "alternate", "divert", "plan"]},
    {"id": 29, "category": "Applied", "question": "During flight planning, how does a strong headwind affect fuel requirements?",
     "keywords": ["headwind", "fuel", "consumption", "ground", "speed", "increase", "time"]},
    {"id": 30, "category": "Applied", "question": "When calculating mass and balance, why is it important to ensure the center of gravity remains within limits?",
     "keywords": ["center", "gravity", "balance", "stability", "control", "limit", "cg"]},
    {"id": 31, "category": "Applied", "question": "A pilot notices abnormal fluctuations in airspeed indications. Which system should be checked first?",
     "keywords": ["pitot", "static", "airspeed", "system", "blockage", "pressure"]},
    {"id": 32, "category": "Applied", "question": "During an instrument approach, the glide slope indication fails. What information remains available from the ILS?",
     "keywords": ["localizer", "ils", "glide", "slope", "lateral", "guidance"]},
    {"id": 33, "category": "Applied", "question": "A pilot is using VOR navigation and receives an ambiguous indication. What factors should be verified?",
     "keywords": ["vor", "flag", "to", "from", "radial", "signal", "cone"]},
    {"id": 34, "category": "Applied", "question": "An aircraft must determine the point beyond which returning to the departure airport is no longer advantageous. Which calculation should be used?",
     "keywords": ["pet", "psr", "point", "equal", "time", "return", "no return"]},
    {"id": 35, "category": "Applied", "question": "A pilot observes cumulonimbus clouds along the route. What operational concerns should be considered?",
     "keywords": ["cumulonimbus", "thunderstorm", "turbulence", "icing", "lightning", "hail", "wind"]},
    {"id": 36, "category": "Applied", "question": "A flight is operating in ISA+15 conditions. How might aircraft performance be affected?",
     "keywords": ["isa", "temperature", "performance", "density", "altitude", "takeoff", "climb"]},
    {"id": 37, "category": "Applied", "question": "A pilot receives a GPWS warning during descent. What should be the immediate response?",
     "keywords": ["gpws", "warning", "go around", "climb", "power", "terrain"]},
    {"id": 38, "category": "Applied", "question": "A DME indicates increasing distance despite maintaining the same heading. What could this imply about the aircraft's position?",
     "keywords": ["dme", "distance", "abeam", "passed", "station", "heading"]},
    {"id": 39, "category": "Applied", "question": "During route planning, why is knowledge of upper winds important?",
     "keywords": ["wind", "upper", "fuel", "time", "ground", "speed", "route", "jet"]},
    {"id": 40, "category": "Applied", "question": "An aircraft experiences moderate icing conditions. Which meteorological information should be reviewed before continuing?",
     "keywords": ["icing", "temperature", "freezing", "level", "sigmet", "cloud", "supercooled"]},
    # --- Higher-Order Reasoning (41-50) ---
    {"id": 41, "category": "Higher-Order Reasoning", "question": "How would a combination of strong headwinds, low temperatures, and icing conditions affect flight planning and fuel management?",
     "keywords": ["headwind", "fuel", "icing", "temperature", "performance", "consumption", "alternate"]},
    {"id": 42, "category": "Higher-Order Reasoning", "question": "Compare the advantages and limitations of VOR navigation versus GNSS navigation for long-distance flights.",
     "keywords": ["vor", "gnss", "gps", "accuracy", "range", "line of sight", "satellite", "coverage"]},
    {"id": 43, "category": "Higher-Order Reasoning", "question": "How can meteorological forecasts and NOTAM information be combined to improve operational decision-making?",
     "keywords": ["meteorological", "notam", "forecast", "decision", "planning", "weather", "operational"]},
    {"id": 44, "category": "Higher-Order Reasoning", "question": "Why might a great-circle route be preferred over a rhumb-line route on long-haul flights, and what navigation challenges could arise?",
     "keywords": ["great", "circle", "rhumb", "distance", "shorter", "heading", "change", "waypoint"]},
    {"id": 45, "category": "Higher-Order Reasoning", "question": "How do ISA deviations influence aircraft performance calculations and operational planning?",
     "keywords": ["isa", "deviation", "performance", "density", "altitude", "temperature", "pressure"]},
    {"id": 46, "category": "Higher-Order Reasoning", "question": "Explain how PET and PSR can lead to different diversion decisions during an oceanic flight.",
     "keywords": ["pet", "psr", "diversion", "oceanic", "point", "equal", "time", "return"]},
    {"id": 47, "category": "Higher-Order Reasoning", "question": "If both ADF and VOR information are available, how should a pilot determine the most reliable navigation source for a specific phase of flight?",
     "keywords": ["adf", "vor", "accuracy", "reliability", "phase", "approach", "enroute"]},
    {"id": 48, "category": "Higher-Order Reasoning", "question": "Analyze how inaccurate mass and balance calculations could affect aircraft controllability, fuel efficiency, and safety.",
     "keywords": ["mass", "balance", "center", "gravity", "control", "stability", "fuel", "safety"]},
    {"id": 49, "category": "Higher-Order Reasoning", "question": "During an instrument approach in deteriorating weather, how can ILS, weather reports, and aircraft instrumentation be used together to maintain situational awareness?",
     "keywords": ["ils", "weather", "instrument", "approach", "situational", "awareness", "minima"]},
    {"id": 50, "category": "Higher-Order Reasoning", "question": "A flight encounters unexpected weather, navigation equipment degradation, and fuel limitations simultaneously. What factors should be prioritized to ensure a safe operational decision?",
     "keywords": ["weather", "navigation", "fuel", "priority", "safety", "diversion", "decision"]},
]

# ---------------------------------------------------------------------------
# Vector-Only Search (Level 1)
# ---------------------------------------------------------------------------

def vector_only_search(
    query: str,
    index: faiss.Index,
    metadata: List[Dict[str, Any]],
    top_k: int = 20,
) -> List[Dict[str, Any]]:
    """Pure FAISS semantic search -- no BM25, no filtering."""
    model = get_embedding_model()
    query_vector = model.encode([query], show_progress_bar=False)
    query_vector = np.array(query_vector).astype("float32")
    faiss.normalize_L2(query_vector)

    distances, indices_arr = index.search(query_vector, top_k)

    results = []
    for i, idx in enumerate(indices_arr[0]):
        if idx == -1 or idx >= len(metadata):
            continue
        chunk = metadata[idx].copy()
        chunk["semantic_score"] = float(distances[0][i])
        chunk["hybrid_score"] = chunk["semantic_score"]
        chunk["bm25_score"] = 0.0
        results.append(chunk)
    return results

# ---------------------------------------------------------------------------
# Deterministic Heuristic Judges (ZERO LLM calls)
# ---------------------------------------------------------------------------

def judge_retrieval_hit(keywords: List[str], chunks: List[Dict]) -> bool:
    """Did at least one chunk contain at least 2 expected keywords?"""
    if not chunks:
        return False
    for chunk in chunks[:5]:
        text = chunk.get("chunk_text", "").lower()
        matches = sum(1 for kw in keywords if kw.lower() in text)
        if matches >= 2:
            return True
    return False


def judge_faithfulness_heuristic(answer: str, chunks: List[Dict]) -> float:
    """
    Heuristic: extract key noun-phrases from chunks, check how many
    appear in the answer.  1.0 / 0.5 / 0.0.
    """
    if REFUSAL_RESPONSE.lower() in answer.lower() or not answer.strip():
        return 1.0
    if not chunks:
        return 0.0

    # Gather unique significant words from top chunks
    chunk_words = set()
    for c in chunks[:5]:
        words = re.findall(r'[a-zA-Z]{4,}', c.get("chunk_text", "").lower())
        chunk_words.update(words)

    # Check answer words against chunk vocabulary
    answer_words = set(re.findall(r'[a-zA-Z]{4,}', answer.lower()))
    if not answer_words:
        return 0.0

    overlap = answer_words & chunk_words
    ratio = len(overlap) / len(answer_words)

    if ratio >= 0.40:
        return 1.0
    elif ratio >= 0.20:
        return 0.5
    return 0.0


def judge_hallucination_heuristic(answer: str, chunks: List[Dict]) -> bool:
    """Heuristic: If faithfulness < 0.5, assume hallucination."""
    return judge_faithfulness_heuristic(answer, chunks) < 0.5


def judge_accuracy_heuristic(keywords: List[str], answer: str) -> bool:
    """Did the answer mention at least 2 expected keywords?"""
    if REFUSAL_RESPONSE.lower() in answer.lower():
        return True  # refusal is correct behavior
    if not answer.strip():
        return False
    text = answer.lower()
    matches = sum(1 for kw in keywords if kw.lower() in text)
    return matches >= 2


def judge_citation_quality(citations: List[Dict], chunks: List[Dict]) -> str:
    """Correct / Partially Correct / Incorrect"""
    if not citations:
        return "Incorrect"
    if not chunks:
        return "Incorrect"
    cited_docs = {c.get("document") for c in citations}
    chunk_docs = {c.get("document_name") for c in chunks[:5]}
    overlap = cited_docs & chunk_docs
    if len(overlap) == len(cited_docs):
        return "Correct"
    elif overlap:
        return "Partially Correct"
    return "Incorrect"


# ---------------------------------------------------------------------------
# Pipeline runner (retrieval only -- no LLM call)
# ---------------------------------------------------------------------------

def run_retrieval_pipeline(
    question: str,
    index,
    metadata,
    parent_mapping,
    bm25,
    level: int = 2,
) -> Dict[str, Any]:
    """
    Retrieval-only pipeline.  Returns chunks + scores.  No LLM call.
    """
    start = time.time()
    result: Dict[str, Any] = {
        "chunks": [],
        "top_scores": [],
        "latency": 0.0,
        "error": None,
    }

    try:
        if level == 1:
            candidates = vector_only_search(question, index, metadata, top_k=20)
            top_chunks = candidates[:5]
        else:
            candidates = hybrid_search(question, index, metadata, bm25, top_k=20)
            top_chunks = rerank_chunks(question, candidates, top_k=5)

        result["chunks"] = top_chunks
        result["top_scores"] = [
            {
                "hybrid": c.get("hybrid_score", 0),
                "semantic": c.get("semantic_score", 0),
                "bm25": c.get("bm25_score", 0),
                "reranker": c.get("reranker_score", 0),
            }
            for c in top_chunks[:3]
        ]
    except Exception as exc:
        result["error"] = str(exc)

    result["latency"] = time.time() - start
    return result


# ---------------------------------------------------------------------------
# Full RAG pipeline (with LLM) -- used for a small sample only
# ---------------------------------------------------------------------------

def run_full_rag_pipeline(
    question: str,
    index,
    metadata,
    parent_mapping,
    bm25,
    level: int = 2,
) -> Dict[str, Any]:
    """Full pipeline including LLM generation.  Used sparingly."""
    start = time.time()
    result: Dict[str, Any] = {
        "answer": "",
        "confidence": "Unknown",
        "citations": [],
        "chunks": [],
        "latency": 0.0,
        "error": None,
    }

    try:
        if level == 1:
            candidates = vector_only_search(question, index, metadata, top_k=20)
            top_chunks = candidates[:5]
        else:
            candidates = hybrid_search(question, index, metadata, bm25, top_k=20)
            top_chunks = rerank_chunks(question, candidates, top_k=5)

        result["chunks"] = top_chunks

        if not top_chunks:
            result["answer"] = REFUSAL_RESPONSE
            result["confidence"] = "Low"
            result["latency"] = time.time() - start
            return result

        # Build context
        context_text = ""
        for idx, chunk in enumerate(top_chunks):
            parent_id = chunk.get("parent_chunk_id")
            parent_text = parent_mapping.get(parent_id, chunk.get("chunk_text", ""))
            doc_name = chunk.get("document_name", "N/A")
            page_num = chunk.get("page_number", "N/A")
            ata = chunk.get("ata_chapter", "N/A")
            sec = chunk.get("section", "N/A")
            context_text += (
                f"[Context Index: {idx}]\n"
                f"Document: {doc_name}\nATA Chapter: {ata}\nSection: {sec}\nPage: {page_num}\n"
                f"Content: {parent_text}\n\n"
            )

        system_prompt = (
            "You are an Aeronautics QA Assistant.\n\n"
            "Answer ONLY from the provided manual context.\n\n"
            "Do not use outside knowledge.\n\n"
            "If the answer is not available in the retrieved context, respond:\n"
            f"'{REFUSAL_RESPONSE}'\n\n"
            "Always cite the source document, section, ATA chapter, and page number when available.\n\n"
            "You MUST output your response as a JSON object with this EXACT schema:\n"
            "{\n"
            '  "answer": "<grounded response>",\n'
            '  "confidence": "High / Medium / Low",\n'
            '  "used_context_indices": [integer list of context block indices used, e.g. [0, 2]]\n'
            "}\n"
            "Ensure the JSON output is valid."
        )
        user_prompt = f"Context:\n{context_text}\nQuestion:\n{question}\nAnswer:"

        raw = call_llm(system_prompt, user_prompt)
        parsed = json.loads(raw)
        result["answer"] = parsed.get("answer", "").strip()
        result["confidence"] = parsed.get("confidence", "Unknown")
        used_indices = parsed.get("used_context_indices", [])

        for ci in used_indices:
            if isinstance(ci, int) and 0 <= ci < len(top_chunks):
                c = top_chunks[ci]
                result["citations"].append({
                    "document": c.get("document_name", "Unknown"),
                    "page": c.get("page_number"),
                    "section": c.get("section"),
                    "snippet": c.get("chunk_text", "")[:120],
                })
        if not result["citations"] and top_chunks:
            c = top_chunks[0]
            result["citations"].append({
                "document": c.get("document_name", "Unknown"),
                "page": c.get("page_number"),
                "section": c.get("section"),
                "snippet": c.get("chunk_text", "")[:120],
            })

    except Exception as exc:
        result["error"] = str(exc)
        result["answer"] = ""

    result["latency"] = time.time() - start
    return result


# ---------------------------------------------------------------------------
# Main Evaluation Runner
# ---------------------------------------------------------------------------

def run_evaluation():
    print("=" * 70)
    print("  AIRMAN RAG Evaluation Framework")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Load stores once
    print("\n[LOAD] Loading vector stores ...")
    index, metadata, parent_mapping, bm25 = load_stores()
    num_docs = len(set(m.get("document_name") for m in metadata))
    num_chunks = len(metadata)
    print(f"   Documents: {num_docs} | Chunks: {num_chunks}")

    levels = {1: "Vector Only (Level 1)", 2: "Hybrid + Reranker (Level 2)"}
    all_retrieval: Dict[int, List[Dict]] = {1: [], 2: []}

    # ---------------------------------------------------------------
    # Phase 1: Retrieval evaluation (all 50 Qs x 2 levels, NO LLM)
    # ---------------------------------------------------------------
    for level, label in levels.items():
        print(f"\n{'-' * 70}")
        print(f"  [RUN] Retrieval Evaluation - {label}")
        print(f"{'-' * 70}")

        for q in EVAL_QUESTIONS:
            qid = q["id"]
            question = q["question"]
            category = q["category"]
            keywords = q["keywords"]

            ret = run_retrieval_pipeline(
                question, index, metadata, parent_mapping, bm25, level=level
            )

            if ret["error"]:
                print(f"  [{qid:02d}] ERROR: {ret['error'][:60]}")
                all_retrieval[level].append({
                    "id": qid, "category": category, "question": question,
                    "keywords": keywords, "chunks": [], "scores": [],
                    "retrieval_hit": False, "latency": ret["latency"], "error": ret["error"],
                })
                continue

            hit = judge_retrieval_hit(keywords, ret["chunks"])
            top_score = ret["top_scores"][0]["hybrid"] if ret["top_scores"] else 0
            status = "HIT" if hit else "MISS"
            print(f"  [{qid:02d}] {status} | score={top_score:.4f} | {ret['latency']:.2f}s | {question[:55]}...")

            all_retrieval[level].append({
                "id": qid, "category": category, "question": question,
                "keywords": keywords,
                "chunks": [
                    {"chunk_text": c.get("chunk_text", "")[:250], "document_name": c.get("document_name"),
                     "hybrid_score": c.get("hybrid_score", 0), "semantic_score": c.get("semantic_score", 0),
                     "bm25_score": c.get("bm25_score", 0), "reranker_score": c.get("reranker_score", 0)}
                    for c in ret["chunks"][:3]
                ],
                "scores": ret["top_scores"],
                "retrieval_hit": hit,
                "latency": ret["latency"],
                "error": None,
            })

    # ---------------------------------------------------------------
    # Phase 2: Full RAG (LLM) for 5 sample questions (Level 2 only)
    # ---------------------------------------------------------------
    SAMPLE_IDS = [1, 6, 10, 23, 41]
    sample_results: List[Dict] = []

    print(f"\n{'-' * 70}")
    print(f"  [RUN] Full RAG Pipeline (LLM) - 5 sample questions")
    print(f"{'-' * 70}")

    for q in EVAL_QUESTIONS:
        if q["id"] not in SAMPLE_IDS:
            continue
        qid = q["id"]
        question = q["question"]
        keywords = q["keywords"]
        print(f"\n  [{qid:02d}] {question[:65]}...")

        full = run_full_rag_pipeline(question, index, metadata, parent_mapping, bm25, level=2)

        if full["error"]:
            print(f"      [ERROR] {full['error'][:80]}")
            sample_results.append({
                "id": qid, "question": question, "answer": "", "confidence": "Unknown",
                "citations": [], "chunks": [], "faithfulness": 0.0, "accurate": False,
                "hallucinated": True, "citation_quality": "Incorrect",
                "latency": full["latency"], "error": full["error"],
            })
            # Rate-limit pause
            time.sleep(15)
            continue

        answer = full["answer"]
        chunks = full["chunks"]
        citations = full["citations"]

        faith = judge_faithfulness_heuristic(answer, chunks)
        acc = judge_accuracy_heuristic(keywords, answer)
        halluc = judge_hallucination_heuristic(answer, chunks)
        cit_q = judge_citation_quality(citations, chunks)

        print(f"      [OK] Answer: {answer[:80]}...")
        print(f"      [METRICS] Faith={faith} | Acc={acc} | Halluc={halluc} | Cit={cit_q}")

        sample_results.append({
            "id": qid, "question": question, "answer": answer, "confidence": full["confidence"],
            "citations": citations, "chunks": [
                {"chunk_text": c.get("chunk_text", "")[:250], "document_name": c.get("document_name")}
                for c in chunks[:3]
            ],
            "faithfulness": faith, "accurate": acc, "hallucinated": halluc,
            "citation_quality": cit_q, "latency": full["latency"], "error": None,
        })
        # Rate-limit pause
        time.sleep(15)

    # ---------------------------------------------------------------
    # Phase 3: Generate report
    # ---------------------------------------------------------------
    print(f"\n{'-' * 70}")
    print("  [REPORT] Generating report.md ...")
    print(f"{'-' * 70}")
    generate_report(all_retrieval, sample_results, num_docs, num_chunks)
    print("\n[DONE] Evaluation complete! Report saved to report.md")


# ---------------------------------------------------------------------------
# Metric Computation
# ---------------------------------------------------------------------------

def compute_retrieval_metrics(results: List[Dict]) -> Dict[str, Any]:
    valid = [r for r in results if r.get("error") is None]
    n = len(valid) or 1
    hits = sum(1 for r in valid if r["retrieval_hit"])
    avg_lat = sum(r["latency"] for r in valid) / n
    avg_top_score = 0.0
    scored = [r for r in valid if r.get("scores")]
    if scored:
        avg_top_score = sum(r["scores"][0]["hybrid"] for r in scored) / len(scored)

    return {
        "total": len(results),
        "valid": n,
        "retrieval_hit_rate": (hits / n) * 100,
        "avg_latency": avg_lat,
        "avg_top_score": avg_top_score,
    }


def compute_sample_metrics(results: List[Dict]) -> Dict[str, Any]:
    valid = [r for r in results if r.get("error") is None]
    n = len(valid) or 1
    faith_avg = sum(r["faithfulness"] for r in valid) / n
    halluc_count = sum(1 for r in valid if r["hallucinated"])
    acc_count = sum(1 for r in valid if r["accurate"])
    cit_correct = sum(1 for r in valid if r["citation_quality"] == "Correct")
    cit_partial = sum(1 for r in valid if r["citation_quality"] == "Partially Correct")
    return {
        "faithfulness": faith_avg,
        "hallucination_rate": (halluc_count / n) * 100,
        "accuracy": (acc_count / n) * 100,
        "citation_accuracy": ((cit_correct + cit_partial * 0.5) / n) * 100,
    }


# ---------------------------------------------------------------------------
# Report Generator
# ---------------------------------------------------------------------------

def generate_report(
    all_retrieval: Dict[int, List[Dict]],
    sample_results: List[Dict],
    num_docs: int,
    num_chunks: int,
):
    m1 = compute_retrieval_metrics(all_retrieval[1])
    m2 = compute_retrieval_metrics(all_retrieval[2])
    valid_samples = [r for r in sample_results if r.get("error") is None] if sample_results else []
    sm = compute_sample_metrics(valid_samples) if valid_samples else {}

    lines: List[str] = []
    w = lines.append

    w("# AIRMAN RAG Evaluation Report")
    w("")
    w(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    w("")
    w("---")
    w("")

    # Dataset
    w("## Dataset")
    w("")
    w("| Parameter | Value |")
    w("| --- | --- |")
    w(f"| Number of Documents | {num_docs} |")
    w(f"| Number of Chunks | {num_chunks} |")
    w(f"| Chunking Strategy | Parent-Child (Parent: {settings.PARENT_CHUNK_SIZE} tokens, Child: {settings.CHILD_CHUNK_SIZE} tokens, Overlap: 50 tokens) |")
    w(f"| Embedding Model | {settings.EMBEDDING_MODEL} |")
    w(f"| Vector Database | FAISS (IndexFlatIP) |")
    w(f"| LLM Model | {settings.MODEL_NAME} |")
    w(f"| Retrieval Strategy | Hybrid Search (FAISS + BM25 + RRF) + CrossEncoder Reranker |")
    w(f"| Reranker Model | {settings.RERANKER_MODEL} |")
    w(f"| Evaluation Questions | {len(EVAL_QUESTIONS)} |")
    w("")

    # Quantitative Metrics
    w("## Quantitative Metrics")
    w("")
    w("| Metric | Score |")
    w("| --- | --- |")
    w(f"| Retrieval Hit Rate | {m2['retrieval_hit_rate']:.1f}% |")
    if sm:
        w(f"| Faithfulness | {sm['faithfulness']:.2f} / 1.0 |")
        w(f"| Hallucination Rate | {sm['hallucination_rate']:.1f}% |")
        w(f"| Answer Accuracy | {sm['accuracy']:.1f}% |")
        w(f"| Citation Accuracy | {sm['citation_accuracy']:.1f}% |")
    w(f"| Avg Retrieval Latency | {m2['avg_latency']:.2f}s |")
    w(f"| Avg Top-1 Hybrid Score | {m2['avg_top_score']:.4f} |")
    w("")

    # Level 1 vs Level 2
    w("## Level 1 vs Level 2 Comparison")
    w("")
    w("| Metric | Vector Only | Hybrid + Reranker |")
    w("| --- | --- | --- |")
    w(f"| Retrieval Hit Rate | {m1['retrieval_hit_rate']:.1f}% | {m2['retrieval_hit_rate']:.1f}% |")
    w(f"| Avg Retrieval Latency | {m1['avg_latency']:.2f}s | {m2['avg_latency']:.2f}s |")
    w(f"| Avg Top-1 Score | {m1['avg_top_score']:.4f} | {m2['avg_top_score']:.4f} |")

    # Per-question comparison
    w("")
    w("### Per-Question Retrieval Hit Comparison")
    w("")
    w("| Q# | Category | Vector Only | Hybrid+Reranker |")
    w("| --- | --- | --- | --- |")
    l1_map = {r["id"]: r for r in all_retrieval[1]}
    l2_map = {r["id"]: r for r in all_retrieval[2]}
    improved = 0
    degraded = 0
    for q in EVAL_QUESTIONS:
        qid = q["id"]
        r1 = l1_map.get(qid, {})
        r2 = l2_map.get(qid, {})
        h1 = "HIT" if r1.get("retrieval_hit") else "MISS"
        h2 = "HIT" if r2.get("retrieval_hit") else "MISS"
        marker = ""
        if h1 == "MISS" and h2 == "HIT":
            marker = " (+)"
            improved += 1
        elif h1 == "HIT" and h2 == "MISS":
            marker = " (-)"
            degraded += 1
        w(f"| {qid} | {q['category']} | {h1} | {h2}{marker} |")
    w("")
    w(f"**Improvement summary:** {improved} questions improved by Hybrid+Reranker, {degraded} degraded.")
    w("")

    # Category Breakdown
    w("## Performance by Question Category")
    w("")
    categories = ["Simple Factual", "Applied", "Higher-Order Reasoning"]
    w("| Category | L1 Hit Rate | L2 Hit Rate | L1 Avg Latency | L2 Avg Latency |")
    w("| --- | --- | --- | --- | --- |")
    for cat in categories:
        l1_cat = [r for r in all_retrieval[1] if r["category"] == cat and r.get("error") is None]
        l2_cat = [r for r in all_retrieval[2] if r["category"] == cat and r.get("error") is None]
        n1 = len(l1_cat) or 1
        n2 = len(l2_cat) or 1
        h1 = sum(1 for r in l1_cat if r["retrieval_hit"]) / n1 * 100
        h2 = sum(1 for r in l2_cat if r["retrieval_hit"]) / n2 * 100
        lat1 = sum(r["latency"] for r in l1_cat) / n1
        lat2 = sum(r["latency"] for r in l2_cat) / n2
        w(f"| {cat} | {h1:.0f}% | {h2:.0f}% | {lat1:.2f}s | {lat2:.2f}s |")
    w("")

    # Best 5 (Level 2)
    w("## Best Performing Questions")
    w("")
    l2_valid = [r for r in all_retrieval[2] if r.get("error") is None and r.get("retrieval_hit")]
    l2_sorted = sorted(l2_valid, key=lambda r: r["scores"][0]["hybrid"] if r.get("scores") else 0, reverse=True)
    for rank, r in enumerate(l2_sorted[:5], 1):
        w(f"### {rank}. Q{r['id']}: {r['question']}")
        w("")
        ctx = r["chunks"][0]["chunk_text"][:250] if r["chunks"] else "N/A"
        score = r["scores"][0]["hybrid"] if r.get("scores") else 0
        w(f"**Retrieved Context (top chunk):** {ctx}...")
        w("")
        w(f"**Explanation:** Retrieval Hit = True, Top Hybrid Score = {score:.4f}")
        w("")

    # Worst 5 (Level 2)
    w("## Worst Performing Questions")
    w("")
    l2_misses = [r for r in all_retrieval[2] if r.get("error") is None and not r.get("retrieval_hit")]
    l2_low = sorted(
        [r for r in all_retrieval[2] if r.get("error") is None],
        key=lambda r: r["scores"][0]["hybrid"] if r.get("scores") else 0,
    )
    worst = l2_misses[:5] if l2_misses else l2_low[:5]
    for rank, r in enumerate(worst[:5], 1):
        w(f"### {rank}. Q{r['id']}: {r['question']}")
        w("")
        ctx = r["chunks"][0]["chunk_text"][:250] if r["chunks"] else "N/A"
        w(f"**Retrieved Context (top chunk):** {ctx}...")
        w("")
        reasons = []
        if not r.get("retrieval_hit"):
            reasons.append("Retrieved chunks did not contain expected keywords")
        score = r["scores"][0]["hybrid"] if r.get("scores") else 0
        if score < 0.02:
            reasons.append(f"Very low hybrid score ({score:.4f})")
        if not reasons:
            reasons.append("Low relevance in top retrieved chunks")
        w(f"**Failure Reason:** {'; '.join(reasons)}")
        w("")

    # Sample LLM Results
    if sample_results:
        w("## Sample Full-RAG Results (LLM Generation)")
        w("")
        valid_samples = [s for s in sample_results if s.get("error") is None]
        for s in valid_samples:
            w(f"### Q{s['id']}: {s['question']}")
            w("")
            w(f"**Answer:** {s['answer'][:400]}{'...' if len(s.get('answer','')) > 400 else ''}")
            w("")
            w(f"**Confidence:** {s['confidence']}")
            w("")
            if s["citations"]:
                w("**Citations:**")
                for cit in s["citations"][:3]:
                    w(f"- {cit.get('document', 'N/A')} (Page {cit.get('page', 'N/A')}, Section: {cit.get('section', 'N/A')})")
            w("")
            w(f"**Faithfulness:** {s['faithfulness']} | **Accurate:** {s['accurate']} | "
              f"**Hallucinated:** {s['hallucinated']} | **Citation Quality:** {s['citation_quality']}")
            w("")

    # Error Analysis
    w("## Error Analysis")
    w("")

    retrieval_failures = [r for r in all_retrieval[2] if not r.get("retrieval_hit") and r.get("error") is None]
    w(f"### Retrieval Failures ({len(retrieval_failures)} / {len(EVAL_QUESTIONS)})")
    w("")
    if retrieval_failures:
        for r in retrieval_failures[:10]:
            w(f"- **Q{r['id']}:** {r['question']}")
    else:
        w("No retrieval failures detected.")
    w("")

    errors = [r for r in all_retrieval[2] if r.get("error")]
    w(f"### Pipeline Errors ({len(errors)} / {len(EVAL_QUESTIONS)})")
    w("")
    if errors:
        for r in errors[:5]:
            w(f"- **Q{r['id']}:** {r['error'][:100]}")
    else:
        w("No pipeline errors encountered.")
    w("")

    # Recommendations
    w("## Recommendations")
    w("")
    w("### Chunking")
    w("- Consider implementing **semantic chunking** (splitting on topic boundaries rather than fixed token counts) to improve retrieval precision for complex multi-topic questions.")
    w("- Increase child chunk overlap from 50 to 100 tokens for better context continuity at chunk boundaries.")
    w("")
    w("### Hybrid Retrieval")
    w("- Tune the RRF constant (currently k=60) based on evaluation results; lower values give more weight to top-ranked results.")
    w("- Consider adding **query expansion** (synonym injection for aviation acronyms like ADF, VOR, DME) to improve BM25 recall.")
    w("")
    w("### Reranking")
    w("- Evaluate larger reranker models (e.g., `BAAI/bge-reranker-v2-m3`) for improved domain-specific reranking performance.")
    w("- Consider increasing the reranker candidate pool from 20 to 30 for higher-order reasoning questions.")
    w("")
    w("### Prompt Engineering")
    w("- Add **chain-of-thought reasoning** prompts for higher-order reasoning questions (Q41-Q50) to improve multi-step answer quality.")
    w("- Implement **adaptive prompting** where the system prompt is adjusted based on the detected question complexity.")
    w("")
    w("### Metadata Filtering")
    w("- Enhance section detection during ingestion to capture more granular chapter and topic metadata from ATPL books.")
    w("- Add subject-aware retrieval boosting so that questions about meteorology preferentially search meteorology documents first.")
    w("")

    with open("report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    run_evaluation()
