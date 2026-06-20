# AIRMAN RAG Evaluation Report

**Generated:** 2026-06-20 15:37:51

---

## Dataset

| Parameter | Value |
| --- | --- |
| Number of Documents | 7 |
| Number of Chunks | 13222 |
| Chunking Strategy | Parent-Child (Parent: 1500 tokens, Child: 400 tokens, Overlap: 50 tokens) |
| Embedding Model | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Database | FAISS (IndexFlatIP) |
| LLM Model | gemini-2.5-flash |
| Retrieval Strategy | Hybrid Search (FAISS + BM25 + RRF) + CrossEncoder Reranker |
| Reranker Model | BAAI/bge-reranker-base |
| Evaluation Questions | 50 |

## Quantitative Metrics

**Definitions:**
- **Retrieval Hit Rate:** Did the retrieved chunks actually contain the answer?
- **Faithfulness:** Is the answer fully grounded in retrieved text?
- **Hallucination Rate:** Any unsupported claims count as hallucination.

| Metric | Score |
| --- | --- |
| Retrieval Hit Rate | 98.0% |
| Faithfulness | 0.92 / 1.0 |
| Hallucination Rate | 8.0% |
| Avg Retrieval Latency | 12.45s |
| Avg Top-1 Hybrid Score | 0.0167 |

## Level 1 vs Level 2 Comparison

| Metric | Vector Only | Hybrid + Reranker |
| --- | --- | --- |
| Retrieval Hit Rate | 100.0% | 98.0% |
| Avg Retrieval Latency | 0.12s | 12.45s |
| Avg Top-1 Score | 0.6482 | 0.0167 |

### Per-Question Retrieval Hit Comparison

| Q# | Category | Vector Only | Hybrid+Reranker |
| --- | --- | --- | --- |
| 1 | Simple Factual | HIT | HIT |
| 2 | Simple Factual | HIT | HIT |
| 3 | Simple Factual | HIT | HIT |
| 4 | Simple Factual | HIT | HIT |
| 5 | Simple Factual | HIT | HIT |
| 6 | Simple Factual | HIT | HIT |
| 7 | Simple Factual | HIT | HIT |
| 8 | Simple Factual | HIT | HIT |
| 9 | Simple Factual | HIT | HIT |
| 10 | Simple Factual | HIT | HIT |
| 11 | Simple Factual | HIT | HIT |
| 12 | Simple Factual | HIT | HIT |
| 13 | Simple Factual | HIT | HIT |
| 14 | Simple Factual | HIT | HIT |
| 15 | Simple Factual | HIT | HIT |
| 16 | Simple Factual | HIT | HIT |
| 17 | Simple Factual | HIT | HIT |
| 18 | Simple Factual | HIT | HIT |
| 19 | Simple Factual | HIT | HIT |
| 20 | Simple Factual | HIT | HIT |
| 21 | Applied | HIT | HIT |
| 22 | Applied | HIT | HIT |
| 23 | Applied | HIT | HIT |
| 24 | Applied | HIT | HIT |
| 25 | Applied | HIT | HIT |
| 26 | Applied | HIT | HIT |
| 27 | Applied | HIT | HIT |
| 28 | Applied | HIT | HIT |
| 29 | Applied | HIT | HIT |
| 30 | Applied | HIT | HIT |
| 31 | Applied | HIT | HIT |
| 32 | Applied | HIT | HIT |
| 33 | Applied | HIT | HIT |
| 34 | Applied | HIT | HIT |
| 35 | Applied | HIT | HIT |
| 36 | Applied | HIT | HIT |
| 37 | Applied | HIT | HIT |
| 38 | Applied | HIT | HIT |
| 39 | Applied | HIT | HIT |
| 40 | Applied | HIT | HIT |
| 41 | Higher-Order Reasoning | HIT | HIT |
| 42 | Higher-Order Reasoning | HIT | HIT |
| 43 | Higher-Order Reasoning | HIT | HIT |
| 44 | Higher-Order Reasoning | HIT | HIT |
| 45 | Higher-Order Reasoning | HIT | HIT |
| 46 | Higher-Order Reasoning | HIT | HIT |
| 47 | Higher-Order Reasoning | HIT | HIT |
| 48 | Higher-Order Reasoning | HIT | HIT |
| 49 | Higher-Order Reasoning | HIT | HIT |
| 50 | Higher-Order Reasoning | HIT | MISS (-) |

**Improvement summary:** 0 questions improved by Hybrid+Reranker, 1 degraded.

## Performance by Question Category

| Category | L1 Hit Rate | L2 Hit Rate | L1 Avg Latency | L2 Avg Latency |
| --- | --- | --- | --- | --- |
| Simple Factual | 100% | 100% | 0.28s | 12.25s |
| Applied | 100% | 100% | 0.02s | 12.70s |
| Higher-Order Reasoning | 100% | 90% | 0.02s | 12.35s |

## Best Performing Questions

### 1. Q1: What is the definition of meteorology?

**Retrieved Context (top chunk):** 1
3

The Atmosphere
1
The Atmosphere
A Definition of Meteorology
“The branch of science dealing with the earth’s atmosphere and the physical processes
occurring in it.”
Reasons for Studying Meteorology
• To understand the physical processes in the at...

**Explanation:** Retrieval Hit = True, Top Hybrid Score = 0.0173

### 2. Q8: What does DME measure?

**Retrieved Context (top chunk):** 15
243
Distance Measuring Equipment (DME)
15

Distance Measuring Equipment (DME)
Introduction
Distance Measuring Equipment (DME) is a secondary radar system that enables an aircraft to
establish its range from a ground station. A pilot obtains accura...

**Explanation:** Retrieval Hit = True, Top Hybrid Score = 0.0173

### 3. Q12: What is the 1-in-60 rule used for?

**Retrieved Context (top chunk):** The 1 in 60 Rule
10
The 1 in 60 Rule
The 1 in 60 Rule
When you are flying, one or both hands will be on the control yoke. The other may have to be
on the throttle some of the time. Either way, this does not leave you many spare hands to use
to measur...

**Explanation:** Retrieval Hit = True, Top Hybrid Score = 0.0173

### 4. Q14: What does NOTAM stand for?

**Retrieved Context (top chunk):** GENERAL (GEN) volume.
NOTAM. A notice distributed by means of telecommunications containing information
concerning the establishment, condition or change in any aeronautical facility, service,
procedure or hazard, the timely knowledge of which is ess...

**Explanation:** Retrieval Hit = True, Top Hybrid Score = 0.0173

### 5. Q15: What is the purpose of a Point of Equal Time (PET)?

**Retrieved Context (top chunk):** 13
243
Point of Equal Time (PET)
13
Point of Equal Time (PET)
Introduction
Figure 13.1 All-engine Point of Equal Time (critical point)
The Point of Equal Time (PET), or sometimes referred to as Critical Point (CP) or Equal Time
Point (ETP), is that t...

**Explanation:** Retrieval Hit = True, Top Hybrid Score = 0.0173

## Worst Performing Questions

### 1. Q50: A flight encounters unexpected weather, navigation equipment degradation, and fuel limitations simultaneously. What factors should be prioritized to ensure a safe operational decision?

**Retrieved Context (top chunk):** aeroplane is forced to descend, terrain, such as mountains, may present a flight hazard.
When assessing the terrain hazard a safety margin must be introduced. When planning routes
and planning the flight profile, it is not the gross flight profile th...

**Failure Reason:** Retrieved chunks did not contain expected keywords; Very low hybrid score (0.0173)

## Sample Full-RAG Results (LLM Generation)

## Error Analysis

### Retrieval Failures (1 / 50)

- **Q50:** A flight encounters unexpected weather, navigation equipment degradation, and fuel limitations simultaneously. What factors should be prioritized to ensure a safe operational decision?

### Pipeline Errors (0 / 50)

No pipeline errors encountered.

## Recommendations

### Chunking
- Consider implementing **semantic chunking** (splitting on topic boundaries rather than fixed token counts) to improve retrieval precision for complex multi-topic questions.
- Increase child chunk overlap from 50 to 100 tokens for better context continuity at chunk boundaries.

### Hybrid Retrieval
- Tune the RRF constant (currently k=60) based on evaluation results; lower values give more weight to top-ranked results.
- Consider adding **query expansion** (synonym injection for aviation acronyms like ADF, VOR, DME) to improve BM25 recall.

### Reranking
- Evaluate larger reranker models (e.g., `BAAI/bge-reranker-v2-m3`) for improved domain-specific reranking performance.
- Consider increasing the reranker candidate pool from 20 to 30 for higher-order reasoning questions.

### Prompt Engineering
- Add **chain-of-thought reasoning** prompts for higher-order reasoning questions (Q41-Q50) to improve multi-step answer quality.
- Implement **adaptive prompting** where the system prompt is adjusted based on the detected question complexity.

### Metadata Filtering
- Enhance section detection during ingestion to capture more granular chapter and topic metadata from ATPL books.
- Add subject-aware retrieval boosting so that questions about meteorology preferentially search meteorology documents first.
