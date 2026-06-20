import json
import time
import re
from typing import List, Dict, Any
from app.rag import generate_rag_response, call_llm
from app.utils import setup_logger

logger = setup_logger(__name__)

# Test set
TEST_DATA = [
    {
        "query": "What is an Automatic Direction Finder (ADF)?",
        "expected_topic": "Radio Navigation"
    },
    {
        "query": "Explain Clear Air Turbulence (CAT).",
        "expected_topic": "Meteorology"
    },
    {
        "query": "How do you calculate the Point of No Return (PNR)?",
        "expected_topic": "Flight Planning"
    },
    {
        "query": "What are the properties of a Lambert Conformal Conic chart?",
        "expected_topic": "General Navigation"
    },
    {
        "query": "Hello! Can you help me?",
        "expected_topic": "Chitchat"
    }
]

def evaluate_faithfulness(query: str, answer: str, context: str) -> float:
    if "not available" in answer.lower() or not context.strip():
        return 1.0 # Refusal is faithful
        
    prompt = (
        "You are an evaluator. Assess if the given answer is strictly faithful to the provided context.\n"
        "Return a score between 0.0 and 1.0 where 1.0 means fully supported by context, and 0.0 means completely hallucinated.\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer: {answer}\n\n"
        "Return ONLY a JSON object: {\"score\": 1.0}"
    )
    try:
        res = call_llm("You output JSON.", prompt)
        parsed = json.loads(res)
        return float(parsed.get("score", 0.0))
    except Exception as e:
        logger.error(f"Eval error: {e}")
        return 0.0

def evaluate_relevance(query: str, answer: str) -> float:
    prompt = (
        "You are an evaluator. Assess how relevant and direct the answer is to the question.\n"
        "Return a score between 0.0 and 1.0.\n"
        f"Question: {query}\nAnswer: {answer}\n\n"
        "Return ONLY a JSON object: {\"score\": 1.0}"
    )
    try:
        res = call_llm("You output JSON.", prompt)
        parsed = json.loads(res)
        return float(parsed.get("score", 0.0))
    except Exception as e:
        logger.error(f"Eval error: {e}")
        return 0.0

def run_evaluation():
    print("Starting RAG Pipeline Evaluation...")
    results = []
    
    total_faithfulness = 0.0
    total_relevance = 0.0
    latency_sum = 0.0
    
    for item in TEST_DATA:
        query = item["query"]
        print(f"\nEvaluating Query: '{query}'")
        
        start_t = time.time()
        try:
            # We ask for debug=True to get chunks
            response = generate_rag_response(query, debug=True)
            latency = time.time() - start_t
            
            ans = response.answer
            chunks = response.retrieved_chunks or []
            context_text = "\n".join([c.get("chunk_text", "") for c in chunks])
            
            faith_score = evaluate_faithfulness(query, ans, context_text)
            rel_score = evaluate_relevance(query, ans)
            
            # Special case for chitchat
            if item["expected_topic"] == "Chitchat" and not chunks:
                faith_score = 1.0
                rel_score = 1.0
                
            total_faithfulness += faith_score
            total_relevance += rel_score
            latency_sum += latency
            
            print(f"  Faithfulness: {faith_score:.2f} | Relevance: {rel_score:.2f} | Latency: {latency:.2f}s")
            
            results.append({
                "query": query,
                "answer": ans[:100] + "...",
                "faithfulness": faith_score,
                "relevance": rel_score,
                "latency": latency
            })
            
        except Exception as e:
            print(f"  FAILED: {str(e)}")
            
    num_queries = len(TEST_DATA)
    avg_faith = total_faithfulness / num_queries
    avg_rel = total_relevance / num_queries
    avg_lat = latency_sum / num_queries
    
    # Generate Markdown Report
    report = f"# RAG Pipeline Evaluation Report\n\n"
    report += f"**Total Queries Evaluated:** {num_queries}\n\n"
    report += "## Aggregate Metrics\n"
    report += f"- **Average Faithfulness (Grounding):** {avg_faith:.2f} / 1.0\n"
    report += f"- **Average Answer Relevance:** {avg_rel:.2f} / 1.0\n"
    report += f"- **Average Latency:** {avg_lat:.2f} seconds\n\n"
    
    report += "## Detailed Results\n"
    for r in results:
        report += f"**Q:** {r['query']}\n"
        report += f"**A:** {r['answer']}\n"
        report += f"> Faithfulness: {r['faithfulness']} | Relevance: {r['relevance']} | Latency: {r['latency']:.2f}s\n\n"
        
    with open("evaluation_report.md", "w") as f:
        f.write(report)
        
    print("\nEvaluation Complete! Report saved to evaluation_report.md")

if __name__ == "__main__":
    run_evaluation()
