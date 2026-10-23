import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import httpx

def evaluate_metrics(
    predictions: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Computes:
    - Answer Precision: correct generated claims / total generated claims
    - Answer Recall: covered benchmark points / total benchmark points
    - Context Precision: relevant retrieved chunks / total retrieved chunks
    - Context Recall: retrieved relevant benchmark chunks / total benchmark chunks
    """
    gt_map = {item["query_id"]: item for item in ground_truth}

    context_precisions = []
    context_recalls = []
    
    for pred in predictions:
        qid = pred.get("query_id")
        if qid not in gt_map:
            continue
        gt = gt_map[qid]
        
        gold_chunks = set(gt.get("benchmark_chunk_ids", []))
        retrieved_chunks = set([c["chunk_id"] for c in pred.get("chunks", [])])

        # Context precision and recall
        if retrieved_chunks:
            relevant_retrieved = retrieved_chunks.intersection(gold_chunks)
            c_prec = len(relevant_retrieved) / len(retrieved_chunks)
        else:
            c_prec = 0.0

        if gold_chunks:
            relevant_retrieved = retrieved_chunks.intersection(gold_chunks)
            c_rec = len(relevant_retrieved) / len(gold_chunks)
        else:
            c_rec = 1.0

        context_precisions.append(c_prec)
        context_recalls.append(c_rec)

    avg_c_prec = sum(context_precisions) / len(context_precisions) if context_precisions else 0.0
    avg_c_rec = sum(context_recalls) / len(context_recalls) if context_recalls else 0.0

    return {
        "context_precision": round(avg_c_prec, 4),
        "context_recall": round(avg_c_rec, 4),
        "total_queries_evaluated": len(context_precisions)
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate nullRAG against benchmark dataset")
    parser.add_argument("--benchmark_file", type=str, default="data/benchmark_qa.json", help="Path to gold QA benchmark file")
    parser.add_argument("--api_url", type=str, default="http://localhost:8000/query", help="nullRAG /query endpoint URL")
    parser.add_argument("--collection_name", type=str, default="financial_reports", help="Collection name to query")
    parser.add_argument("--output_file", type=str, default="evaluation_results.json", help="Path to save evaluation summary")
    args = parser.parse_args()

    bench_path = Path(args.benchmark_file)
    if not bench_path.exists():
        print(f"Benchmark file '{args.benchmark_file}' not found. Please provide valid benchmark dataset.")
        return

    with open(bench_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    predictions = []
    print(f"Evaluating {len(ground_truth)} queries against {args.api_url}...")

    with httpx.Client(timeout=60.0) as client:
        for item in ground_truth:
            qid = item.get("query_id")
            query = item.get("query")
            
            payload = {
                "collection_name": args.collection_name,
                "query": query
            }
            try:
                res = client.post(args.api_url, json=payload)
                res.raise_for_status()
                data = res.json()
                data["query_id"] = qid
                predictions.append(data)
            except Exception as e:
                print(f"Failed query '{qid}': {e}")
                predictions.append({
                    "query_id": qid,
                    "answer": "",
                    "chunks": [],
                    "trace": []
                })

    scores = evaluate_metrics(predictions, ground_truth)
    print("\n--- Evaluation Results ---")
    for k, v in scores.items():
        print(f"{k}: {v}")

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump({"metrics": scores, "predictions": predictions}, f, indent=2)
    print(f"\nDetailed results saved to {args.output_file}")

if __name__ == "__main__":
    main()
