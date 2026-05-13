"""Debug retrieval scoring for a specific query."""

from memory_graph import cosine_similarity, lexical_overlap, shared_keywords, tokenize, normalize_text
from qdrant_client import QdrantClient
import numpy as np

storage_path = "qdrant_data"
client = QdrantClient(path=storage_path)

records, _ = client.scroll(
    collection_name="graph_memory_nodes",
    limit=100,
    with_payload=True,
    with_vectors=True,
)

query = "我下周要去哪？"

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
query_vector = np.asarray(model.encode(query), dtype=np.float32)
norm = np.linalg.norm(query_vector)
if norm > 0:
    query_vector = query_vector / norm

print(f"Query: {query}\n")
print(f"{'='*80}\n")

ranked = []
for record in records:
    payload = record.payload or {}
    node_embedding = np.asarray(payload.get("embedding", []), dtype=np.float32)
    
    semantic_score = cosine_similarity(query_vector, node_embedding)
    text_for_lexical = f"{payload.get('question', '')} {payload.get('answer', '')} {payload.get('summary', '')}"
    lexical_score = lexical_overlap(query, text_for_lexical)
    positive_semantic = max(semantic_score, 0.0)
    combined_score = 0.7 * positive_semantic + 0.3 * lexical_score
    shared = shared_keywords(query, f"{payload.get('question', '')} {payload.get('answer', '')}")
    
    ranked.append({
        "node_id": payload.get("id", record.id),
        "title": payload.get("title", ""),
        "question": payload.get("question", ""),
        "semantic": round(semantic_score, 4),
        "lexical": round(lexical_score, 4),
        "combined": round(combined_score, 4),
        "shared": shared,
    })

ranked.sort(key=lambda x: x["combined"], reverse=True)

print("Top 10 retrieval results:\n")
for i, item in enumerate(ranked[:10], 1):
    print(f"[{i}] {item['node_id']} - {item['title']}")
    print(f"    Question: {item['question'][:80]}")
    print(f"    semantic={item['semantic']}, lexical={item['lexical']}, combined={item['combined']}")
    print(f"    shared terms: {item['shared']}")
    print()

print("\n" + "="*80)
print("\nLooking for '杭州' related nodes:\n")
for item in ranked:
    if "杭州" in item["question"] or "杭州" in item["title"]:
        print(f"  {item['node_id']} - {item['title']}")
        print(f"    Question: {item['question']}")
        print(f"    semantic={item['semantic']}, lexical={item['lexical']}, combined={item['combined']}")
        print(f"    Rank: #{ranked.index(item) + 1}")
        print()

client.close()
