"""View contents of local Qdrant storage."""

from qdrant_client import QdrantClient

storage_path = "qdrant_data"
client = QdrantClient(path=storage_path)

collections = client.get_collections().collections
print(f"Found {len(collections)} collection(s):\n")

for collection in collections:
    print(f"Collection: {collection.name}")
    print()

    records, _ = client.scroll(
        collection_name=collection.name,
        limit=100,
        with_payload=True,
        with_vectors=False,
    )

    print(f"  Total records: {len(records)}\n")
    for i, record in enumerate(records, 1):
        payload = record.payload or {}
        print(f"  [{i}] ID: {record.id}")
        print(f"      Question: {payload.get('question', 'N/A')[:80]}")
        print(f"      Summary: {payload.get('summary', 'N/A')[:80]}")
        print(f"      Title: {payload.get('title', 'N/A')}")
        print(f"      Access count: {payload.get('access_count', 0)}")
        print(f"      Links: {len(payload.get('links', []))} edge(s)")
        print()
