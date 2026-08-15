import os
import uuid
import pytest
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

load_dotenv()

qdrant = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"), prefer_grpc=False)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
COLLECTION = os.getenv("QDRANT_COLLECTION_NAME", "saas_knowledge_base")

def test_tenant_isolation_retrieval():
    vector = model.encode("This is a secret for tenant A", convert_to_numpy=True).tolist()
    point_id = str(uuid.uuid4())
    qdrant.upsert(
        collection_name=COLLECTION,
        points=[{
            "id": point_id,
            "vector": vector,
            "payload": {
                "business_id": "tenant-a",
                "text": "This is a secret for tenant A",
                "document_id": "doc-a",
            },
        }],
    )

    query_vector = model.encode("secret", convert_to_numpy=True).tolist()
    results = qdrant.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        query_filter=Filter(
            must=[FieldCondition(key="business_id", match=MatchValue(value="tenant-b"))]
        ),
        limit=5,
    ).points

    for p in results:
        assert p.id != point_id

    # Cleanup
    qdrant.delete(
        collection_name=COLLECTION,
        points_selector=Filter(
            must=[FieldCondition(key="business_id", match=MatchValue(value="tenant-a"))]
        ),
    )