import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType

load_dotenv()

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    prefer_grpc=False,
)
collection = os.getenv("QDRANT_COLLECTION_NAME", "saas_knowledge_base")

for field in ["business_id", "document_id"]:
    try:
        client.create_payload_index(
            collection_name=collection,
            field_name=field,
            field_schema="keyword",
        )
        print(f"✅ Index on '{field}' created")
    except Exception as e:
        print(f"Index '{field}' already exists or error: {e}")

print("✅ Indexes ready")