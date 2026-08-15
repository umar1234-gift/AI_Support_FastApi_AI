"""
Test script for AI Service
Run: python test_ai_service.py
"""
import requests
import os
from dotenv import load_dotenv

# Load .env from current directory
load_dotenv()

BASE_URL = "http://localhost:8000"
INTERNAL_KEY = os.getenv("INTERNAL_AI_SERVICE_KEY", "dev-internal-key-change-me")
HEADERS = {
    "X-Internal-Key": INTERNAL_KEY,
    "Content-Type": "application/json",
}

def test_health():
    """Test health endpoint (no auth needed)"""
    print("\n🔍 Testing Health...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    print("✅ Health check passed!")

def test_unauthorized():
    """Test unauthorized access (no key provided)"""
    print("\n🔍 Testing Security...")
    response = requests.post(
        f"{BASE_URL}/internal/chat",
        headers={"Content-Type": "application/json"},  # No auth header
        json={
            "business_id": "test-biz",
            "message": "Hello",
            "assistant_name": "Test",
            "business_name": "Test Biz"
        },
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    # Should be 403 (forbidden) or 422 (missing header)
    assert response.status_code in [403, 422], f"Expected 403 or 422, got {response.status_code}"
    print("✅ Security test passed!")

def test_unauthorized_wrong_key():
    """Test with wrong API key"""
    print("\n🔍 Testing Wrong Key...")
    response = requests.post(
        f"{BASE_URL}/internal/chat",
        headers={
            "X-Internal-Key": "wrong-key-here",
            "Content-Type": "application/json"
        },
        json={
            "business_id": "test-biz",
            "message": "Hello",
            "assistant_name": "Test",
            "business_name": "Test Biz"
        },
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    assert response.status_code == 403, f"Expected 403, got {response.status_code}"
    print("✅ Wrong key test passed!")

def test_embeddings():
    """Test embedding generation"""
    print("\n🔍 Testing Embeddings...")
    data = ["Hello world", "How are you?"]
    response = requests.post(
        f"{BASE_URL}/internal/embeddings",
        headers=HEADERS,
        json=data,
    )
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Embeddings count: {result['count']}")
    print(f"Dimensions: {result['dimensions']}")
    assert response.status_code == 200
    assert len(result["embeddings"]) == 2
    print("✅ Embeddings test passed!")

def test_chat_without_context():
    """Test chat without any indexed documents"""
    print("\n🔍 Testing Chat (no context)...")
    data = {
        "business_id": "test-biz-123",
        "message": "What are your business hours?",
        "assistant_name": "Test Assistant",
        "business_name": "Test Business",
        "tone": "friendly",
        "language": "English",
        "fallback_message": "I don't have that information. Please contact the business directly."
    }
    response = requests.post(
        f"{BASE_URL}/internal/chat",
        headers=HEADERS,
        json=data,
    )
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response: {result['response'][:150]}...")
    print(f"Sources: {len(result.get('sources', []))}")
    print(f"Context Used: {result.get('context_used', False)}")
    assert response.status_code == 200
    assert "response" in result
    print("✅ Chat test passed!")

def test_document_processing():
    """Test PDF upload and processing"""
    print("\n🔍 Testing Document Processing...")
    # Create a simple test PDF if you have one
    pdf_path = "test_document.pdf"
    
    if not os.path.exists(pdf_path):
        print("⚠️  No test PDF found. Skipping document test.")
        print("   Create a test_document.pdf to test this feature.")
        return
    
    with open(pdf_path, "rb") as f:
        response = requests.post(
            f"{BASE_URL}/internal/process-document",
            headers={
                "X-Internal-Key": INTERNAL_KEY,
                "business_id": "test-biz-123"
            },
            files={"file": f}
        )
    
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Filename: {result.get('filename')}")
    print(f"Chunks: {result.get('total_chunks')}")
    assert response.status_code == 200
    assert result["status"] == "success"
    print("✅ Document processing test passed!")

if __name__ == "__main__":
    print("🧪 Running AI Service Tests...")
    print(f"Base URL: {BASE_URL}")
    print(f"Using auth key: {INTERNAL_KEY[:10]}...")
    
    try:
        test_health()
        test_unauthorized()
        test_unauthorized_wrong_key()
        test_embeddings()
        test_chat_without_context()
        test_document_processing()
        
        print("\n" + "="*50)
        print("🎉 ALL TESTS PASSED!")
        print("="*50)
    except AssertionError as e:
        print(f"\n❌ Assertion failed: {e}")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")