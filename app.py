import os
import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
import openai
import pinecone
from pinecone import Pinecone, ServerlessSpec
from PyPDF2 import PdfReader
import hashlib
from typing import List, Dict
from sentence_transformers import SentenceTransformer


load_dotenv()
# python environment flask 
app = Flask(__name__, static_folder='static')
CORS(app)

# Initialize OpenAI client with Groq API (compatible with OpenAI client)
# Using Groq's API key and base URL for LLM
groq_api_key = os.getenv("GROQ_API_KEY")
client = openai.OpenAI(
    api_key=groq_api_key,
    base_url="https://api.groq.com/openai/v1"
)

print("Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  
EMBEDDING_DIMENSION = 384
print("Embedding model loaded!")
# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME", "nisum-hr-manual")
#pinecone is vector database

if index_name not in [idx.name for idx in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=EMBEDDING_DIMENSION,  # all-MiniLM-L6-v2 dimension
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region=os.getenv("PINECONE_REGION", "us-east-1")
        )
    )

pinecone_index = pc.Index(index_name)


SYSTEM_PROMPT = """You are a helpful assistant for Nisum Technologies, Inc. You can ONLY answer questions about:
- Nisum Technologies company policies, procedures, and HR manual
- Employee benefits, leave policies, and work arrangements at Nisum
- Nisum's code of conduct, recruitment, and employment conditions
- Any information contained in the Nisum HR Procedure Manual

If asked about anything unrelated to Nisum Technologies, politely decline and redirect the conversation back to Nisum-related topics. Always be professional, accurate, and helpful when answering questions about Nisum."""

def extract_text_from_pdf(pdf_path: str) -> List[str]:
    """Extract text from PDF and split into chunks."""
    reader = PdfReader(pdf_path)
    text_chunks = []
    
    for page in reader.pages:
        text = page.extract_text()
        if text.strip():

            chunk_size = 1000
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i + chunk_size]
                if chunk.strip():
                    text_chunks.append(chunk)
    
    return text_chunks

def create_embeddings(texts: List[str]) -> List[List[float]]:
    """Create embeddings using free local sentence-transformers model."""
    embeddings = embedding_model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return [embedding.tolist() for embedding in embeddings]

def upload_to_pinecone(chunks: List[str], embeddings: List[List[float]]):
    """Upload chunks and embeddings to Pinecone."""
    vectors = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        chunk_id = hashlib.md5(chunk.encode()).hexdigest()
        vectors.append({
            "id": chunk_id,
            "values": embedding,
            "metadata": {"text": chunk}
        })
    
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        pinecone_index.upsert(vectors=batch)

def query_pinecone(query_embedding: List[float], top_k: int = 5) -> List[Dict]:
    """Query Pinecone for similar chunks."""
    results = pinecone_index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    return results.matches

@app.route('/')
def index():
    """Serve the frontend HTML page."""
    return send_from_directory('static', 'index.html')

@app.route('/api/initialize', methods=['POST'])
def initialize():
    """Initialize the RAG system by processing the PDF."""
    try:
        pdf_path = "Nisum Technologies - HR Procedure Manual Updated.pdf"
        
        if not os.path.exists(pdf_path):
            return jsonify({"error": "PDF file not found"}), 404
        
        # Extract text from PDF
        chunks = extract_text_from_pdf(pdf_path)
        
        # Create embeddings
        embeddings = create_embeddings(chunks)
        
        # Upload to Pinecone
        upload_to_pinecone(chunks, embeddings)
        
        return jsonify({
            "message": f"Successfully processed {len(chunks)} chunks and uploaded to Pinecone",
            "chunks_count": len(chunks)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat queries."""
    try:
        data = request.json
        user_query = data.get("message", "").strip()
        
        if not user_query:
            return jsonify({"error": "Message is required"}), 400
        
        nisum_keywords = ["nisum", "company", "policy", "employee", "hr", "leave", "benefit", 
                         "work", "recruitment", "procedure", "manual", "employment"]
        query_lower = user_query.lower()
        
        is_about_nisum = any(keyword in query_lower for keyword in nisum_keywords) or \
                        "what" in query_lower or "how" in query_lower or "tell" in query_lower or \
                        "explain" in query_lower or "describe" in query_lower
        
        if not is_about_nisum:
            return jsonify({
                "response": "I can only answer questions about Nisum Technologies, its policies, procedures, and HR manual. Please ask me something related to Nisum Technologies."
            })
        
        query_embedding = embedding_model.encode(user_query, convert_to_numpy=True).tolist()
        
        # Query Pinecone for relevant chunks
        results = query_pinecone(query_embedding, top_k=5)
        
        # Build context from retrieved chunks
        context = "\n\n".join([match.metadata["text"] for match in results])
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context from Nisum HR Manual:\n\n{context}\n\n\nUser Question: {user_query}\n\nPlease answer based on the provided context. If the context doesn't contain relevant information, say so politely."}
        ]
        
        # Get response from Groq using OpenAI client
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="meta-llama/llama-4-scout-17b-16e-instruct",  
            temperature=0.7,
            max_tokens=1000
        )
        
        response = chat_completion.choices[0].message.content
        
        
        if len(response) < 50 and "nisum" not in response.lower():
            response = "I can only provide information about Nisum Technologies based on the HR Procedure Manual. Could you please rephrase your question to be more specific about Nisum's policies or procedures?"
        
        return jsonify({"response": response})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "index": index_name})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
