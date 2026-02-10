# Nisum Technologies HR Assistant - RAG System

A Retrieval-Augmented Generation (RAG) system that allows users to query the Nisum Technologies HR Procedure Manual using natural language. The system uses **Groq API (via OpenAI client)** for LLM chat completions, **free local sentence-transformers** for embeddings, and Pinecone as the vector database. **All services are free to use!**

## Features

- 🤖 **Intelligent Q&A**: Ask questions about Nisum's HR policies, procedures, and benefits
- 🔒 **Topic Restriction**: Only answers questions related to Nisum Technologies
- 🎨 **Modern UI**: Beautiful, responsive chat interface
- ⚡ **Fast Responses**: Powered by Groq's high-performance LLM inference
- 📚 **RAG Architecture**: Retrieves relevant context from the HR manual before generating answers

## Tech Stack

- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS, JavaScript
- **Embeddings**: Sentence Transformers `all-MiniLM-L6-v2` (free, local)
- **LLM**: Groq API via OpenAI client (Llama 3.1 70B - free tier available)
- **Vector DB**: Pinecone
- **PDF Processing**: PyPDF2

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the project root. You can use `env_template.txt` as a reference:

```bash
# On Windows (PowerShell)
Copy-Item env_template.txt .env

# On Linux/Mac
cp env_template.txt .env
```

Edit `.env` and add your actual API keys:
- `GROQ_API_KEY`: Your Groq API key (for LLM chat completions)
- `PINECONE_API_KEY`: Your Pinecone API key
- `PINECONE_INDEX_NAME`: Name for your Pinecone index (default: `nisum-hr-manual`)
- `PINECONE_REGION`: Pinecone region (default: `us-east-1`)

**Note**: Embeddings use a free local model (sentence-transformers), so no API key needed for embeddings!

### 3. Get API Keys

- **Groq**: Sign up at [console.groq.com](https://console.groq.com) - Free tier available!
- **Pinecone**: Sign up at [pinecone.io](https://pinecone.io) - Free tier available!

### 4. Run the Application

```bash
python app.py
```

The application will start on `http://localhost:5000`

### 5. Initialize the System

1. Open your browser and navigate to `http://localhost:5000`
2. Click the "Initialize System (Process PDF)" button
3. Wait for the PDF to be processed and embeddings uploaded to Pinecone
4. Start asking questions!

## Usage

1. **Initialize**: Click the initialization button to process the PDF and create embeddings
2. **Ask Questions**: Type your question in the input field and press Enter or click Send
3. **Get Answers**: The system will retrieve relevant context from the HR manual and generate accurate answers

### Example Questions

- "What is Nisum's leave policy?"
- "How does the recruitment process work?"
- "What are the working hours at Nisum?"
- "Tell me about employee benefits"
- "What is the dress code policy?"

## Project Structure

```
.
├── app.py                          # Flask backend application
├── static/
│   └── index.html                # Frontend HTML page
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variables template
├── .env                            # Your actual environment variables (not in git)
├── README.md                       # This file
└── Nisum Technologies - HR Procedure Manual.pdf  # Source PDF
```

## How It Works

1. **PDF Processing**: The system extracts text from the PDF and splits it into chunks
2. **Embedding Creation**: Each chunk is converted to a vector embedding using OpenAI
3. **Vector Storage**: Embeddings are stored in Pinecone for fast similarity search
4. **Query Processing**: When a user asks a question:
   - The question is converted to an embedding
   - Similar chunks are retrieved from Pinecone
   - Relevant context is passed to Groq along with the question
   - Groq generates an answer based on the context
5. **Response Filtering**: The system ensures only Nisum-related questions are answered

## Security Notes

- Never commit your `.env` file to version control
- Keep your API keys secure and don't share them
- The system includes topic filtering to prevent answering non-Nisum questions

## Troubleshooting

### Index Already Exists
If you get an error about the index already existing, you can either:
- Use a different index name in `.env`
- Delete the existing index from Pinecone console

### PDF Not Found
Make sure `Nisum Technologies - HR Procedure Manual.pdf` is in the project root directory.

### API Errors
- Verify your API keys are correct in `.env`
- Check your API quotas and limits
- Ensure you have sufficient credits

## License

This project is for educational purposes.

## Authors

- Subhan Khurshid (21K-3096)
- Ruhama Umer Khan (21k-3097)
- Muhammad Taha Rahat (21k-3114)
- Aroon Kumar (21k-4707)
- Laiba Ali (21k-3068)
