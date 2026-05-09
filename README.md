Local-Doc: Private RAG Assistant
A local, privacy-first Retrieval-Augmented Generation (RAG) system built to chat with PDF documents. This project uses Gemma 3:4B and Ollama to ensure that no data ever leaves your local machine.

🌟 Key Features
100% Private: Runs entirely on your local hardware using Ollama.

No API Costs: Uses open-source models (Gemma 3 & Nomic Embeddings).

Factual Grounding: Uses strict system prompting to eliminate hallucinations.

Persistent Storage: Built on ChromaDB to save your document index for future use.

🛠️ Technical Stack
LLM: Google Gemma 3 (4B Parameters)

Embedding Model: Nomic-Embed-Text

Orchestration: LangChain

Vector Database: ChromaDB

Environment: Python 3.14 (Supports 3.10+)

🚀 Quick Start
1. Prerequisites
Ensure you have Ollama installed and running.

Bash
# Pull the required models
ollama pull gemma3:4b
ollama pull nomic-embed-text
2. Installation
Bash
# Clone the repository
git clone https://github.com/Raghaven1992/local_RAG.git
cd Local-RAG-Assistant

# Install dependencies
pip install langchain langchain-community langchain-ollama langchain-chroma chromadb pypdf
3. Usage
Place your PDF files in the /data folder, then run the master script:

Bash
python main.py
Select Option 1 to ingest new documents.

Select Option 2 to chat with existing indexed data.

🧠 How it Works (RAG Pipeline)
Ingestion: PDFs are loaded and cleaned using PyPDFLoader.

Chunking: Text is split into 700-character segments with a 100-character overlap to preserve semantic context.

Vectorization: Each chunk is converted into a 768-dimension vector using nomic-embed-text.

Retrieval: When a question is asked, the system performs a Similarity Search in ChromaDB to find the top 7 most relevant chunks.

Augmentation & Generation: The retrieved context is injected into a strict system prompt and processed by Gemma 3:4B to generate a grounded response.

📝 Important Terminology
Context Window: The "memory" limit of the AI. We use chunking to stay within this limit.

Temperature: Set to 0.1 to ensure factual, consistent answers rather than creative ones.

Top-K: We retrieve k=7 chunks to balance between rich context and noise reduction.

🛡️ License
Distributed under the MIT License. See LICENSE for more information.