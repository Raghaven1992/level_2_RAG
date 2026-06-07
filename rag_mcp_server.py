import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from mcp.server.fastmcp import FastMCP
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
OLLAMA_BASE_URL = "http://127.0.0.1:11434"

# --- Initialize models once at startup ---
embeddings = OllamaEmbeddings(model="embeddinggemma", base_url=OLLAMA_BASE_URL)
llm = OllamaLLM(model="llama3.2", base_url=OLLAMA_BASE_URL, temperature=0.1)
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
all_data = db.get(include=['documents', 'metadatas'])
docs = [Document(page_content=d, metadata=m) 
        for d, m in zip(all_data['documents'], all_data['metadatas'])]

bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 8
vector_retriever = db.as_retriever(search_kwargs={"k": 8})

# --- Create MCP server (host/port set here, not in run()) ---
mcp = FastMCP("3GPP RAG Server", host="127.0.0.1", port=8000)

PROMPT_TEMPLATE = """
### [SYSTEM INSTRUCTION]
You are a professional technical 3GPP Packet core expert and trainer.
If the answer is not in the context, say: "I am sorry, the documentation does not contain this information."

### [CONTEXT]
{context}

### [USER QUESTION]
{question}

### [RESPONSE]
"""

@mcp.tool()
def query_3gpp_docs(question: str) -> str:
    """Query the 3GPP documentation knowledge base and return an expert answer."""
    
    # Hybrid retrieval
    bm25_results = bm25_retriever.invoke(question)
    vector_results = vector_retriever.invoke(question)
    results = bm25_results + vector_results

    # Deduplicate
    seen = set()
    unique_results = []
    for doc in results:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_results.append(doc)

    # Rerank
    pairs = [(question, doc.page_content) for doc in unique_results]
    scores = cross_encoder.predict(pairs)
    scored = sorted(zip(unique_results, scores), key=lambda x: x[1], reverse=True)
    top_results = [doc for doc, score in scored[:5]]

    # Generate
    context_text = "\n\n---\n\n".join([doc.page_content for doc in top_results])
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    formatted = prompt.format(context=context_text, question=question)
    response = llm.invoke(formatted)

    sources = list(set([doc.metadata.get('source', 'unknown') for doc in top_results]))
    return f"{response}\n\nSources: {sources}"


@mcp.tool()
def list_loaded_documents() -> str:
    """List all documents currently loaded in the knowledge base."""
    sources = set()
    for doc in docs:
        src = doc.metadata.get('source', 'unknown')
        sources.add(os.path.basename(src))
    return f"Loaded documents: {list(sources)}"


if __name__ == "__main__":
    # Run as HTTP server so browser-based clients (Streamlit) can connect via MCP
    print("🚀 Starting MCP server on http://127.0.0.1:8000/mcp")
    mcp.run(transport="streamable-http")
