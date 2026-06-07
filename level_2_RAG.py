import os
import shutil
import warnings
import time
import pickle
from collections import deque

# Suppress irrelevant warnings for a cleaner terminal UI
warnings.filterwarnings("ignore", category=UserWarning)

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder

# --- DYNAMIC CONFIGURATION ---
# Detects the script location to ensure it runs on any machine (Portable)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
OLLAMA_BASE_URL = "http://127.0.0.1:11434"

def run_rag_system():
    # 1. INITIALIZATION
    print("--- 🤖 Initializing AI Models (llama3.2 & embeddinggemma) ---")
    embeddings = OllamaEmbeddings(model="embeddinggemma", base_url=OLLAMA_BASE_URL)
    llm = OllamaLLM(
        model="llama3.2", 
        base_url=OLLAMA_BASE_URL, 
        temperature=0.1  # Low temp for factual accuracy
    )

    # Initialize reranker
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    # Safety Check: Create data folder if missing
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"📁 Created 'data' folder. Place your PDFs in: {DATA_PATH}")
        return

    # 2. SELECTION MENU
    print("\n[RAG MODE SELECTION]")
    print("(1) Ingest: Wipe old data and process new PDFs")
    print("(2) Chat:  Ask questions using existing database")
    choice = input("Enter selection [1/2]: ")

    if choice == "1":
        # WIPE OLD DATA
        if os.path.exists(CHROMA_PATH):
            print(f"🧹 Clearing existing vector database...")
            shutil.rmtree(CHROMA_PATH)

        # LOAD AND CHUNK
        print(f"📂 Loading documents from: {DATA_PATH}")
        loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
        docs = loader.load()
        
        if not docs:
            print("❌ No PDF files found! Please add documents to the /data folder.")
            return

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
        chunking_start = time.perf_counter()
        chunks = text_splitter.split_documents(docs)
        chunking_time = time.perf_counter() - chunking_start
        print(f"✂️  Created {len(chunks)} text chunks.")
        print(f"Total time taken for chunking: {chunking_time:.2f} seconds")

        # CREATE VECTOR STORE (with progress percentage)
        print("🔢 Indexing vectors to ChromaDB (This may take a minute)...")
        indexing_start = time.perf_counter()

        BATCH_SIZE = 50
        total_chunks = len(chunks)
        db = None

        for i in range(0, total_chunks, BATCH_SIZE):
            batch = chunks[i : i + BATCH_SIZE]
            if db is None:
                db = Chroma.from_documents(batch, embeddings, persist_directory=CHROMA_PATH)
            else:
                db.add_documents(batch)

            indexed_so_far = min(i + BATCH_SIZE, total_chunks)
            percent = (indexed_so_far / total_chunks) * 100
            print(f"  ⏳ Indexing... {indexed_so_far}/{total_chunks} chunks ({percent:.1f}%)", end="\r")

        print()  # newline after the progress line
        indexing_time = time.perf_counter() - indexing_start
        print("✅ Indexing complete! Your knowledge base is ready.")
        print(f"Total time taken for indexing vectors to ChromaDB: {indexing_time:.2f} seconds")

        # Prepare retrievers after ingestion so the chat loop works immediately
        all_data = db.get(include=['documents', 'metadatas'])
        documents = all_data['documents']
        metadatas = all_data['metadatas']
        docs = [Document(page_content=doc, metadata=meta) for doc, meta in zip(documents, metadatas)]
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = 8
        vector_retriever = db.as_retriever(search_kwargs={"k": 8})

    else:
        # LOAD EXISTING DATABASE
        if not os.path.exists(CHROMA_PATH):
            print("❌ Database not found. Please run Option 1 first.")
            return
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
        print("✅ Successfully loaded the existing knowledge base.")

        # Prepare retrievers for hybrid search
        all_data = db.get(include=['documents', 'metadatas'])
        documents = all_data['documents']
        metadatas = all_data['metadatas']
        docs = [Document(page_content=doc, metadata=meta) for doc, meta in zip(documents, metadatas)]
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = 8
        vector_retriever = db.as_retriever(search_kwargs={"k":8})

    # 3. THE INTERACTIVE CHAT LOOP
    print("\n--- 🧠 AI Assistant Ready! (Type 'quit' to exit) ---")

    # --- MEMORY & CACHE SETUP ---
    query_cache = {}                        # exact-match cache: query -> response
    conversation_history = deque(maxlen=6)  # keeps last 3 Q&A pairs (6 entries)

    template_string = """
    ### [SYSTEM INSTRUCTION]
    You are a professional technical 3GPP Packet core expert and trainer. Explain the concepts clearly and in detail 
    If the answer is not in the context, strictly say: "I am sorry, but the provided documentation does not contain this information."

    ### [CONVERSATION HISTORY]
    {history}

    ### [CONTEXT]
    {context}
    
    ### [USER QUESTION]
    {question}
    
    ### [RESPONSE]
    """
    prompt_template = ChatPromptTemplate.from_template(template_string)

    while True:
        query_text = input("\nYour Question: ")
        if query_text.lower() == "quit":
            print("Shutting down...")
            break

        # --- LATENCY TRACKING START ---
        start_total = time.perf_counter()

        # CHECK CACHE FIRST
        cache_key = query_text.strip().lower()
        if cache_key in query_cache:
            cached = query_cache[cache_key]
            print(f"\nResponse: {cached['response']}")
            print(f"\nSources: {cached['sources']}")
            print("-" * 30)
            print(f"⚡ CACHE HIT — answered instantly (saved ~{cached['last_latency']:.1f}s)")
            print("-" * 30)
            continue

        # STAGE 1: RETRIEVAL
        start_retrieval = time.perf_counter()
        # Hybrid search
        bm25_results = bm25_retriever.invoke(query_text)
        vector_results = vector_retriever.invoke(query_text)
        results = bm25_results + vector_results
        # Deduplicate
        seen = set()
        unique_results = []
        for doc in results:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique_results.append(doc)
        results = unique_results
        # Reranking
        pairs = [(query_text, doc.page_content) for doc in results]
        scores = cross_encoder.predict(pairs)
        scored_results = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
        top_results = [doc for doc, score in scored_results[:5]]
        retrieval_time = time.perf_counter() - start_retrieval

        # STAGE 2: GENERATION
        context_text = "\n\n---\n\n".join([doc.page_content for doc in top_results])

        # Build conversation history string
        history_text = ""
        if conversation_history:
            history_lines = []
            for entry in conversation_history:
                history_lines.append(entry)
            history_text = "\n".join(history_lines)
        else:
            history_text = "No previous conversation."

        formatted_prompt = prompt_template.format(
            history=history_text,
            context=context_text,
            question=query_text
        )

        print("AI is thinking...")
        start_gen = time.perf_counter()
        response = llm.invoke(formatted_prompt)
        generation_time = time.perf_counter() - start_gen

        # Self-Correction: Evaluate relevance
        correction_prompt = f"""
        Evaluate if the following response is directly based on the provided context and answers the question. If the response says it does not contain the information or is not relevant, or if it's not specific, respond with 'REFINE'. Otherwise, respond with 'OK'.

        Question: {query_text}
        Context: {context_text}
        Response: {response}

        Evaluation (REFINE or OK):
        """
        correction_response = llm.invoke(correction_prompt).strip()
        if "REFINE" in correction_response.upper():
            print("🔄 Refining retrieval for better context...")
            # Retrieve more chunks
            top_results = [doc for doc, score in scored_results[:10]]
            context_text = "\n\n---\n\n".join([doc.page_content for doc in top_results])
            formatted_prompt = prompt_template.format(
                history=history_text,
                context=context_text,
                question=query_text
            )
            response = llm.invoke(formatted_prompt)

        total_latency = time.perf_counter() - start_total
        sources = list(set([doc.metadata.get('source') for doc in top_results]))

        # --- OUTPUT ---
        print(f"\nResponse: {response}")
        print(f"\nSources: {sources}")
        
        print("-" * 30)
        print(f"⏱️  LATENCY REPORT:")
        print(f"  └─ Search:     {retrieval_time:.3f}s")
        print(f"  └─ Generation: {generation_time:.3f}s")
        print(f"  └─ TOTAL:      {total_latency:.3f}s")
        print("-" * 30)

        # Store in cache and update conversation history
        query_cache[cache_key] = {
            "response": response,
            "sources": sources,
            "last_latency": total_latency
        }
        conversation_history.append(f"User: {query_text}")
        conversation_history.append(f"Assistant: {response}")

if __name__ == "__main__":
    run_rag_system()
