import streamlit as st
import asyncio
import time
import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="3GPP RAG Assistant",
    page_icon="📡",
    layout="wide"
)

MCP_SERVER_URL = "http://127.0.0.1:8000/mcp"

# MCP 1.27 has DNS rebinding protection — must send Host: 127.0.0.1
# Pass a custom httpx client with the correct Host header
def _make_http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        headers={"Host": "127.0.0.1:8000"},
        timeout=httpx.Timeout(120.0, read=120.0)
    )


# ── MCP client helpers (async) ─────────────────────────────────────────────────
async def mcp_call_tool(tool_name: str, arguments: dict) -> str:
    """Connect to MCP server, call a tool, return the text result."""
    try:
        async with _make_http_client() as http_client:
            async with streamable_http_client(MCP_SERVER_URL, http_client=http_client) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, arguments)
                    texts = [
                        block.text
                        for block in result.content
                        if hasattr(block, "text")
                    ]
                    return "\n".join(texts) if texts else "No response received."
    except Exception as e:
        err = str(e)
        if "Connect" in err or "Connection" in err or "All connection" in err:
            return "❌ Cannot connect to MCP server. Make sure `rag_mcp_server.py` is running on port 8000."
        return f"❌ Error: {err}"


async def mcp_list_tools() -> list:
    """Return list of available tools from the MCP server."""
    try:
        async with _make_http_client() as http_client:
            async with streamable_http_client(MCP_SERVER_URL, http_client=http_client) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.list_tools()
                    return [t.name for t in result.tools]
    except Exception as e:
        return [f"❌ {str(e)}"]


# ── Sync wrappers for Streamlit ────────────────────────────────────────────────
def call_tool(tool_name: str, arguments: dict) -> str:
    return asyncio.run(mcp_call_tool(tool_name, arguments))


def list_tools() -> list:
    return asyncio.run(mcp_list_tools())


def query_rag(question: str) -> str:
    return call_tool("query_3gpp_docs", {"question": question})


def get_loaded_documents() -> str:
    return call_tool("list_loaded_documents", {})


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📡 3GPP RAG Assistant")
    st.markdown("---")

    st.subheader("🔌 MCP Server Status")
    if st.button("Check Connection", use_container_width=True):
        with st.spinner("Connecting to MCP server..."):
            tools = list_tools()
            if tools and not str(tools[0]).startswith("❌"):
                st.success("Connected ✅")
                st.info(f"Available tools: {', '.join(tools)}")
            else:
                st.error(tools[0] if tools else "Connection failed")

    st.markdown("---")
    st.subheader("📂 Knowledge Base")
    if st.button("List Documents", use_container_width=True):
        with st.spinner("Fetching..."):
            docs = get_loaded_documents()
            if not docs.startswith("❌"):
                st.success(docs)
            else:
                st.error(docs)

    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption("**Flow:**")
    st.caption("Browser → Streamlit → MCP HTTP → RAG Pipeline → llama3.2")
    st.caption(f"MCP endpoint: `{MCP_SERVER_URL}`")


# ── Main chat area ─────────────────────────────────────────────────────────────
st.title("💬 3GPP Technical Assistant")
st.caption("Ask questions about your 3GPP documentation. Powered by RAG + llama3.2 via MCP.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("meta"):
            st.caption(message["meta"])

# Chat input
if prompt := st.chat_input("Ask a 3GPP question (e.g. What is the role of AMF?)"):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call MCP and show assistant response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching docs and generating answer via MCP..."):
            start = time.perf_counter()
            response = query_rag(prompt)
            elapsed = time.perf_counter() - start

        if response.startswith("❌"):
            st.error(response)
            meta = ""
        else:
            # Split answer from sources if present
            if "\n\nSources:" in response:
                answer, sources_part = response.split("\n\nSources:", 1)
                st.markdown(answer.strip())
                st.markdown(f"**📄 Sources:** `{sources_part.strip()}`")
            else:
                st.markdown(response)

            meta = f"⏱️ {elapsed:.2f}s  |  via MCP → BM25 + Vector + CrossEncoder + llama3.2"
            st.caption(meta)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "meta": meta
        })
