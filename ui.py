import os
import time
import requests
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Aviation Document AI Assistant",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium CSS to inject
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2.5rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
    }
    
    .main-header h1 {
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        font-size: 1.1rem;
        font-weight: 300;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    
    /* Status indicators */
    .status-badge {
        display: inline-block;
        padding: 0.35em 0.65em;
        font-size: 0.85em;
        font-weight: 700;
        line-height: 1;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 50rem;
    }
    
    .status-online {
        background-color: rgba(40, 167, 69, 0.2);
        color: #28a745;
        border: 1px solid #28a745;
    }
    
    .status-offline {
        background-color: rgba(220, 53, 69, 0.2);
        color: #dc3545;
        border: 1px solid #dc3545;
    }
    
    /* Answer Card & Citation Cards styling */
    .answer-card {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        border-left: 5px solid #30A2FF;
        margin-top: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    
    .citation-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 0.8rem;
    }
    
    .citation-pill {
        background-color: rgba(48, 162, 255, 0.12);
        border: 1px solid rgba(48, 162, 255, 0.3);
        color: #30A2FF;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 5px;
    }
    
    .refusal-card {
        background-color: rgba(220, 53, 69, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        border-left: 5px solid #dc3545;
        margin-top: 1rem;
        margin-bottom: 1rem;
        color: #dc3545;
        font-weight: 500;
    }

    /* Retrieved Chunks styling */
    .chunk-card {
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    .score-badge {
        float: right;
        background-color: #ffd700;
        color: #111;
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# Configurations & Constants
API_URL = os.getenv(
    "API_URL",
    "http://localhost:8000"
)
UPLOAD_FOLDER = os.path.abspath("data")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Helper function to check FastAPI status
def check_api_status() -> bool:
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        if response.status_code == 200 and response.json().get("status") == "healthy":
            return True
    except Exception:
        pass
    return False

# Initialize Session State Variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# Check Backend Status
api_online = check_api_status()

# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/airplane-take-off.png", width=100)
    st.title("Aviation Control Panel")
    
    # 1. System Status
    st.subheader("System Status")
    if api_online:
        st.markdown('<span class="status-badge status-online">🟢 API Backend Online</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-offline">🔴 API Backend Offline</span>', unsafe_allow_html=True)
        st.warning("FastAPI is offline. Run `uvicorn app.api:app --reload` to start it.")
        
    st.divider()
    
    # 2. Indexed Documents Section
    st.subheader("Available Filters")
    has_ata = False
    subjects_data = []
    
    if api_online:
        try:
            res = requests.get(f"{API_URL}/filters", timeout=5)
            if res.status_code == 200:
                data = res.json()
                has_ata = data.get("has_ata", False)
                subjects_data = data.get("subjects", [])
                
                if subjects_data:
                    st.success(f"Loaded {len(subjects_data)} aviation subjects successfully!")
                else:
                    st.info("No documents are currently indexed.")
            else:
                st.error("Failed to retrieve filters.")
        except Exception as e:
            st.error(f"Connection error to filters metadata: {str(e)}")
    else:
        st.warning("Start backend to view loaded filters.")
        
    st.divider()
    
    # 3. Parameters / Debug Settings
    st.subheader("Search Scope")
    
    ata_filter = None
    subject_filter = None
    chapter_filter = None
    
    if has_ata:
        ata_filter = st.text_input("Filter by ATA Chapter", placeholder="e.g. ATA 21", help="Leave blank for no filter.")
    else:
        subject_names = ["All Subjects"] + [s["name"] for s in subjects_data]
        selected_subject_name = st.selectbox("Filter by Subject", subject_names)
        
        if selected_subject_name != "All Subjects":
            subject_filter = selected_subject_name
            # Find the chapters for the selected subject
            selected_subject_data = next((s for s in subjects_data if s["name"] == selected_subject_name), None)
            if selected_subject_data and selected_subject_data["chapters"]:
                chapter_names = ["All Chapters"] + selected_subject_data["chapters"]
                selected_chapter_name = st.selectbox("Filter by Chapter", chapter_names)
                if selected_chapter_name != "All Chapters":
                    chapter_filter = selected_chapter_name
    
    debug_mode = st.toggle("Debug Mode", value=True, help="Display retrieved source chunks, similarity scores, and LLM reasoning statistics.")
    
    st.divider()
    st.caption("Aviation Document Assistant | AIRMAN Internship Level 1&2")

# ==============================================================================
# MAIN AREA
# ==============================================================================

# Page Header
st.markdown("""
<div class="main-header">
    <h1>Aviation Document AI Assistant</h1>
    <p>Retrieval-Augmented Generation (RAG) System</p>
</div>
""", unsafe_allow_html=True)

# Check if database exists or print warning
if api_online:
    pass

# Chat History Display
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        
        # Display Confidence if present
        if msg.get("confidence") and msg["role"] == "assistant":
            st.caption(f"Confidence Level: {msg['confidence']}")
            
        # Display Citations if present
        if msg.get("citations"):
            st.markdown("**Citations:**")
            st.markdown('<div class="citation-container">', unsafe_allow_html=True)
            for cit in msg["citations"]:
                doc_name = cit.get("document", "Unknown")
                page = cit.get("page")
                ata = cit.get("ata_chapter")
                chunk_id = cit.get("chunk_id")
                
                parts = [f"📁 {doc_name}"]
                if ata: parts.append(f"🔖 {ata}")
                if page is not None: parts.append(f"📄 Pg {page}")
                if not page and chunk_id: parts.append(f"🔑 ID: {chunk_id}")
                
                pill_html = f'<div class="citation-pill">{" | ".join(parts)}</div>'
                st.markdown(pill_html, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Show snippets if no pages are available in citations
            for cit in msg["citations"]:
                if cit.get("page") is None and cit.get("snippet"):
                    with st.expander(f"Citation Snippet ({cit.get('chunk_id')})"):
                        st.caption(cit.get("snippet"))
                        
        # Display debug info (retrieved chunks)
        if debug_mode and msg.get("retrieved_chunks"):
            with st.expander("🔍 Debug Info: Retrieved Source Chunks"):
                for idx, chunk in enumerate(msg["retrieved_chunks"]):
                    h_score = chunk.get("hybrid_score", 0.0)
                    r_score = chunk.get("reranker_score", 0.0)
                    st.markdown(f"""
                    <div class="chunk-card">
                        <span class="score-badge">Hybrid: {h_score:.2f} | Rerank: {r_score:.2f}</span>
                        <strong>#{idx+1} | Source: {chunk.get('document_name')} (Page {chunk.get('page_number', 'N/A')})</strong><br>
                        <small>ATA: {chunk.get('ata_chapter', 'N/A')} | Parent ID: {chunk.get('parent_chunk_id', 'N/A')}</small>
                        <p style="margin-top: 0.5rem; font-size: 0.9rem; color: #ccc;">{chunk.get('chunk_text')}</p>
                    </div>
                    """, unsafe_allow_html=True)

# User Query Input
if prompt := st.chat_input("Ask a question based on your aviation manuals (e.g. Speed limits, emergency procedures):", disabled=not api_online):
    # Add User message to session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
        
    # Generate Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing aviation manuals and generating answer..."):
            start_time = time.time()
            payload = {
                "question": prompt,
                "ata_filter": ata_filter if ata_filter and ata_filter.strip() else None,
                "subject_filter": subject_filter,
                "chapter_filter": chapter_filter,
                "debug": True
            }
            
            try:
                res = requests.post(f"{API_URL}/ask", json=payload, timeout=60)
                if res.status_code == 200:
                    data = res.json()
                    answer = data.get("answer", "")
                    confidence = data.get("confidence", "Unknown")
                    citations = data.get("citations", [])
                    retrieved_chunks = data.get("retrieved_chunks", [])
                    
                    is_refusal = ("This information is not available" in answer or "The information is not available" in answer)
                    
                    # Display Answer
                    if is_refusal:
                        st.markdown(f'<div class="refusal-card">⚠️ {answer}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="answer-card">{answer}</div>', unsafe_allow_html=True)
                        st.caption(f"Confidence Level: {confidence}")
                        
                    # Display Citations
                    if citations and not is_refusal:
                        st.markdown("**Citations:**")
                        st.markdown('<div class="citation-container">', unsafe_allow_html=True)
                        for cit in citations:
                            doc_name = cit.get("document", "Unknown")
                            page = cit.get("page")
                            ata = cit.get("ata_chapter")
                            chunk_id = cit.get("chunk_id")
                            
                            parts = [f"📁 {doc_name}"]
                            if ata: parts.append(f"🔖 {ata}")
                            if page is not None: parts.append(f"📄 Pg {page}")
                            if not page and chunk_id: parts.append(f"🔑 ID: {chunk_id}")
                            
                            pill_html = f'<div class="citation-pill">{" | ".join(parts)}</div>'
                            st.markdown(pill_html, unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Show snippets if page is missing
                        for cit in citations:
                            if cit.get("page") is None and cit.get("snippet"):
                                with st.expander(f"Citation Snippet ({cit.get('chunk_id')})"):
                                    st.caption(cit.get("snippet"))
                                    
                    # Display retrieved chunks if debug mode is active
                    if debug_mode and retrieved_chunks:
                        with st.expander("🔍 Debug Info: Retrieved Source Chunks"):
                            for idx, chunk in enumerate(retrieved_chunks):
                                h_score = chunk.get("hybrid_score", 0.0)
                                r_score = chunk.get("reranker_score", 0.0)
                                st.markdown(f"""
                                <div class="chunk-card">
                                    <span class="score-badge">Hybrid: {h_score:.2f} | Rerank: {r_score:.2f}</span>
                                    <strong>#{idx+1} | Source: {chunk.get('document_name')} (Page {chunk.get('page_number', 'N/A')})</strong><br>
                                    <small>ATA: {chunk.get('ata_chapter', 'N/A')} | Parent ID: {chunk.get('parent_chunk_id', 'N/A')}</small>
                                    <p style="margin-top: 0.5rem; font-size: 0.9rem; color: #ccc;">{chunk.get('chunk_text')}</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                    # Save Assistant response to state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "citations": citations,
                        "retrieved_chunks": retrieved_chunks
                    })
                    
                else:
                    detail = res.json().get("detail", "Error generating response.")
                    st.error(f"Error: {detail}")
            except Exception as e:
                st.error(f"Failed to connect to RAG backend: {str(e)}")

# Bottom actions (Download Chat)
if len(st.session_state.messages) > 0:
    st.divider()
    # Format chat history for export
    chat_text = "# Aviation Assistant Conversation Log\n\n"
    for msg in st.session_state.messages:
        role_label = "🧑‍✈️ User" if msg["role"] == "user" else "🤖 Assistant"
        chat_text += f"### {role_label}\n{msg['content']}\n\n"
        if msg.get("citations"):
            chat_text += "**Citations:**\n"
            for c in msg["citations"]:
                if c.get("page") is not None:
                    chat_text += f"- Document: {c['document']} | Page: {c['page']}\n"
                else:
                    chat_text += f"- Document: {c['document']} | Chunk: {c['chunk_id']}\n"
            chat_text += "\n"
            
    st.download_button(
        label="Download Conversation Log 📥",
        data=chat_text,
        file_name="aviation_rag_conversation.md",
        mime="text/markdown"
    )
