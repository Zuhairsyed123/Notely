"""Notely Streamlit Application.

Tagline: Your notes. Smarter.
Portfolio-quality study assistant powered by local embeddings, ChromaDB, and grounded generation.
"""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from frontend.client import client
from frontend.styles import CUSTOM_CSS

# Page Configuration
st.set_page_config(
    page_title="Notely — Your notes. Smarter.",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply Custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "flashcards" not in st.session_state:
    st.session_state.flashcards = []
if "flashcard_idx" not in st.session_state:
    st.session_state.flashcard_idx = 0
if "flashcard_revealed" not in st.session_state:
    st.session_state.flashcard_revealed = False
if "flashcard_known" not in st.session_state:
    st.session_state.flashcard_known = set()
if "flashcard_needs_rev" not in st.session_state:
    st.session_state.flashcard_needs_rev = set()
if "quiz_questions" not in st.session_state:
    st.session_state.quiz_questions = []
if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False
if "quiz_user_answers" not in st.session_state:
    st.session_state.quiz_user_answers = {}
if "revision_notes" not in st.session_state:
    st.session_state.revision_notes = None


# ---------------------------------------------------------------------------
# Sidebar Navigation & System Telemetry
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 10px 0 20px 0;">
            <div style="font-size: 1.8rem; font-weight: 700; color: #1e1b4b; display: flex; align-items: center; gap: 8px;">
                <span>📚 Notely</span>
            </div>
            <div style="font-size: 0.88rem; color: #6366f1; font-weight: 600;">Your notes. Smarter.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "My Documents",
            "Summarize",
            "Chat with Notes",
            "Chunk Explorer",
            "Study Tools",
            "Settings",
        ],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Backend Connectivity & Model Status
    is_alive = client.check_backend_alive()
    if is_alive:
        st.markdown("🟢 **API Status:** Connected (`127.0.0.1:8000`)")
    else:
        st.markdown("🟡 **API Status:** Standalone Mode (Direct Service)")

    metrics = client.get_metrics()
    provider_name = metrics.get("llm_provider", "mock")
    is_mock = metrics.get("is_mock_provider", True)

    if is_mock:
        st.markdown(
            '<div class="badge-mock" style="width:100%; text-align:center;">Mode: Local Demo / Mock</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="badge-ai" style="width:100%; text-align:center;">Mode: {provider_name}</div>',
            unsafe_allow_html=True,
        )

    st.caption(f"Embeddings: `{metrics.get('embedding_model')}`")


# ---------------------------------------------------------------------------
# PAGE 1: DASHBOARD
# ---------------------------------------------------------------------------
if page == "Dashboard":
    st.markdown(
        """
        <div class="notely-header">
            <h1 class="notely-title">Study Intelligence Dashboard</h1>
        </div>
        <p class="notely-tagline">Transform lecture notes, textbooks, and research into searchable, structured mastery.</p>
        """,
        unsafe_allow_html=True,
    )

    # Top KPI Metrics
    metrics = client.get_metrics()
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-value">{metrics.get("total_documents", 0)}</div>
                <div class="stat-label">Documents Indexed</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-value">{metrics.get("total_chunks", 0)}</div>
                <div class="stat-label">Vector Chunks</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-value">{metrics.get("total_pages", 0)}</div>
                <div class="stat-label">Extracted Pages</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        storage_kb = metrics.get("total_storage_bytes", 0) / 1024
        storage_str = f"{storage_kb:.1f} KB" if storage_kb < 1024 else f"{storage_kb / 1024:.2f} MB"
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-value">{storage_str}</div>
                <div class="stat-label">Processed Content</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Charts & Quick Search
    docs = client.list_documents()

    c_left, c_right = st.columns([1, 1])

    with c_left:
        st.subheader("Document Breakdown")
        if docs:
            types = {}
            for d in docs:
                ft = d.get("file_type", "unknown").upper()
                types[ft] = types.get(ft, 0) + 1

            fig_donut = go.Figure(
                data=[
                    go.Pie(
                        labels=list(types.keys()),
                        values=list(types.values()),
                        hole=0.6,
                        marker_colors=["#6366f1", "#a855f7", "#3b82f6"],
                    )
                ]
            )
            fig_donut.update_layout(
                margin=dict(t=10, b=10, l=10, r=10),
                height=220,
                showlegend=True,
            )
            st.plotly_chart(fig_donut, use_container_width=True)
        else:
            st.info("No documents uploaded yet. Upload a PDF or paste notes in 'My Documents'.")

    with c_right:
        st.subheader("Chunks Distribution")
        if docs:
            names = [d.get("name")[:20] for d in docs[:6]]
            chunk_counts = [d.get("chunk_count", 0) for d in docs[:6]]

            fig_bar = px.bar(
                x=chunk_counts,
                y=names,
                orientation="h",
                labels={"x": "Chunks", "y": "Document"},
                color_discrete_sequence=["#4f46e5"],
            )
            fig_bar.update_layout(
                margin=dict(t=10, b=10, l=10, r=10),
                height=220,
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Vector database is empty. Ingest notes to see chunk distribution.")

    # Recent Activity
    st.subheader("Recent Document Activity")
    recent = metrics.get("recent_documents", [])
    if recent:
        table_rows = []
        for r in recent:
            table_rows.append(
                {
                    "Document Name": r.get("name"),
                    "Type": r.get("file_type").upper(),
                    "Pages": r.get("page_count"),
                    "Chunks": r.get("chunk_count"),
                    "Status": r.get("status").capitalize(),
                    "Created": r.get("created_at")[:10],
                }
            )
        st.dataframe(table_rows, use_container_width=True, hide_index=True)
    else:
        st.write("No document activity recorded yet.")


# ---------------------------------------------------------------------------
# PAGE 2: MY DOCUMENTS
# ---------------------------------------------------------------------------
elif page == "My Documents":
    st.markdown(
        """
        <div class="notely-header">
            <h1 class="notely-title">My Study Documents</h1>
        </div>
        <p class="notely-tagline">Manage lecture slides, textbook chapters, and pasted notes with page-level provenance.</p>
        """,
        unsafe_allow_html=True,
    )

    tab_upload, tab_paste = st.tabs(["📁 Upload File (PDF / TXT)", "📝 Paste Text Notes"])

    with tab_upload:
        st.markdown("Upload lecture notes, syllabi, or textbook chapters in **PDF** or **TXT** format.")

        u_col1, u_col2 = st.columns([2, 1])
        with u_col1:
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=["pdf", "txt"],
                help="Max size 25MB. PDF must contain text (scanned image PDFs will require OCR).",
            )
        with u_col2:
            st.caption("Chunking Parameters for Ingestion:")
            u_chunk_size = st.slider("Chunk Size (characters)", 100, 2000, 500, 50, key="u_cs")
            u_chunk_overlap = st.slider("Chunk Overlap (characters)", 0, 400, 100, 20, key="u_co")

        if uploaded_file is not None:
            if st.button("Ingest and Index Document", type="primary"):
                with st.spinner(f"Extracting pages and building vector chunks for '{uploaded_file.name}'..."):
                    try:
                        content_bytes = uploaded_file.read()
                        res = client.upload_file(
                            filename=uploaded_file.name,
                            content=content_bytes,
                            chunk_size=u_chunk_size,
                            chunk_overlap=u_chunk_overlap,
                        )
                        st.success(
                            f"Successfully ingested **{res.get('name')}**! Generated **{res.get('chunk_count')}** chunks across **{res.get('page_count')}** pages."
                        )
                        st.rerun()
                    except Exception as e:
                        err_msg = str(e)
                        if "scanned" in err_msg.lower() or "ocr" in err_msg.lower():
                            st.error(f"⚠️ **OCR Required:** {err_msg}")
                        else:
                            st.error(f"Ingestion failed: {err_msg}")

    with tab_paste:
        st.markdown("Directly paste quick lecture notes, professor emails, or web summaries.")
        p_title = st.text_input("Note Title", placeholder="e.g., Biology Chapter 4: Photosynthesis")
        p_text = st.text_area("Note Content", placeholder="Paste lecture notes or text here...", height=200)

        p_col1, p_col2 = st.columns([1, 1])
        with p_col1:
            p_chunk_size = st.slider("Chunk Size", 100, 2000, 500, 50, key="p_cs")
        with p_col2:
            p_chunk_overlap = st.slider("Chunk Overlap", 0, 400, 100, 20, key="p_co")

        if st.button("Save & Index Notes", type="primary"):
            if not p_title.strip() or not p_text.strip():
                st.warning("Please provide both a title and text content.")
            else:
                with st.spinner("Processing notes..."):
                    try:
                        res = client.paste_notes(
                            title=p_title,
                            text=p_text,
                            chunk_size=p_chunk_size,
                            chunk_overlap=p_chunk_overlap,
                        )
                        st.success(f"Saved notes '{res.get('name')}' with {res.get('chunk_count')} chunks.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to process pasted notes: {e}")

    st.markdown("---")
    st.subheader("Stored Documents")

    docs = client.list_documents()
    if not docs:
        st.info("No documents are currently indexed.")
    else:
        for doc in docs:
            with st.expander(f"📄 {doc.get('name')} — {doc.get('chunk_count')} chunks ({doc.get('file_type').upper()})"):
                d_col1, d_col2 = st.columns([3, 1])
                with d_col1:
                    st.write(f"**Document ID:** `{doc.get('id')}`")
                    st.write(f"**Pages:** {doc.get('page_count')} | **Size:** {doc.get('file_size_bytes')} bytes")
                    st.write(f"**Ingested:** {doc.get('created_at')[:19].replace('T', ' ')}")

                    # Fetch page preview
                    if st.button("Inspect Pages Preview", key=f"prev_{doc.get('id')}"):
                        detail = client.get_document(doc.get("id"))
                        pages = detail.get("pages", [])
                        for p in pages:
                            st.markdown(f"**Page {p.get('page_number')}** ({p.get('char_count')} chars)")
                            st.caption(p.get("preview"))

                with d_col2:
                    if st.button("🗑️ Delete Document", key=f"del_{doc.get('id')}", type="secondary"):
                        success = client.delete_document(doc.get("id"))
                        if success:
                            st.success("Document and vector records deleted.")
                            st.rerun()
                        else:
                            st.error("Failed to delete document.")


# ---------------------------------------------------------------------------
# PAGE 3: SUMMARIZE
# ---------------------------------------------------------------------------
elif page == "Summarize":
    st.markdown(
        """
        <div class="notely-header">
            <h1 class="notely-title">Document Summarization</h1>
        </div>
        <p class="notely-tagline">Generate fast extractive summaries or comprehensive AI-generated study outlines.</p>
        """,
        unsafe_allow_html=True,
    )

    docs = client.list_documents()
    if not docs:
        st.warning("Please upload at least one document in 'My Documents' to summarize.")
    else:
        doc_options = {d["id"]: f"{d['name']} ({d['file_type'].upper()}, {d['chunk_count']} chunks)" for d in docs}
        selected_doc_id = st.selectbox(
            "Select Document to Summarize",
            options=list(doc_options.keys()),
            format_func=lambda x: doc_options[x],
        )

        s_col1, s_col2 = st.columns([1, 1])
        with s_col1:
            mode_labels = {
                "extractive": "Extractive Summary (Offline, Term-Frequency Salience)",
                "ai_short": "AI Short Summary (Executive 2-3 Sentences)",
                "ai_detailed": "AI Detailed Summary (Comprehensive Outline)",
                "ai_bullet_points": "AI Key Takeaways (Bullet Points)",
                "ai_revision": "AI Exam Revision Guide (Concepts & Focus Areas)",
                "ai_chapter": "AI Chapter / Section Overview",
            }
            selected_mode = st.selectbox(
                "Summarization Style",
                options=list(mode_labels.keys()),
                format_func=lambda m: mode_labels[m],
            )

        with s_col2:
            if selected_mode == "extractive":
                max_sentences = st.slider("Max Extractive Sentences", 3, 15, 5)
            else:
                max_sentences = 5
                st.info("AI mode will synthesize structured points using the active provider.")

        if st.button("Generate Summary", type="primary"):
            with st.spinner("Generating document summary..."):
                try:
                    res = client.summarize_document(
                        document_id=selected_doc_id,
                        mode=selected_mode,
                        max_sentences=max_sentences,
                    )

                    st.markdown("---")
                    st.subheader(f"Summary for {res.get('document_name')}")

                    is_ai = res.get("is_ai_generated", False)
                    provider = res.get("provider", "local")

                    if not is_ai:
                        st.markdown(
                            f'<div class="badge-extractive">Mode: Extractive Salience ({provider})</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<div class="badge-ai">Mode: AI Generated ({provider})</div>',
                            unsafe_allow_html=True,
                        )

                    st.markdown(f"\n{res.get('summary')}")
                    st.caption(f"Summary length: {res.get('char_count')} characters")

                except Exception as e:
                    st.error(f"Summarization error: {e}")


# ---------------------------------------------------------------------------
# PAGE 4: CHAT WITH NOTES (RAG)
# ---------------------------------------------------------------------------
elif page == "Chat with Notes":
    st.markdown(
        """
        <div class="notely-header">
            <h1 class="notely-title">Chat with Notes</h1>
        </div>
        <p class="notely-tagline">Ask questions grounded strictly in your indexed material with verified source citations.</p>
        """,
        unsafe_allow_html=True,
    )

    docs = client.list_documents()

    filter_col, topk_col, clear_col = st.columns([2, 1, 1])
    with filter_col:
        doc_filter_options = {"all": "Search Across All Documents"}
        for d in docs:
            doc_filter_options[d["id"]] = f"{d['name']} ({d['file_type'].upper()})"
        selected_filter = st.selectbox(
            "Filter Material",
            options=list(doc_filter_options.keys()),
            format_func=lambda x: doc_filter_options[x],
        )
        doc_id_arg = None if selected_filter == "all" else selected_filter

    with topk_col:
        top_k = st.slider("Evidence Chunks (Top-K)", 1, 8, 4)

    with clear_col:
        st.write("")
        st.write("")
        if st.button("Clear Conversation"):
            st.session_state.chat_history = []
            st.rerun()

    # Render previous conversation
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                with st.expander(f"📑 Verified Source Citations ({len(msg['citations'])})"):
                    for c in msg["citations"]:
                        st.markdown(
                            f"""
                            <div class="citation-box">
                                <div class="citation-header">Source: {c.get('document_name')} | Page {c.get('page_number')} | Similarity: {int((c.get('relevance_score') or 0)*100)}%</div>
                                <div>"{c.get('excerpt')}"</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

    # Chat user input
    user_query = st.chat_input("Ask a question about your study materials...")
    if user_query:
        # Show user message
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate RAG response
        with st.chat_message("assistant"):
            with st.spinner("Searching indexed notes and synthesizing evidence..."):
                try:
                    res = client.chat_with_notes(
                        query=user_query,
                        document_id=doc_id_arg,
                        top_k=top_k,
                    )

                    answer = res.get("answer", "")
                    citations = res.get("citations", [])
                    sufficient = res.get("sufficient_evidence", True)
                    provider = res.get("provider", "mock")
                    is_ai = res.get("is_ai_generated", False)

                    st.markdown(answer)

                    badge_class = "badge-ai" if is_ai else "badge-mock"
                    st.markdown(
                        f'<div class="{badge_class}">Provider: {provider}</div>',
                        unsafe_allow_html=True,
                    )

                    if citations:
                        with st.expander(f"📑 Verified Source Citations ({len(citations)})", expanded=True):
                            for c in citations:
                                st.markdown(
                                    f"""
                                    <div class="citation-box">
                                        <div class="citation-header">Source: {c.get('document_name')} | Page {c.get('page_number')} | Match: {int((c.get('relevance_score') or 0)*100)}%</div>
                                        <div>"{c.get('excerpt')}"</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                    # Save to session history
                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "citations": citations,
                        }
                    )

                except Exception as e:
                    st.error(f"Error querying study materials: {e}")


# ---------------------------------------------------------------------------
# PAGE 5: CHUNK EXPLORER
# ---------------------------------------------------------------------------
elif page == "Chunk Explorer":
    st.markdown(
        """
        <div class="notely-header">
            <h1 class="notely-title">Intelligent Chunk Explorer</h1>
        </div>
        <p class="notely-tagline">Inspect how documents are split into overlapping passages with full provenance tracking.</p>
        """,
        unsafe_allow_html=True,
    )

    docs = client.list_documents()
    if not docs:
        st.warning("Please upload a document first to explore chunks.")
    else:
        doc_options = {d["id"]: f"{d['name']} ({d['chunk_count']} chunks)" for d in docs}
        selected_doc_id = st.selectbox(
            "Select Document to Inspect",
            options=list(doc_options.keys()),
            format_func=lambda x: doc_options[x],
        )

        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            exp_chunk_size = st.slider("Chunk Target Size", 100, 2000, 500, 50)
        with c2:
            exp_overlap = st.slider("Chunk Overlap", 0, 500, 100, 25)
        with c3:
            st.write("")
            st.write("")
            reindex_check = st.checkbox("Re-index & overwrite current vectors", value=False)

        action_col1, action_col2 = st.columns([1, 3])
        with action_col1:
            run_chunking = st.button("Inspect Chunks", type="primary")

        if run_chunking:
            with st.spinner("Processing chunk boundaries..."):
                try:
                    res = client.inspect_chunks(
                        document_id=selected_doc_id,
                        chunk_size=exp_chunk_size,
                        chunk_overlap=exp_overlap,
                        reindex=reindex_check,
                    )
                    chunks = res.get("chunks", [])
                    st.success(
                        f"Generated **{len(chunks)} chunks** [Target Size: {exp_chunk_size} chars | Overlap: {exp_overlap} chars]"
                    )
                    if reindex_check:
                        st.info("Re-indexing complete: new chunks updated in vector store.")

                    st.markdown("### Chunk Inspection Table")
                    filter_text = st.text_input("Filter chunks by keyword", "")

                    displayed = 0
                    for chunk in chunks:
                        if filter_text and filter_text.lower() not in chunk.get("text", "").lower():
                            continue
                        displayed += 1

                        with st.container():
                            st.markdown(
                                f"""
                                <div class="chunk-card">
                                    <div class="chunk-meta">
                                        CHUNK #{chunk.get('chunk_index')} | ID: <code>{chunk.get('chunk_id')}</code> | Page: {chunk.get('page_number')} | Length: {chunk.get('char_count')} chars
                                    </div>
                                    <div style="font-size:0.92rem; color:#1e293b; white-space:pre-wrap; background:#f8fafc; padding:10px; border-radius:6px;">{chunk.get('text')}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                        if displayed >= 20:
                            st.caption(f"Displaying first 20 of {len(chunks)} chunks.")
                            break

                except Exception as e:
                    st.error(f"Chunking inspection error: {e}")


# ---------------------------------------------------------------------------
# PAGE 6: STUDY TOOLS
# ---------------------------------------------------------------------------
elif page == "Study Tools":
    st.markdown(
        """
        <div class="notely-header">
            <h1 class="notely-title">Study Tools & Interactive Revision</h1>
        </div>
        <p class="notely-tagline">Test your knowledge with flip flashcards, MCQ quizzes, and structured revision sheets.</p>
        """,
        unsafe_allow_html=True,
    )

    docs = client.list_documents()
    study_doc_options = {"all": "All Uploaded Notes"}
    for d in docs:
        study_doc_options[d["id"]] = f"{d['name']} ({d['file_type'].upper()})"

    tab_flash, tab_quiz, tab_revision = st.tabs(["🗂️ Interactive Flashcards", "❓ MCQ Quiz", "📖 Revision Notes"])

    # --- TAB 1: FLASHCARDS ---
    with tab_flash:
        f_doc_id = st.selectbox(
            "Select Study Material for Flashcards",
            options=list(study_doc_options.keys()),
            format_func=lambda x: study_doc_options[x],
            key="flash_doc_select",
        )
        f_count = st.slider("Number of Flashcards", 3, 15, 5, key="f_count_slider")

        if st.button("Generate Flashcards", type="primary"):
            with st.spinner("Generating flashcards from indexed material..."):
                try:
                    res = client.generate_flashcards(
                        document_id=None if f_doc_id == "all" else f_doc_id,
                        count=f_count,
                    )
                    st.session_state.flashcards = res.get("flashcards", [])
                    st.session_state.flashcard_idx = 0
                    st.session_state.flashcard_revealed = False
                    st.session_state.flashcard_known = set()
                    st.session_state.flashcard_needs_rev = set()
                except Exception as e:
                    st.error(f"Flashcard generation failed: {e}")

        # Render Flashcards
        cards = st.session_state.flashcards
        if cards:
            total_cards = len(cards)
            curr_idx = st.session_state.flashcard_idx
            card = cards[curr_idx]

            st.progress((curr_idx + 1) / total_cards)
            st.caption(
                f"Card {curr_idx + 1} of {total_cards} | Known: {len(st.session_state.flashcard_known)} | Needs Revision: {len(st.session_state.flashcard_needs_rev)}"
            )

            # Flashcard Body
            if not st.session_state.flashcard_revealed:
                st.markdown(
                    f"""
                    <div class="flashcard">
                        <div class="flashcard-question">{card.get('question')}</div>
                        <div class="flashcard-source">Citation: {card.get('source_citation')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button("🔄 Flip to Reveal Answer", use_container_width=True):
                    st.session_state.flashcard_revealed = True
                    st.rerun()
            else:
                st.markdown(
                    f"""
                    <div class="flashcard">
                        <div class="flashcard-question" style="font-size:1.1rem; color:#64748b; margin-bottom:12px;">{card.get('question')}</div>
                        <div class="flashcard-answer">{card.get('answer')}</div>
                        <div class="flashcard-source">Citation: {card.get('source_citation')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button("🔄 Flip to Question", use_container_width=True):
                    st.session_state.flashcard_revealed = False
                    st.rerun()

            # Navigation & Progress Buttons
            btn_col1, btn_col2, btn_col3, btn_col4 = st.columns([1, 1, 1, 1])

            with btn_col1:
                if st.button("⬅️ Previous", disabled=(curr_idx == 0)):
                    st.session_state.flashcard_idx = max(0, curr_idx - 1)
                    st.session_state.flashcard_revealed = False
                    st.rerun()

            with btn_col2:
                if st.button("✅ Mark as Known"):
                    st.session_state.flashcard_known.add(curr_idx)
                    st.session_state.flashcard_needs_rev.discard(curr_idx)
                    if curr_idx < total_cards - 1:
                        st.session_state.flashcard_idx += 1
                        st.session_state.flashcard_revealed = False
                    st.rerun()

            with btn_col3:
                if st.button("⚠️ Needs Revision"):
                    st.session_state.flashcard_needs_rev.add(curr_idx)
                    st.session_state.flashcard_known.discard(curr_idx)
                    if curr_idx < total_cards - 1:
                        st.session_state.flashcard_idx += 1
                        st.session_state.flashcard_revealed = False
                    st.rerun()

            with btn_col4:
                if st.button("Next ➡️", disabled=(curr_idx == total_cards - 1)):
                    st.session_state.flashcard_idx = min(total_cards - 1, curr_idx + 1)
                    st.session_state.flashcard_revealed = False
                    st.rerun()

    # --- TAB 2: MCQ QUIZ ---
    with tab_quiz:
        q_doc_id = st.selectbox(
            "Select Study Material for Quiz",
            options=list(study_doc_options.keys()),
            format_func=lambda x: study_doc_options[x],
            key="quiz_doc_select",
        )
        q_count = st.slider("Number of Questions", 3, 10, 5, key="q_count_slider")

        if st.button("Generate Practice Quiz", type="primary"):
            with st.spinner("Extracting conceptual questions and plausible distractors..."):
                try:
                    res = client.generate_quiz(
                        document_id=None if q_doc_id == "all" else q_doc_id,
                        count=q_count,
                    )
                    st.session_state.quiz_questions = res.get("questions", [])
                    st.session_state.quiz_submitted = False
                    st.session_state.quiz_user_answers = {}
                except Exception as e:
                    st.error(f"Quiz generation failed: {e}")

        # Render Quiz Questions
        questions = st.session_state.quiz_questions
        if questions:
            st.markdown("---")
            for q in questions:
                qid = q.get("id")
                st.markdown(f"**Question {qid}:** {q.get('question')}")
                st.caption(f"Reference: {q.get('source_citation')}")

                selected_opt = st.radio(
                    f"Options for Q{qid}",
                    options=q.get("options", []),
                    key=f"quiz_opt_{qid}",
                    label_visibility="collapsed",
                    disabled=st.session_state.quiz_submitted,
                )
                st.session_state.quiz_user_answers[qid] = selected_opt

                if st.session_state.quiz_submitted:
                    correct = q.get("correct_answer")
                    if selected_opt == correct:
                        st.success(f"✅ Correct! {q.get('explanation')}")
                    else:
                        st.error(f"❌ Incorrect. Correct answer: **{correct}**\n\nExplanation: {q.get('explanation')}")
                st.markdown("")

            if not st.session_state.quiz_submitted:
                if st.button("Submit Quiz Answers", type="primary"):
                    st.session_state.quiz_submitted = True
                    st.rerun()
            else:
                correct_count = sum(
                    1
                    for q in questions
                    if st.session_state.quiz_user_answers.get(q.get("id")) == q.get("correct_answer")
                )
                score_pct = int((correct_count / len(questions)) * 100)
                st.markdown(
                    f"""
                    <div style="background:#ede9fe; border-radius:12px; padding:20px; text-align:center; margin-top:20px;">
                        <h2 style="color:#4338ca; margin:0;">Quiz Complete! Score: {correct_count} / {len(questions)} ({score_pct}%)</h2>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button("Retake or Reset Quiz"):
                    st.session_state.quiz_submitted = False
                    st.session_state.quiz_user_answers = {}
                    st.rerun()

    # --- TAB 3: REVISION NOTES ---
    with tab_revision:
        r_doc_id = st.selectbox(
            "Select Study Material for Revision Sheet",
            options=list(study_doc_options.keys()),
            format_func=lambda x: study_doc_options[x],
            key="rev_doc_select",
        )

        if st.button("Generate Revision Guide", type="primary"):
            with st.spinner("Extracting definitions, formulas, and potential exam topics..."):
                try:
                    res = client.generate_revision_notes(
                        document_id=None if r_doc_id == "all" else r_doc_id
                    )
                    st.session_state.revision_notes = res
                except Exception as e:
                    st.error(f"Revision sheet generation failed: {e}")

        notes = st.session_state.revision_notes
        if notes:
            st.markdown("---")
            st.markdown("### 📋 Course Revision Guide")

            # Definitions
            defs = notes.get("definitions", [])
            if defs:
                st.markdown("#### 🔑 Core Definitions & Vocabulary")
                for d in defs:
                    st.markdown(f"- **{d.get('term')}:** {d.get('definition')} *[{d.get('source')}]*")

            # Key Concepts
            concepts = notes.get("key_concepts", [])
            if concepts:
                st.markdown("#### 💡 Fundamental Concepts")
                for c in concepts:
                    st.markdown(f"**{c.get('concept')}**")
                    st.write(c.get("explanation"))
                    st.caption(f"Source: {c.get('source')}")

            # Formulas or Rules
            formulas = notes.get("formulas_or_rules", [])
            if formulas:
                st.markdown("#### 📐 Formulas, Laws & Governing Rules")
                for f in formulas:
                    st.markdown(f"- `{f}`")

            # Comparisons
            comps = notes.get("common_comparisons", [])
            if comps:
                st.markdown("#### ⚖️ Contrast & Distinctions")
                for cp in comps:
                    st.markdown(f"- {cp.get('comparison')} *[{cp.get('source')}]*")

            # Potential Exam Questions
            questions = notes.get("potential_exam_questions", [])
            if questions:
                st.markdown("#### 🎯 Likely Exam Questions")
                for q in questions:
                    st.markdown(f"- ❓ {q}")


# ---------------------------------------------------------------------------
# PAGE 7: SETTINGS
# ---------------------------------------------------------------------------
elif page == "Settings":
    st.markdown(
        """
        <div class="notely-header">
            <h1 class="notely-title">Application Settings</h1>
        </div>
        <p class="notely-tagline">Configure embedding models, generation providers, and local persistence paths.</p>
        """,
        unsafe_allow_html=True,
    )

    metrics = client.get_metrics()

    st.subheader("RAG & Chunking Parameters")
    s1, s2, s3 = st.columns(3)
    with s1:
        st.metric("Default Chunk Size", f"{metrics.get('default_chunk_size')} chars")
    with s2:
        st.metric("Default Overlap", f"{metrics.get('default_chunk_overlap')} chars")
    with s3:
        st.metric("Default Retrieval Top-K", f"{metrics.get('default_top_k')} chunks")

    st.markdown("---")
    st.subheader("Model & Generation Providers")

    col_model, col_prov = st.columns(2)
    with col_model:
        st.markdown("**Local Embedding Model**")
        st.code(metrics.get("embedding_model"), language="text")
        st.caption("Embeddings are generated locally using Sentence Transformers. Vector store: ChromaDB.")

    with col_prov:
        st.markdown("**Active LLM Generation Provider**")
        st.code(metrics.get("llm_provider"), language="text")
        if metrics.get("is_mock_provider"):
            st.info(
                "Currently using **Mock/Demo Provider**. Answers and study materials are synthesized locally without contacting external paid APIs."
            )
        else:
            st.success(f"Connected to external provider: {metrics.get('llm_provider')}")

    st.markdown("---")
    st.subheader("Privacy & Local Storage Transparency")
    st.markdown(
        """
        - **Local Storage:** Uploaded files and metadata are stored inside the local `data/` directory.
        - **Vector Storage:** Vector embeddings are stored persistently in `data/chroma/`.
        - **External Providers:** When configured with OpenAI, Gemini, or Groq, only retrieved evidence passages relevant to user queries are transmitted.
        - **Document Deletion:** Deleting a document removes its metadata and purges all associated vector records from ChromaDB.
        """
    )
