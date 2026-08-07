from __future__ import annotations

from pathlib import Path

import streamlit as st

from services.document_service import DocumentService


st.set_page_config(
    page_title="Project Atlas",
    page_icon="📚",
    layout="wide",
)


@st.cache_resource
def get_document_service() -> DocumentService:
    """
    Create a single DocumentService instance.
    """
    return DocumentService()


service = get_document_service()

st.title("📚 Project Atlas")
st.caption("Offline AI Knowledge Platform")

uploaded_file = st.file_uploader(
    "Upload a document",
    type=["pdf", "docx", "txt"],
)

if uploaded_file is not None:

    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_path = upload_dir / uploaded_file.name

    file_path.write_bytes(uploaded_file.getbuffer())

    document = service.load(file_path)

    st.success("Document parsed successfully!")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Information")

        st.write(f"**Filename:** {document.filename}")
        st.write(f"**Type:** {document.document_type.value}")
        st.write(f"**Pages:** {document.page_count}")
        st.write(f"**File Size:** {document.file_size:,} bytes")

    with col2:
        st.subheader("Metadata")

        st.write(f"**Title:** {document.metadata.title or '-'}")
        st.write(f"**Author:** {document.metadata.author or '-'}")
        st.write(f"**Subject:** {document.metadata.subject or '-'}")

    st.divider()

    st.subheader("Extracted Text")

    st.text_area(
        "Content",
        document.extracted_text,
        height=450,
        label_visibility="collapsed",
    )