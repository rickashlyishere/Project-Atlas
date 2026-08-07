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
def get_service() -> DocumentService:
    return DocumentService()


service = get_service()

st.title("📚 Project Atlas")
st.caption("Offline AI Knowledge Platform")

uploaded_file = st.file_uploader(
    "Upload a document",
    type=["pdf", "docx", "txt"],
)

if uploaded_file:

    upload_dir = Path("data/uploads")

    upload_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_path = upload_dir / uploaded_file.name

    file_path.write_bytes(
        uploaded_file.getbuffer()
    )

    document = service.load(file_path)

    st.success("Document imported successfully!")

st.divider()

st.subheader("📚 Document Library")

documents = service.list_documents()

if not documents:

    st.info("No documents imported.")

else:

    for row in documents:

        st.container(border=True)

        st.write(f"**{row['filename']}**")

        st.caption(
            f"{row['document_type'].upper()} • "
            f"{row['page_count']} pages • "
            f"{row['file_size']:,} bytes"
        )