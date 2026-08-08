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
    """
    Create and cache the Atlas document service.
    """

    return DocumentService()


service = get_service()


st.title("📚 Project Atlas")
st.caption("Offline AI Knowledge Platform")


uploaded_file = st.file_uploader(
    "Upload a document",
    type=[
        "pdf",
        "docx",
        "pptx",
        "txt",
        "png",
        "jpg",
        "jpeg",
    ],
)


if uploaded_file is not None:

    upload_dir = Path("data/uploads")

    upload_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_path = upload_dir / uploaded_file.name

    file_path.write_bytes(
        uploaded_file.getbuffer()
    )

    try:

        document = service.load(file_path)

        st.success(
            f"'{document.filename}' imported successfully!"
        )

        st.write(
            f"**Type:** {document.document_type.value.upper()}"
        )

        st.write(
            f"**Pages:** {document.page_count}"
        )

        st.write(
            f"**File Size:** {document.file_size:,} bytes"
        )

        if document.extracted_text:

            st.subheader("Extracted Text")

            st.text_area(
                "OCR Result",
                document.extracted_text,
                height=300,
            )

        else:

            st.warning(
                "Atlas imported the image, but OCR did not "
                "extract any text."
            )

    except Exception as error:

        st.error(
            f"Failed to import "
            f"'{uploaded_file.name}': {error}"
        )


st.divider()

st.subheader("📚 Document Library")


documents = service.list_documents()


if not documents:

    st.info("No documents imported.")

else:

    for row in documents:

        with st.container(border=True):

            st.write(
                f"**{row['filename']}**"
            )

            st.caption(
                f"{row['document_type'].upper()} • "
                f"{row['page_count']} pages • "
                f"{row['file_size']:,} bytes"
            )

            if row["title"]:

                st.write(
                    f"**Title:** {row['title']}"
                )

            if row["author"]:

                st.write(
                    f"**Author:** {row['author']}"
                )