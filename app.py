from __future__ import annotations

from pathlib import Path

import streamlit as st

from services.document_service import DocumentService
from services.search_service import SearchService
from services.vector_search_service import VectorSearchService


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


def get_search_service(
    service: DocumentService,
) -> SearchService:
    """
    Create the semantic search service using the
    already-cached DocumentService dependencies.

    This function is intentionally not cached because
    DocumentService contains objects that Streamlit cannot
    reliably hash.
    """

    return SearchService(
        embedding_service=service.embedding_service,
        embedding_repository=service.embedding_repository,
        vector_search_service=VectorSearchService(),
    )


service = get_service()

search_service = get_search_service(
    service
)


# ============================================================
# HEADER
# ============================================================

st.title("📚 Project Atlas")

st.caption(
    "Offline AI Knowledge Platform"
)


# ============================================================
# DOCUMENT UPLOAD
# ============================================================

st.header("Import Document")


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

    upload_dir = Path(
        "data/uploads"
    )

    upload_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_path = (
        upload_dir
        / uploaded_file.name
    )

    file_path.write_bytes(
        uploaded_file.getbuffer()
    )

    try:

        with st.spinner(
            "Atlas is processing the document..."
        ):

            document = service.load(
                file_path
            )

        st.success(
            f"'{document.filename}' "
            "imported successfully!"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Type",
                document.document_type.value.upper(),
            )

        with col2:

            st.metric(
                "Pages",
                document.page_count,
            )

        with col3:

            st.metric(
                "File Size",
                f"{document.file_size:,} bytes",
            )

        chunks = service.get_chunks(
            document.id
        )

        st.caption(
            f"Indexed {len(chunks)} chunks "
            "for semantic search."
        )

        if document.extracted_text:

            st.subheader(
                "Extracted Text"
            )

            st.text_area(
                "OCR Result",
                document.extracted_text,
                height=300,
            )

        else:

            st.warning(
                "Atlas imported the image, but "
                "OCR did not extract any text."
            )

    except Exception as error:

        st.error(
            f"Failed to import "
            f"'{uploaded_file.name}': {error}"
        )


# ============================================================
# SEMANTIC SEARCH
# ============================================================

st.divider()

st.header(
    "🔎 Semantic Search"
)

st.caption(
    "Search your indexed documents using natural language."
)


query = st.text_input(
    "What are you looking for?",
    placeholder=(
        "Example: What does this document say about triangles?"
    ),
)


search_col1, search_col2 = st.columns(
    [3, 1]
)


with search_col1:

    top_k = st.slider(
        "Number of results",
        min_value=1,
        max_value=10,
        value=5,
    )


with search_col2:

    search_button = st.button(
        "Search",
        type="primary",
        use_container_width=True,
    )


if search_button:

    if not query.strip():

        st.warning(
            "Enter a search query first."
        )

    else:

        try:

            with st.spinner(
                "Searching Atlas..."
            ):

                results = search_service.search(
                    query=query,
                    top_k=top_k,
                )

            if not results:

                st.info(
                    "No indexed content was found."
                )

            else:

                st.success(
                    f"Found {len(results)} "
                    "relevant chunks."
                )

                for index, result in enumerate(
                    results,
                    start=1,
                ):

                    st.markdown(
                        f"### Result {index}"
                    )

                    result_col1, result_col2 = (
                        st.columns(2)
                    )

                    with result_col1:

                        st.write(
                            f"**Document:** "
                            f"{result['filename']}"
                        )

                    with result_col2:

                        st.write(
                            f"**Page:** "
                            f"{result['page_number']}"
                        )

                    score = float(
                        result["score"]
                    )

                    st.progress(
                        min(
                            max(
                                score,
                                0.0,
                            ),
                            1.0,
                        )
                    )

                    st.caption(
                        f"Similarity score: "
                        f"{score:.4f}"
                    )

                    st.write(
                        result["text"]
                    )

                    with st.expander(
                        "Chunk metadata"
                    ):

                        st.json(
                            {
                                "document_id": (
                                    result[
                                        "document_id"
                                    ]
                                ),
                                "chunk_id": (
                                    result[
                                        "chunk_id"
                                    ]
                                ),
                                "embedding_id": (
                                    result[
                                        "embedding_id"
                                    ]
                                ),
                                "chunk_type": (
                                    result[
                                        "chunk_type"
                                    ]
                                ),
                                "model": (
                                    result[
                                        "model_name"
                                    ]
                                ),
                            }
                        )

                    if index < len(results):

                        st.divider()

        except Exception as error:

            st.error(
                f"Search failed: {error}"
            )


# ============================================================
# DOCUMENT LIBRARY
# ============================================================

st.divider()

st.subheader(
    "📚 Document Library"
)


documents = service.list_documents()


if not documents:

    st.info(
        "No documents imported."
    )

else:

    for row in documents:

        with st.container(
            border=True
        ):

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