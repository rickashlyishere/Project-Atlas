from __future__ import annotations

from pathlib import Path

import streamlit as st

from services.document_service import DocumentService
from services.rag_factory import create_rag_service
from services.search_service import SearchService
from services.vector_search_service import VectorSearchService


st.set_page_config(
    page_title="Project Atlas",
    page_icon="📚",
    layout="wide",
)


# ============================================================
# SERVICES
# ============================================================

@st.cache_resource
def get_service() -> DocumentService:
    return DocumentService()


def get_search_service(
    service: DocumentService,
) -> SearchService:
    return SearchService(
        embedding_service=service.embedding_service,
        embedding_repository=service.embedding_repository,
        vector_search_service=VectorSearchService(),
    )


@st.cache_resource
def get_rag_service(
    _service: DocumentService,
):
    return create_rag_service(
        embedding_service=_service.embedding_service,
        embedding_repository=_service.embedding_repository,
    )


service = get_service()

search_service = get_search_service(
    service
)

rag_service = get_rag_service(
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


documents = service.list_documents()


# ============================================================
# ASK ATLAS
# ============================================================

st.divider()

st.header("🤖 Ask Atlas")

st.caption(
    "Ask questions about your indexed documents. "
    "Atlas retrieves relevant content and uses the "
    "configured local Ollama model to generate a grounded answer."
)


question = st.text_area(
    "Ask a question",
    placeholder=(
        "Example: Who is Rithvik and what are his interests?"
    ),
    height=100,
)


ask_col1, ask_col2 = st.columns(
    [3, 1]
)


with ask_col1:

    ask_button = st.button(
        "🤖 Ask Atlas",
        type="primary",
        use_container_width=True,
    )


with ask_col2:

    rag_top_k = st.number_input(
        "Sources",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
    )


if ask_button:

    if not question.strip():

        st.warning(
            "Enter a question first."
        )

    else:

        try:

            with st.spinner(
                f"Atlas is processing with "
                f"{rag_service.model_name}..."
            ):

                response = rag_service.answer(
                    question=question,
                    top_k=int(rag_top_k),
                )

            st.subheader("Answer")

            st.write(
                response.answer
            )

            st.subheader("Sources")

            for source in response.sources:

                with st.expander(
                    f"[Source {source.source_number}] "
                    f"{source.filename} — "
                    f"Page {source.page_number}"
                ):

                    st.write(
                        f"**Similarity:** "
                        f"{source.score:.4f}"
                    )

                    st.write(
                        source.text
                    )

        except ValueError as error:

            st.warning(
                str(error)
            )

        except Exception as error:

            st.error(
                f"Atlas could not generate an answer: "
                f"{error}"
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


document_options: dict[str, str] = {
    "All documents": "",
}


for row in documents:

    document_id = str(
        row["id"]
    )

    filename = str(
        row["filename"]
    )

    document_options[
        filename
    ] = document_id


search_col1, search_col2, search_col3 = st.columns(
    [3, 1, 1]
)


with search_col1:

    selected_document_name = st.selectbox(
        "Search in",
        options=list(
            document_options.keys()
        ),
    )


with search_col2:

    top_k = st.number_input(
        "Results",
        min_value=1,
        max_value=50,
        value=10,
        step=1,
    )


with search_col3:

    minimum_score = st.number_input(
        "Minimum similarity",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        format="%.2f",
    )


search_button = st.button(
    "🔎 Search",
    type="primary",
    use_container_width=True,
)


if search_button:

    if not query.strip():

        st.warning(
            "Enter a search query first."
        )

    else:

        selected_document_id = (
            document_options[
                selected_document_name
            ]
        )

        try:

            with st.spinner(
                "Searching Atlas..."
            ):

                if selected_document_id:

                    results = (
                        search_service.search_document(
                            document_id=(
                                selected_document_id
                            ),
                            query=query,
                            top_k=int(top_k),
                        )
                    )

                else:

                    results = (
                        search_service.search(
                            query=query,
                            top_k=int(top_k),
                        )
                    )

            results = [
                result
                for result in results
                if float(
                    result["score"]
                ) >= minimum_score
            ]

            if not results:

                if minimum_score > 0:

                    st.info(
                        "No results met the minimum "
                        f"similarity score of "
                        f"{minimum_score:.2f}."
                    )

                else:

                    st.info(
                        "No indexed content was found "
                        "for this search."
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

                    score = float(
                        result["score"]
                    )

                    st.markdown(
                        f"### Result {index}"
                    )

                    result_col1, result_col2, result_col3 = (
                        st.columns(3)
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

                    with result_col3:

                        st.write(
                            f"**Similarity:** "
                            f"{score:.4f}"
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


if not documents:

    st.info(
        "No documents imported."
    )

else:

    for row in documents:

        document_id = str(
            row["id"]
        )

        filename = str(
            row["filename"]
        )

        with st.container(
            border=True
        ):

            st.write(
                f"**{filename}**"
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

            # ------------------------------------------------
            # DELETE WORKFLOW
            # ------------------------------------------------

            confirmation_key = (
                f"confirm_delete_{document_id}"
            )

            pending_key = (
                f"pending_delete_{document_id}"
            )

            if pending_key not in st.session_state:

                st.session_state[
                    pending_key
                ] = False

            if not st.session_state[
                pending_key
            ]:

                if st.button(
                    "🗑️ Delete",
                    key=f"delete_{document_id}",
                    use_container_width=False,
                ):

                    st.session_state[
                        pending_key
                    ] = True

                    st.rerun()

            else:

                st.warning(
                    f"Are you sure you want to delete "
                    f"'{filename}'?"
                )

                st.caption(
                    "This removes the document, its chunks, "
                    "and its embeddings from Atlas. "
                    "The original source file will not be deleted."
                )

                confirm_col1, confirm_col2 = st.columns(
                    2
                )

                with confirm_col1:

                    if st.button(
                        "Cancel",
                        key=f"cancel_{document_id}",
                        use_container_width=True,
                    ):

                        st.session_state[
                            pending_key
                        ] = False

                        st.rerun()

                with confirm_col2:

                    if st.button(
                        "⚠️ Delete Permanently",
                        key=confirmation_key,
                        type="primary",
                        use_container_width=True,
                    ):

                        try:

                            deleted = service.delete(
                                document_id
                            )

                            st.session_state[
                                pending_key
                            ] = False

                            if deleted:

                                st.success(
                                    f"'{filename}' "
                                    "was deleted from Atlas."
                                )

                                st.rerun()

                            else:

                                st.warning(
                                    "The document was already "
                                    "removed from Atlas."
                                )

                                st.rerun()

                        except Exception as error:

                            st.session_state[
                                pending_key
                            ] = False

                            st.error(
                                f"Failed to delete "
                                f"'{filename}': {error}"
                            )