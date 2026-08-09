from __future__ import annotations

from services.document_service import DocumentService
from services.rag_factory import create_rag_service


def main() -> None:
    """
    Real end-to-end Atlas RAG smoke test.

    Uses:
        - real SQLite database
        - real document records
        - real embeddings
        - real vector search
        - real SearchService
        - real ContextAssembler
        - real GroundedPromptBuilder
        - real LLMService
        - real Ollama
        - real configured local LLM
    """

    print()
    print("=" * 70)
    print("PROJECT ATLAS - REAL RAG SMOKE TEST")
    print("=" * 70)
    print()

    # ------------------------------------------------------------
    # 1. Initialize the real document service
    # ------------------------------------------------------------

    print("Initializing DocumentService...")

    document_service = DocumentService()

    print("DocumentService initialized.")

    print(
        f"Embedding dimension: "
        f"{document_service.embedding_service.dimension}"
    )

    # ------------------------------------------------------------
    # 2. Build the real RAG service
    # ------------------------------------------------------------

    print()
    print("Building RAG service...")

    rag_service = create_rag_service(
        embedding_service=(
            document_service.embedding_service
        ),
        embedding_repository=(
            document_service.embedding_repository
        ),
    )

    print("RAG service initialized.")

    print(
        f"LLM model: {rag_service.model_name}"
    )

    # ------------------------------------------------------------
    # 3. Show documents currently in Atlas
    # ------------------------------------------------------------

    documents = document_service.list_documents()

    print()
    print(
        f"Documents in Atlas: {len(documents)}"
    )

    if not documents:
        raise RuntimeError(
            "No documents exist in the Atlas database."
        )

    print()
    print("DOCUMENTS")
    print("-" * 70)

    for index, document in enumerate(
        documents,
        start=1,
    ):
        filename = document["filename"]

        print(
            f"{index:>3}. {filename}"
        )

    # ------------------------------------------------------------
    # 4. Ask the user for a question
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("ATLAS QUESTION")
    print("=" * 70)
    print()

    question = input(
        "Question: "
    ).strip()

    if not question:
        raise RuntimeError(
            "Question cannot be empty."
        )

    # ------------------------------------------------------------
    # 5. Run the complete RAG pipeline
    # ------------------------------------------------------------

    print()
    print("Searching Atlas documents...")
    print()

    try:
        response = rag_service.answer(
            question,
            top_k=5,
        )

    except Exception as error:
        print()
        print("=" * 70)
        print("RAG PIPELINE FAILED")
        print("=" * 70)
        print()
        print(
            f"{type(error).__name__}: {error}"
        )
        print()

        raise

    # ------------------------------------------------------------
    # 6. Display answer
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("ANSWER")
    print("=" * 70)
    print()

    print(response.answer)

    # ------------------------------------------------------------
    # 7. Display retrieved search results
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("RETRIEVED RESULTS")
    print("=" * 70)
    print()

    if not response.search_results:
        print("No search results returned.")

    else:
        for index, result in enumerate(
            response.search_results,
            start=1,
        ):
            filename = result.get(
                "filename",
                "Unknown",
            )

            page_number = result.get(
                "page_number",
                0,
            )

            score = float(
                result.get(
                    "score",
                    0.0,
                )
            )

            chunk_id = result.get(
                "chunk_id",
                "",
            )

            print(
                f"[{index}] "
                f"{filename} "
                f"| page {page_number} "
                f"| score {score:.4f}"
            )

            print(
                f"    Chunk: {chunk_id}"
            )

            text = str(
                result.get(
                    "text",
                    "",
                )
            ).strip()

            preview = text.replace(
                "\n",
                " ",
            )

            if len(preview) > 200:
                preview = (
                    preview[:200]
                    + "..."
                )

            print(
                f"    Text: {preview}"
            )

            print()

    # ------------------------------------------------------------
    # 8. Display context sent to the LLM
    # ------------------------------------------------------------

    print("=" * 70)
    print("LLM CONTEXT")
    print("=" * 70)
    print()

    print(response.context.text)

    # ------------------------------------------------------------
    # 9. Display structured sources
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("SOURCES")
    print("=" * 70)
    print()

    if not response.sources:
        print("No sources returned.")

    else:
        for source in response.sources:
            print(
                f"[Source {source.source_number}] "
                f"{source.filename} "
                f"(page {source.page_number}, "
                f"score={source.score:.4f})"
            )

    print()
    print("=" * 70)
    print("RAG TEST COMPLETE")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
