from __future__ import annotations

from services.document_service import DocumentService
from services.rag_factory import create_rag_service


QUESTIONS = [
    (
        "Python",
        "What is Python used for?",
        "python.txt",
    ),
    (
        "Linux",
        "What is Linux?",
        "linux.txt",
    ),
    (
        "Mathematics",
        "What are the properties of a triangle?",
        "mathematics.txt",
    ),
    (
        "Football",
        "What is the role of a defender in football?",
        "football.txt",
    ),
    (
        "Atlas",
        "What is Atlas designed to do?",
        "atlas.txt",
    ),
]


def main() -> None:
    print()
    print("=" * 70)
    print("PROJECT ATLAS - RAG RETRIEVAL EVALUATION")
    print("=" * 70)
    print()

    print("Initializing DocumentService...")

    document_service = DocumentService()

    print("DocumentService initialized.")

    print(
        f"Embedding dimension: "
        f"{document_service.embedding_service.dimension}"
    )

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

    documents = document_service.list_documents()

    print()
    print(
        f"Documents currently in Atlas: "
        f"{len(documents)}"
    )

    print()

    if not documents:
        raise RuntimeError(
            "Atlas contains no documents."
        )

    document_names = {
        str(document["filename"])
        for document in documents
    }

    expected_documents = {
        expected
        for _, _, expected in QUESTIONS
    }

    missing = (
        expected_documents
        - document_names
    )

    if missing:
        raise RuntimeError(
            "Missing evaluation documents: "
            + ", ".join(
                sorted(missing)
            )
        )

    print("=" * 70)
    print("RUNNING EVALUATION")
    print("=" * 70)
    print()

    top_1_correct = 0
    top_3_correct = 0

    results = []

    for index, (
        name,
        question,
        expected_document,
    ) in enumerate(
        QUESTIONS,
        start=1,
    ):

        print(
            f"[{index}/{len(QUESTIONS)}] "
            f"{name}"
        )

        print(
            f"Question: {question}"
        )

        print(
            f"Expected: {expected_document}"
        )

        try:

            response = rag_service.answer(
                question,
                top_k=5,
            )

        except Exception as error:

            print(
                f"ERROR: "
                f"{type(error).__name__}: "
                f"{error}"
            )

            print()

            results.append(
                {
                    "name": name,
                    "question": question,
                    "expected": expected_document,
                    "retrieved": [],
                    "top_1": False,
                    "top_3": False,
                }
            )

            continue

        search_results = (
            response.search_results
        )

        retrieved_documents = [
            str(
                result.get(
                    "filename",
                    "",
                )
            )
            for result in search_results
        ]

        top_1_match = (
            len(retrieved_documents) > 0
            and retrieved_documents[0]
            == expected_document
        )

        top_3_match = (
            expected_document
            in retrieved_documents[:3]
        )

        if top_1_match:
            top_1_correct += 1

        if top_3_match:
            top_3_correct += 1

        print()

        print("Retrieved:")

        if not search_results:

            print(
                "  No results returned."
            )

        else:

            for rank, result in enumerate(
                search_results,
                start=1,
            ):

                filename = str(
                    result.get(
                        "filename",
                        "Unknown",
                    )
                )

                score = float(
                    result.get(
                        "score",
                        0.0,
                    )
                )

                page = result.get(
                    "page_number",
                    0,
                )

                print(
                    f"  {rank}. "
                    f"{filename} "
                    f"| page {page} "
                    f"| score {score:.4f}"
                )

        print()

        print(
            "Top-1: "
            + (
                "PASS"
                if top_1_match
                else "FAIL"
            )
        )

        print(
            "Top-3: "
            + (
                "PASS"
                if top_3_match
                else "FAIL"
            )
        )

        results.append(
            {
                "name": name,
                "question": question,
                "expected": expected_document,
                "retrieved": retrieved_documents,
                "top_1": top_1_match,
                "top_3": top_3_match,
            }
        )

        print("-" * 70)
        print()

    total = len(QUESTIONS)

    top_1_accuracy = (
        top_1_correct / total
    ) * 100

    top_3_accuracy = (
        top_3_correct / total
    ) * 100

    print("=" * 70)
    print("RAG EVALUATION RESULTS")
    print("=" * 70)
    print()

    for result in results:

        top_1_status = (
            "PASS"
            if result["top_1"]
            else "FAIL"
        )

        top_3_status = (
            "PASS"
            if result["top_3"]
            else "FAIL"
        )

        print(
            f"{result['name']:<15}"
            f"Top-1: {top_1_status:<5}"
            f"Top-3: {top_3_status}"
        )

    print()

    print(
        f"Top-1 Accuracy: "
        f"{top_1_correct}/{total} "
        f"({top_1_accuracy:.1f}%)"
    )

    print(
        f"Top-3 Accuracy: "
        f"{top_3_correct}/{total} "
        f"({top_3_accuracy:.1f}%)"
    )

    print()

    print("=" * 70)

    if (
        top_1_correct == total
        and top_3_correct == total
    ):
        print(
            "RESULT: ALL RETRIEVAL TESTS PASSED"
        )
    else:
        print(
            "RESULT: SOME RETRIEVAL TESTS FAILED"
        )

    print("=" * 70)
    print()


if __name__ == "__main__":
    main()