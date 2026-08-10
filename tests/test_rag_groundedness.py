from __future__ import annotations

from services.document_service import DocumentService
from services.rag_factory import create_rag_service


TEST_CASES = [
    (
        "Python",
        "What is Python used for?",
        "python.txt",
        [
            "automation",
            "data analysis",
            "web development",
            "scientific computing",
            "artificial intelligence",
        ],
    ),
    (
        "Linux",
        "What is Linux?",
        "linux.txt",
        [
            "open-source",
            "operating system kernel",
            "servers",
            "desktop computers",
            "embedded devices",
        ],
    ),
    (
        "Mathematics",
        "What are the properties of a triangle?",
        "mathematics.txt",
        [
            "three sides",
            "three angles",
            "180 degrees",
            "equilateral",
            "isosceles",
            "scalene",
        ],
    ),
    (
        "Football",
        "What is the role of a defender in football?",
        "football.txt",
        [
            "preventing opposing players",
            "scoring opportunities",
            "protect",
            "goal",
        ],
    ),
    (
        "Atlas",
        "What is Atlas designed to do?",
        "atlas.txt",
        [
            "offline AI knowledge platform",
            "documents",
            "embeddings",
            "vector search",
            "Ollama",
        ],
    ),
]


def normalize(text: str) -> str:
    return " ".join(
        text.lower().split()
    )


def main() -> None:
    print()
    print("=" * 70)
    print("PROJECT ATLAS - GROUNDEDNESS EVALUATION")
    print("=" * 70)
    print()

    print("Initializing Atlas...")

    document_service = DocumentService()

    rag_service = create_rag_service(
        embedding_service=(
            document_service.embedding_service
        ),
        embedding_repository=(
            document_service.embedding_repository
        ),
    )

    print(
        f"LLM model: {rag_service.model_name}"
    )

    print()

    documents = document_service.list_documents()

    document_names = {
        str(document["filename"])
        for document in documents
    }

    expected_documents = {
        expected_document
        for _, _, expected_document, _ in TEST_CASES
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

    passed = 0
    total = len(TEST_CASES)

    print("=" * 70)
    print("RUNNING GROUNDEDNESS TESTS")
    print("=" * 70)
    print()

    for index, (
        name,
        question,
        expected_document,
        expected_terms,
    ) in enumerate(
        TEST_CASES,
        start=1,
    ):

        print(
            f"[{index}/{total}] {name}"
        )

        print(
            f"Question: {question}"
        )

        response = rag_service.answer(
            question,
            top_k=5,
        )

        answer = normalize(
            response.answer
        )

        context = normalize(
            response.context.text
        )

        source_documents = {
            str(source.filename)
            for source in response.sources
        }

        correct_source = (
            expected_document
            in source_documents
        )

        answer_terms_found = [
            term
            for term in expected_terms
            if normalize(term) in answer
        ]

        answer_terms_missing = [
            term
            for term in expected_terms
            if normalize(term) not in answer
        ]

        answer_uses_context = any(
            normalize(term) in context
            for term in expected_terms
        )

        grounded = (
            correct_source
            and answer_uses_context
            and len(answer_terms_found) > 0
        )

        if grounded:
            passed += 1

        print(
            f"Expected source: "
            f"{expected_document}"
        )

        print(
            f"Retrieved expected source: "
            f"{'PASS' if correct_source else 'FAIL'}"
        )

        print(
            f"Answer contains expected "
            f"evidence: "
            f"{'PASS' if answer_terms_found else 'FAIL'}"
        )

        print(
            f"Groundedness: "
            f"{'PASS' if grounded else 'FAIL'}"
        )

        print()
        print("Answer:")
        print(response.answer)

        print()
        print(
            "Evidence found in answer:"
        )

        if answer_terms_found:
            for term in answer_terms_found:
                print(
                    f"  + {term}"
                )
        else:
            print("  None")

        if answer_terms_missing:
            print()
            print(
                "Expected evidence not found:"
            )

            for term in answer_terms_missing:
                print(
                    f"  - {term}"
                )

        print()
        print("-" * 70)
        print()

    accuracy = (
        passed / total
    ) * 100

    print("=" * 70)
    print("GROUNDEDNESS RESULTS")
    print("=" * 70)
    print()

    print(
        f"Grounded answers: "
        f"{passed}/{total}"
    )

    print(
        f"Groundedness score: "
        f"{accuracy:.1f}%"
    )

    print()

    if passed == total:
        print(
            "RESULT: ALL GROUNDEDNESS TESTS PASSED"
        )
    else:
        print(
            "RESULT: SOME GROUNDEDNESS TESTS FAILED"
        )

    print("=" * 70)
    print()


if __name__ == "__main__":
    main()