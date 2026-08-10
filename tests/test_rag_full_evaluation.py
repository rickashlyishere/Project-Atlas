from __future__ import annotations

from services.document_service import DocumentService
from services.rag_factory import create_rag_service


RETRIEVAL_TESTS = [
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


GROUNDEDNESS_TESTS = [
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


OUT_OF_CONTEXT_TESTS = [
    (
        "France Capital",
        "What is the capital of France?",
    ),
    (
        "Solar System",
        "How many planets are in the Solar System?",
    ),
    (
        "World War",
        "When did World War II end?",
    ),
]


def normalize(text: str) -> str:
    return " ".join(
        text.lower().split()
    )


def build_services():
    document_service = DocumentService()

    rag_service = create_rag_service(
        embedding_service=(
            document_service.embedding_service
        ),
        embedding_repository=(
            document_service.embedding_repository
        ),
    )

    return document_service, rag_service


def check_documents(
    document_service: DocumentService,
) -> None:

    documents = (
        document_service.list_documents()
    )

    document_names = {
        str(document["filename"])
        for document in documents
    }

    required_documents = {
        expected
        for _, _, expected in RETRIEVAL_TESTS
    }

    missing = (
        required_documents
        - document_names
    )

    if missing:
        raise RuntimeError(
            "Missing evaluation documents: "
            + ", ".join(
                sorted(missing)
            )
        )


def run_retrieval_tests(
    rag_service,
) -> tuple[int, int]:

    print()
    print("=" * 70)
    print("1. RETRIEVAL EVALUATION")
    print("=" * 70)

    top_1_passed = 0
    top_3_passed = 0

    for index, (
        name,
        question,
        expected_document,
    ) in enumerate(
        RETRIEVAL_TESTS,
        start=1,
    ):

        response = rag_service.answer(
            question,
            top_k=5,
        )

        results = response.search_results

        retrieved = [
            str(
                result.get(
                    "filename",
                    "",
                )
            )
            for result in results
        ]

        top_1 = (
            len(retrieved) > 0
            and retrieved[0]
            == expected_document
        )

        top_3 = (
            expected_document
            in retrieved[:3]
        )

        if top_1:
            top_1_passed += 1

        if top_3:
            top_3_passed += 1

        status = (
            "PASS"
            if top_1
            else "FAIL"
        )

        score = (
            float(results[0]["score"])
            if results
            else 0.0
        )

        print(
            f"[{index}/{len(RETRIEVAL_TESTS)}] "
            f"{name:<15} "
            f"{status:<5} "
            f"Top-1={retrieved[0] if retrieved else 'NONE'} "
            f"| score={score:.4f}"
        )

    return (
        top_1_passed,
        top_3_passed,
    )


def run_groundedness_tests(
    rag_service,
) -> int:

    print()
    print("=" * 70)
    print("2. GROUNDEDNESS EVALUATION")
    print("=" * 70)

    passed = 0

    for index, (
        name,
        question,
        expected_document,
        expected_terms,
    ) in enumerate(
        GROUNDEDNESS_TESTS,
        start=1,
    ):

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

        context_has_evidence = any(
            normalize(term)
            in context
            for term in expected_terms
        )

        answer_has_evidence = any(
            normalize(term)
            in answer
            for term in expected_terms
        )

        grounded = (
            correct_source
            and context_has_evidence
            and answer_has_evidence
        )

        if grounded:
            passed += 1

        print(
            f"[{index}/{len(GROUNDEDNESS_TESTS)}] "
            f"{name:<15} "
            f"{'PASS' if grounded else 'FAIL'}"
        )

    return passed


def run_out_of_context_tests(
    rag_service,
) -> int:

    print()
    print("=" * 70)
    print("3. OUT-OF-CONTEXT EVALUATION")
    print("=" * 70)

    passed = 0

    refusal_indicators = [
        "not found",
        "not available",
        "not provided",
        "does not contain",
        "don't have",
        "do not have",
        "cannot answer",
        "can't answer",
        "insufficient",
        "no relevant",
        "not mentioned",
        "not specified",
        "outside",
        "cannot determine",
    ]

    for index, (
        name,
        question,
    ) in enumerate(
        OUT_OF_CONTEXT_TESTS,
        start=1,
    ):

        try:

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

            relevant_sources = len(
                response.sources
            )

            refusal = any(
                indicator in answer
                for indicator in refusal_indicators
            )

            weak_context = (
                len(context.strip()) == 0
                or relevant_sources == 0
            )

            passed_case = (
                refusal
                or weak_context
            )

        except ValueError:

            passed_case = True

        if passed_case:
            passed += 1

        print(
            f"[{index}/{len(OUT_OF_CONTEXT_TESTS)}] "
            f"{name:<20} "
            f"{'PASS' if passed_case else 'FAIL'}"
        )

    return passed


def main() -> None:

    print()
    print("=" * 70)
    print("PROJECT ATLAS")
    print("FULL RAG EVALUATION")
    print("=" * 70)

    print()
    print("Initializing Atlas...")

    document_service, rag_service = (
        build_services()
    )

    print(
        f"LLM model: "
        f"{rag_service.model_name}"
    )

    print(
        f"Embedding dimension: "
        f"{document_service.embedding_service.dimension}"
    )

    check_documents(
        document_service
    )

    print(
        "Evaluation corpus: READY"
    )

    top_1, top_3 = (
        run_retrieval_tests(
            rag_service
        )
    )

    grounded = (
        run_groundedness_tests(
            rag_service
        )
    )

    out_of_context = (
        run_out_of_context_tests(
            rag_service
        )
    )

    retrieval_total = len(
        RETRIEVAL_TESTS
    )

    grounded_total = len(
        GROUNDEDNESS_TESTS
    )

    out_of_context_total = len(
        OUT_OF_CONTEXT_TESTS
    )

    total_passed = (
        top_1
        + top_3
        + grounded
        + out_of_context
    )

    total_tests = (
        retrieval_total
        + retrieval_total
        + grounded_total
        + out_of_context_total
    )

    overall_accuracy = (
        total_passed
        / total_tests
    ) * 100

    print()
    print("=" * 70)
    print("FINAL ATLAS RAG EVALUATION")
    print("=" * 70)

    print()

    print(
        f"Top-1 Retrieval: "
        f"{top_1}/{retrieval_total}"
    )

    print(
        f"Top-3 Retrieval: "
        f"{top_3}/{retrieval_total}"
    )

    print(
        f"Groundedness: "
        f"{grounded}/{grounded_total}"
    )

    print(
        f"Out-of-Context Handling: "
        f"{out_of_context}/{out_of_context_total}"
    )

    print()

    print(
        f"Overall: "
        f"{total_passed}/{total_tests} "
        f"({overall_accuracy:.1f}%)"
    )

    print()

    if total_passed == total_tests:

        print(
            "RESULT: ALL ATLAS RAG TESTS PASSED"
        )

    else:

        print(
            "RESULT: SOME ATLAS RAG TESTS FAILED"
        )

    print("=" * 70)
    print()


if __name__ == "__main__":
    main()