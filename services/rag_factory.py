from __future__ import annotations

from infrastructure.llm.ollama_provider import (
    OllamaProvider,
)
from config.llm import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT,
)
from config.rag import (
    RAG_MAX_CONTEXT_CHARACTERS,
    RAG_MAX_SOURCES,
    RAG_MINIMUM_SCORE,
)

from services.context_assembler import (
    ContextAssembler,
)
from services.embedding_service import (
    EmbeddingService,
)
from services.llm_service import (
    LLMService,
)
from services.prompt_builder import (
    GroundedPromptBuilder,
)
from services.rag_service import (
    RAGService,
)
from services.search_service import (
    SearchService,
)
from services.vector_search_service import (
    VectorSearchService,
)


def create_rag_service(
    embedding_service: EmbeddingService,
    embedding_repository,
) -> RAGService:
    """
    Build the production Atlas RAG pipeline.
    """

    search_service = SearchService(
        embedding_service=embedding_service,
        embedding_repository=embedding_repository,
        vector_search_service=VectorSearchService(),
    )

    context_assembler = ContextAssembler(
        max_sources=RAG_MAX_SOURCES,
        max_characters=RAG_MAX_CONTEXT_CHARACTERS,
        minimum_score=RAG_MINIMUM_SCORE,
    )

    prompt_builder = GroundedPromptBuilder()

    llm_provider = OllamaProvider(
        model_name=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        timeout=OLLAMA_TIMEOUT,
    )

    llm_service = LLMService(
        provider=llm_provider,
    )

    return RAGService(
        search_service=search_service,
        context_assembler=context_assembler,
        prompt_builder=prompt_builder,
        llm_service=llm_service,
    )
