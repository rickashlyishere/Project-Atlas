from __future__ import annotations

from infrastructure.llm.ollama_provider import (
    OllamaProvider,
)
from config.llm import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT,
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
        max_sources=5,
        max_characters=12000,
        minimum_score=0.0,
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