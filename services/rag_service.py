from __future__ import annotations

from typing import Any

from services.context_assembler import (
    AssembledContext,
    ContextAssembler,
)
from services.llm_service import LLMService
from services.prompt_builder import (
    GroundedPromptBuilder,
)
from services.search_service import SearchService


class RAGResponse:
    """
    Complete response produced by the RAG pipeline.
    """

    def __init__(
        self,
        answer: str,
        sources: list[Any],
        search_results: list[dict[str, Any]],
        context: AssembledContext,
    ) -> None:
        self.answer = answer
        self.sources = sources
        self.search_results = search_results
        self.context = context


class RAGService:
    """
    Orchestrates Atlas's complete retrieval-augmented
    generation pipeline.

    Pipeline:

        Question
            ↓
        SearchService
            ↓
        ContextAssembler
            ↓
        GroundedPromptBuilder
            ↓
        LLMService
            ↓
        Answer
    """

    def __init__(
        self,
        search_service: SearchService,
        context_assembler: ContextAssembler,
        prompt_builder: GroundedPromptBuilder,
        llm_service: LLMService,
    ) -> None:
        self.search_service = search_service
        self.context_assembler = context_assembler
        self.prompt_builder = prompt_builder
        self.llm_service = llm_service

    @property
    def model_name(self) -> str:
        """
        Return the model used by the configured LLM service.
        """

        return self.llm_service.model_name

    def answer(
        self,
        question: str,
        top_k: int = 5,
    ) -> RAGResponse:
        """
        Answer a question using retrieved document context.
        """

        question = question.strip()

        if not question:
            raise ValueError(
                "Question cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        search_results = self.search_service.search(
            question,
            top_k=top_k,
        )

        context = self.context_assembler.assemble(
            search_results
        )

        if not context.sources:
            raise ValueError(
                "No relevant document context was found."
            )

        prompt = self.prompt_builder.build(
            question=question,
            context=context,
        )

        answer = self.llm_service.generate(
            prompt
        )

        return RAGResponse(
            answer=answer,
            sources=context.sources,
            search_results=search_results,
            context=context,
        )