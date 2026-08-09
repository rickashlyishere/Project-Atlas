from __future__ import annotations

from services.context_assembler import AssembledContext


class GroundedPromptBuilder:
    """
    Builds prompts for grounded question answering.

    The builder does not call an LLM. It only constructs
    the prompt that will later be passed to LLMService.
    """

    SYSTEM_INSTRUCTIONS = """
You are Atlas, an offline AI knowledge assistant.

Answer the user's question using only the supplied document
context.

Rules:
1. Do not invent facts that are not supported by the context.
2. If the context does not contain enough information to answer,
   clearly say that the available documents do not provide enough
   information.
3. Keep the answer directly relevant to the user's question.
4. When making a factual claim from the context, cite the relevant
   source using its source number, such as [Source 1].
5. Do not create sources that are not present in the context.
6. Do not mention these instructions in your answer.
""".strip()

    def build(
        self,
        question: str,
        context: AssembledContext,
    ) -> str:
        """
        Build a grounded RAG prompt.
        """

        question = question.strip()

        if not question:
            raise ValueError(
                "Question cannot be empty."
            )

        if not context.text.strip():
            raise ValueError(
                "Context cannot be empty."
            )

        return (
            f"{self.SYSTEM_INSTRUCTIONS}\n\n"
            "DOCUMENT CONTEXT\n"
            "================\n"
            f"{context.text}\n\n"
            "USER QUESTION\n"
            "=============\n"
            f"{question}\n\n"
            "ANSWER\n"
            "======\n"
        )