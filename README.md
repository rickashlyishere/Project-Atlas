# Project Atlas

**Offline AI Knowledge Platform**

Atlas is a local, document-grounded AI knowledge platform designed to let users import documents, search their contents semantically, and ask questions using a local RAG pipeline.

Instead of sending documents to a cloud AI service, Atlas processes the knowledge locally and uses a configured Ollama model to generate answers grounded in retrieved document content.

## What Atlas Does

Atlas provides three core workflows:

1. **Document ingestion** — import supported documents and index their contents.
2. **Semantic search** — search indexed content using natural-language queries.
3. **Grounded RAG** — ask questions and receive answers based on retrieved document context, with source information attached to the response.

The application also provides a document library for managing indexed documents.

## How It Works

```text
                 ┌──────────────────┐
                 │   User Document  │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │     Parsers      │
                 │ PDF / DOCX /     │
                 │ PPTX / TXT /     │
                 │ Images           │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │     Chunking     │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │    Embeddings    │
                 │ Sentence         │
                 │ Transformers     │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │   Vector Search  │
                 └────────┬─────────┘
                          │
             ┌────────────┴────────────┐
             │                         │
             ▼                         ▼
      Semantic Search            RAG Pipeline
                                       │
                                       ▼
                              Context Assembly
                                       │
                                       ▼
                              Grounded Prompt
                                       │
                                       ▼
                                    Ollama
                                       │
                                       ▼
                                  Answer
                                       │
                                       ▼
                                   Sources
```

## RAG Pipeline

When a question is submitted, Atlas:

1. Converts the question into an embedding.
2. Searches the indexed document chunks.
3. Retrieves the most relevant results.
4. Assembles the retrieved content into context.
5. Builds a grounded prompt.
6. Sends the prompt to the configured local Ollama model.
7. Returns the generated answer together with structured sources.

This allows Atlas to separate retrieval from generation and expose the evidence used by the answer.

## Supported Documents

The current application accepts:

* PDF
* DOCX
* PPTX
* TXT
* PNG
* JPG
* JPEG

Image documents can use the configured OCR pipeline for text extraction.

## Semantic Search

Atlas supports natural-language semantic search across indexed documents.

Search results expose information including:

* Document filename
* Page number
* Similarity score
* Retrieved text
* Chunk ID
* Document ID
* Embedding ID
* Chunk type
* Embedding model

A minimum similarity threshold can also be configured from the application interface.

## Document Management

The Document Library provides information about indexed documents, including:

* Filename
* Document type
* Page count
* File size
* Title
* Author

Documents can be removed from the Atlas index through the application. Deletion requires explicit confirmation before the operation is performed.

## Local AI

Atlas uses Ollama as its local LLM provider.

The exact model is configured through the project's RAG/LLM configuration rather than being hard-coded into the user interface.

This keeps the generation layer independent from the retrieval system.

## Installation

Atlas requires Python 3.11 or newer within the supported Python range declared in `pyproject.toml`.

Clone the repository and create a virtual environment:

```powershell
git clone <repository-url>
cd Atlas

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install Atlas with its development dependencies:

```powershell
python -m pip install -e ".[dev]"
```

Atlas also requires a working Ollama installation and a locally available model configured for the project.

## Running Atlas

Start the Streamlit application:

```powershell
streamlit run .\app.py
```

Then open the local Streamlit address shown in the terminal.

## Testing

Atlas has a dedicated automated test suite covering its core services and infrastructure.

Run the complete suite with:

```powershell
pytest -q
```

The project also contains a full RAG evaluation:

```powershell
python -m tests.test_rag_full_evaluation
```

The current evaluation covers:

* Top-1 retrieval
* Top-3 retrieval
* Groundedness
* Out-of-context handling

The current evaluation run achieved:

```text
Top-1 Retrieval           5/5
Top-3 Retrieval           5/5
Groundedness              5/5
Out-of-Context Handling   3/3

Overall                   18/18
Overall Accuracy          100%
```

These figures describe the project's current controlled evaluation set and should not be interpreted as a universal accuracy guarantee.

## Project Structure

```text
Atlas/
│
├── app.py
├── pyproject.toml
├── requirements.txt
├── README.md
│
├── config/
│   ├── llm.py
│   └── rag.py
│
├── core/
│   ├── config/
│   ├── constants/
│   ├── exceptions/
│   ├── logging/
│   └── utils/
│
├── domain/
│   ├── document/
│   ├── embeddings/
│   ├── factories/
│   ├── graph/
│   ├── llm/
│   ├── quiz/
│   ├── reports/
│   └── retrieval/
│
├── infrastructure/
│   ├── database/
│   ├── embeddings/
│   ├── llm/
│   ├── ocr/
│   ├── parsers/
│   ├── registry/
│   ├── storage/
│   └── vector_db/
│
├── services/
│   ├── chunk_service.py
│   ├── context_assembler.py
│   ├── document_service.py
│   ├── embedding_service.py
│   ├── llm_service.py
│   ├── prompt_builder.py
│   ├── rag_factory.py
│   ├── rag_service.py
│   ├── search_service.py
│   ├── storage_service.py
│   └── vector_search_service.py
│
├── scripts/
│
└── tests/
```

## Design Principles

Atlas is built around several principles:

### Local-first

Documents, embeddings, retrieval, and LLM generation are designed around a local execution model.

### Grounded generation

The RAG system retrieves document content before generating an answer rather than treating the LLM as the knowledge source.

### Separation of concerns

Document parsing, domain models, persistence, embeddings, retrieval, prompting, and LLM generation are separated into different layers.

### Testability

Core components have dedicated tests, while the complete RAG pipeline has a separate end-to-end evaluation.

### Inspectability

Search results and generated answers expose their retrieved sources so the user can inspect the information used by the pipeline.

## Current Status

Atlas has completed its core engineering and release-validation phase.

Current validated milestones:

* Document ingestion — complete
* Document chunking — complete
* Embedding generation — complete
* Database persistence — complete
* Document deduplication — complete
* Semantic search — complete
* RAG pipeline — complete
* Local Ollama integration — complete
* Grounded answer generation — validated
* Document deletion — validated
* Delete confirmation UI — validated
* Automated test suite — passing
* Fresh-environment installation — validated
* Full RAG evaluation — 100%

## Limitations

Atlas is currently a project release rather than a production enterprise knowledge platform.

The evaluation dataset is controlled and relatively small, so the reported 100% score should not be interpreted as a guarantee of perfect retrieval or answer quality for arbitrary documents.

LLM output quality also depends on the configured local Ollama model and the quality of the retrieved context.

## Roadmap

Future development may include:

* Improved document management
* More advanced retrieval strategies
* Additional document formats
* Better evaluation datasets
* Retrieval and generation performance improvements
* Expanded knowledge-management features
* Additional user-interface improvements

## License

Atlas is released under the MIT License.

## Project Status

**Atlas v0.1.0 — Release Candidate**

The core RAG system is operational and has passed the current automated and end-to-end evaluation gates.
