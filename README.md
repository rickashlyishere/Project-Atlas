# Project Atlas

**Offline AI Knowledge Platform**

Atlas is a local, document-grounded AI knowledge platform built around retrieval-augmented generation (RAG).

It allows users to import documents, index their contents, search them using natural language, and ask questions whose answers are generated from retrieved document context.

Atlas is designed around a **local-first architecture**. Document processing, embeddings, retrieval, and LLM inference can run locally without sending the user's documents to a cloud AI service.

---

## Features

### Document Ingestion

Atlas can import and index:

* PDF
* DOCX
* PPTX
* TXT
* PNG
* JPG
* JPEG

Documents are parsed, processed into chunks, embedded, and stored for later retrieval.

Image documents can use OCR for text extraction.

### Semantic Search

Atlas provides natural-language semantic search across indexed documents.

Search results include information such as:

* Document filename
* Page number
* Similarity score
* Retrieved text
* Document ID
* Chunk ID
* Embedding ID
* Chunk type
* Embedding model

Search can be performed across the entire document library or within a selected document.

### Retrieval-Augmented Generation

Atlas combines semantic retrieval with a local LLM.

When a user asks a question:

```text
Question
   │
   ▼
Query Embedding
   │
   ▼
Vector Search
   │
   ▼
Relevant Chunks
   │
   ▼
Context Assembly
   │
   ▼
Grounded Prompt
   │
   ▼
Local Ollama LLM
   │
   ▼
Answer + Sources
```

The generated response is accompanied by the sources retrieved from the indexed documents.

### Document Library

Atlas maintains a local document library containing information such as:

* Filename
* Document type
* Page count
* File size
* Title
* Author

Documents can be removed from the library through the application with explicit deletion confirmation.

---

# Architecture

Atlas separates the application into several layers.

```text
                         ┌────────────────────┐
                         │     Streamlit UI   │
                         └─────────┬──────────┘
                                   │
                                   ▼
                         ┌────────────────────┐
                         │     Services       │
                         │                    │
                         │ Document Service   │
                         │ Search Service     │
                         │ RAG Service        │
                         │ Embedding Service  │
                         │ LLM Service        │
                         └─────────┬──────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    │              │              │
                    ▼              ▼              ▼
             ┌────────────┐ ┌────────────┐ ┌────────────┐
             │  Domain    │ │Infrastructure│ │ Database  │
             │   Layer    │ │   Layer      │ │ Repository│
             └────────────┘ └──────┬───────┘ └────────────┘
                                   │
                  ┌────────────────┼────────────────┐
                  │                │                │
                  ▼                ▼                ▼
             Embeddings         Parsers          Ollama
                  │                │                │
                  └────────────────┼────────────────┘
                                   │
                                   ▼
                              RAG Pipeline
```

The repository is organized around separation of concerns between:

* Application services
* Domain models and interfaces
* Database persistence
* Document parsers
* Embedding providers
* OCR
* Local LLM integration
* Retrieval
* Configuration
* Testing

---

# RAG Pipeline

Atlas uses a retrieval-first workflow.

When a question is submitted:

1. The question is converted into an embedding.
2. Atlas searches the indexed document embeddings.
3. Relevant chunks are retrieved.
4. The retrieved chunks are assembled into context.
5. A grounded prompt is constructed.
6. The prompt is sent to the configured local Ollama model.
7. The generated answer is returned.
8. Retrieved sources are displayed alongside the answer.

This separates **retrieval** from **generation** and allows the user to inspect the information used to construct the answer.

---

# Local LLM

Atlas currently uses **Ollama** for local LLM inference.

The default configuration is:

```text
Ollama endpoint:
http://127.0.0.1:11434

Model:
llama3.2

Timeout:
300 seconds
```

RAG configuration:

```text
Maximum sources:
5

Maximum context:
12,000 characters

Minimum similarity score:
0.35
```

These values are defined in:

```text
config/llm.py
config/rag.py
```

---

# Installation

## Prerequisites

Before installing Atlas, install:

* Git
* Python 3.11
* Ollama
* The `llama3.2` Ollama model

Atlas currently targets Python 3.11.

---

## 1. Install Git

Install Git for your operating system.

Verify the installation:

```powershell
git --version
```

---

## 2. Install Python 3.11

Install Python 3.11 and make sure it is available through the Python launcher.

Verify:

```powershell
py -3.11 --version
```

You should see a Python 3.11 version.

---

## 3. Install Ollama

Download and install Ollama from the official website:

https://ollama.com/download

After installation, open a new PowerShell window.

Verify Ollama:

```powershell
ollama --version
```

---

## 4. Pull the Atlas LLM

Atlas currently uses:

```text
llama3.2
```

Pull the model:

```powershell
ollama pull llama3.2
```

Verify that it is installed:

```powershell
ollama list
```

You should see `llama3.2` in the model list.

You can optionally test the model directly:

```powershell
ollama run llama3.2
```

Exit the interactive session when finished.

Atlas connects to Ollama through:

```text
http://127.0.0.1:11434
```

---

# Installing Atlas

## 5. Clone the Repository

Clone Atlas:

```powershell
git clone https://github.com/rickashlyishere/Project-Atlas.git
```

Enter the project directory:

```powershell
cd Atlas
```

---

## 6. Create a Virtual Environment

Create a Python 3.11 virtual environment:

```powershell
py -3.11 -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Your PowerShell prompt should now look similar to:

```text
(.venv) PS C:\...\Atlas>
```

---

## 7. Install Atlas

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

Install Atlas and its development dependencies:

```powershell
python -m pip install -e ".[dev]"
```

Atlas uses `pyproject.toml` as its primary package configuration.

---

# Verify the Installation

Run the complete automated test suite:

```powershell
pytest -q
```

A successful installation should result in the complete test suite passing.

You can also run the full RAG evaluation:

```powershell
python -m tests.test_rag_full_evaluation
```

This validates the retrieval and grounded-answer pipeline against the project's controlled evaluation set.

---

# Running Atlas

Start the Streamlit application:

```powershell
streamlit run .\app.py
```

Streamlit will display the local address in the terminal.

Open that address in your browser.

---

# Using Atlas

## Import a Document

Open:

```text
Import Document
```

Select one of the supported document formats.

Atlas will:

```text
Document
   ↓
Parser
   ↓
Text Extraction
   ↓
Chunking
   ↓
Embedding Generation
   ↓
Database Storage
   ↓
Searchable Knowledge
```

After processing, Atlas displays information about the imported document and the number of indexed chunks.

---

## Ask Atlas

Open:

```text
Ask Atlas
```

Enter a question related to your indexed documents.

For example:

```text
What does this document say about triangles?
```

Then select the number of sources to retrieve and click:

```text
Ask Atlas
```

Atlas retrieves relevant document chunks and passes the resulting context to the configured local Ollama model.

The interface displays:

* Generated answer
* Source documents
* Page numbers
* Similarity scores
* Retrieved source text

---

# Semantic Search

Atlas also provides semantic search independently of answer generation.

Open:

```text
Semantic Search
```

Enter a natural-language query.

You can choose:

* All documents
* A specific document

You can also configure:

* Number of results
* Minimum similarity score

Search results expose:

```text
Document
Page
Similarity
Text
Chunk metadata
```

---

# Document Library

The Document Library displays indexed documents.

Each document can expose:

* Filename
* Document type
* Page count
* File size
* Title
* Author

Documents can be deleted from Atlas.

Deletion uses an explicit confirmation step to prevent accidental removal.

---

# Supported File Types

Atlas currently accepts:

```text
.pdf
.docx
.pptx
.txt
.png
.jpg
.jpeg
```

Image files can be processed through the OCR pipeline.

---

# Project Structure

```text
Atlas/
│
├── app.py
├── pyproject.toml
├── requirements.txt
├── README.md
├── LICENSE
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
│   ├── cache/
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
│   ├── atlas.py
│   └── cli/
│
└── tests/
    ├── test_chunk_repository.py
    ├── test_chunk_service.py
    ├── test_context_assembler.py
    ├── test_document.py
    ├── test_document_deduplication.py
    ├── test_document_delete_contract.py
    ├── test_document_ingestion.py
    ├── test_document_lifecycle.py
    ├── test_embedding_repository.py
    ├── test_embedding_service.py
    ├── test_llm_service.py
    ├── test_ollama_provider.py
    ├── test_prompt_builder.py
    ├── test_rag_config.py
    ├── test_rag_evaluation.py
    ├── test_rag_full_evaluation.py
    ├── test_rag_groundedness.py
    ├── test_rag_service.py
    ├── test_real_rag.py
    ├── test_search_deduplication.py
    ├── test_search_service.py
    └── test_vector_search_service.py
```

---

# Testing

Atlas contains unit, integration, and end-to-end tests covering its core components.

Run everything:

```powershell
pytest -q
```

Run the complete RAG evaluation:

```powershell
python -m tests.test_rag_full_evaluation
```

The release validation process also includes testing Atlas from a fresh Python environment using the project's `pyproject.toml`.

---

# RAG Evaluation

The current controlled RAG evaluation covers:

* Top-1 retrieval
* Top-3 retrieval
* Groundedness
* Out-of-context handling

The current evaluation result is:

```text
Top-1 Retrieval           5/5
Top-3 Retrieval           5/5
Groundedness              5/5
Out-of-Context Handling   3/3
--------------------------------
Overall                  18/18
Accuracy                 100%
```

The evaluation was run against the project's controlled test set.

**The 100% evaluation result should not be interpreted as a guarantee of perfect accuracy on arbitrary documents or questions.**

Real-world performance depends on factors including:

* Document quality
* Chunking
* Embedding quality
* Retrieval quality
* Question complexity
* Ollama model behavior
* Available context

---

# Design Principles

## Local First

Atlas is designed around local document processing, local embeddings, local retrieval, and local LLM inference.

The default LLM provider is Ollama.

## Grounded Generation

Atlas retrieves document content before generating an answer.

The retrieved context becomes the evidence supplied to the LLM.

## Separation of Concerns

The project separates:

* Domain logic
* Application services
* Persistence
* Document parsing
* Embeddings
* Retrieval
* Prompt construction
* LLM integration
* User interface

## Testability

Atlas contains dedicated tests for individual components as well as complete RAG evaluation tests.

## Inspectability

Atlas exposes retrieved sources and similarity information so users can inspect the material supporting an answer.

---

# Configuration

The main LLM configuration is located at:

```text
config/llm.py
```

Current configuration:

```python
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL = "llama3.2"
OLLAMA_TIMEOUT = 300.0
```

RAG configuration is located at:

```text
config/rag.py
```

Current configuration:

```python
RAG_MAX_SOURCES = 5
RAG_MAX_CONTEXT_CHARACTERS = 12000
RAG_MINIMUM_SCORE = 0.35
```

---

# Troubleshooting

## `ollama` is not recognized

Close and reopen PowerShell after installing Ollama.

Then run:

```powershell
ollama --version
```

If it still cannot be found, verify that Ollama was installed correctly.

---

## `llama3.2` is missing

Check installed models:

```powershell
ollama list
```

If `llama3.2` is missing:

```powershell
ollama pull llama3.2
```

Then verify again:

```powershell
ollama list
```

---

## Atlas cannot connect to Ollama

Verify Ollama is running.

The Atlas configuration expects:

```text
http://127.0.0.1:11434
```

You can also test the model directly:

```powershell
ollama run llama3.2
```

---

## Python environment problems

Make sure the Atlas virtual environment is active:

```powershell
.\.venv\Scripts\Activate.ps1
```

Your terminal should show:

```text
(.venv)
```

Then reinstall Atlas:

```powershell
python -m pip install -e ".[dev]"
```

---

## Tests fail after installation

Run:

```powershell
pytest -q
```

If the problem involves the RAG pipeline, verify that Ollama is running and that `llama3.2` is installed:

```powershell
ollama list
```

---

# Release Status

## Atlas v0.1.0

Atlas v0.1.0 represents the first public release of the core Atlas RAG platform.

Validated release gates include:

```text
Document ingestion                 PASS
Document deduplication             PASS
Document lifecycle                 PASS
Embedding generation               PASS
Semantic search                    PASS
Vector search                      PASS
RAG pipeline                       PASS
Groundedness evaluation            PASS
Out-of-context handling            PASS
Document deletion                  PASS
Delete confirmation UI             PASS
Automated test suite               PASS
Fresh environment installation     PASS
Full RAG evaluation                100%
```

---

# Limitations

Atlas v0.1.0 is a public project release and should not be considered a production enterprise knowledge-management system.

The current RAG evaluation uses a controlled dataset. Its results do not establish universal model accuracy.

LLM responses depend on the configured Ollama model and the quality of retrieved context.

Large documents and large collections may require additional performance optimization as the project scales.

---

# Roadmap

Potential future development includes:

* Improved retrieval strategies
* Expanded evaluation datasets
* More document formats
* Retrieval performance improvements
* Better document management
* Additional knowledge-management features
* Improved user-interface workflows
* More advanced RAG evaluation
* Additional local model support

---
# Questions & Support

Have a question, found a problem, or want to discuss Atlas?

Feel free to reach out by email:

**Email:** [kammarithvik2@gmail.com](mailto:kammarithvik2@gmail.com)

You can contact me for:

* Questions about installing Atlas
* Ollama or model configuration issues
* Questions about the RAG pipeline
* Bug reports
* Suggestions and feature ideas
* General questions about the project

When reporting a problem, including the error message, operating system, Python version, and steps to reproduce it will make troubleshooting much easier.

I may not be able to respond immediately, but I'll do my best to help.

# License

Atlas is released under the **MIT License**.

See the `LICENSE` file for the complete license text.

---

# Project Status

**Atlas v0.1.0 — First Public Release**

Atlas's core document ingestion, semantic retrieval, and grounded local RAG pipeline are operational and have passed the current release validation suite.

The project is now ready for public experimentation, feedback, and further development.
