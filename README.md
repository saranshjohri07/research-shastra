# ResearchShastra

ResearchShastra is an AI-powered research assistant designed to help users understand and interact with research papers through document ingestion, semantic retrieval, and LLM-powered responses.

## Current Status

Foundation setup — V1 development has not started yet.

## Planned V1 Architecture

PDF → Text Extraction → Section-Aware Chunking → Embeddings → Chroma → Retrieval → Groq LLM

## Development Principles

* Keep infrastructure configurable and replaceable.
* Evaluate retrieval components on the actual research-paper dataset.
* Separate foundation setup from research and optimization.
* Keep secrets out of version control.