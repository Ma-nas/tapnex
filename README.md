# TapNex AI Customer Support Agent

### Retrieval-Augmented Generation (RAG) Support Platform for Event Technology and NFC Systems

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-Core-1C3C3C?style=flat-square&logo=langchain&logoColor=white)](https://www.langchain.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![FAISS](https://img.shields.io/badge/Vector_DB-FAISS-00599C?style=flat-square)](https://github.com/facebookresearch/faiss)
[![Embeddings](https://img.shields.io/badge/Embeddings-all--MiniLM--L6--v2-orange?style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[![LLM](https://img.shields.io/badge/Inference-Groq_Llama_3-F55036?style=flat-square)](https://groq.com)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)

[Overview](#overview) | [RAG Architecture](#rag-architecture) | [Technical Specifications](#technical-specifications) | [Setup & Execution](#setup--execution) | [Grounding Guarantees](#grounding--hallucination-prevention)

---

## Overview

The TapNex AI Customer Support Agent is an intelligent domain-specific question-answering system built on a Retrieval-Augmented Generation (RAG) framework. The agent serves event organizers, production engineers, and attendees seeking technical and operational guidance regarding TapNex's NFC-based cashless payment infrastructure, access control devices, wristband management, and ticketing workflows.

By strictly anchoring responses within the company's verified technical documentation (`knowledge_base.txt`), the system provides high-precision answers while actively suppressing out-of-domain hallucinations.

---

## RAG Architecture

```
+-------------------------------------------------------------------------+
|                  Offline Indexing & Document Ingestion                  |
|                                                                         |
|  [knowledge_base.txt]                                                   |
|          |                                                              |
|          v                                                              |
|  [LangChain TextLoader]                                                 |
|          |                                                              |
|          v                                                              |
|  [RecursiveCharacterTextSplitter] (Chunk: 500 chars, Overlap: 50 chars) |
|          |                                                              |
|          v                                                              |
|  [HuggingFace Embeddings: all-MiniLM-L6-v2] (Local CPU/GPU inference)  |
|          |                                                              |
|          v                                                              |
|  [FAISS Vector Index] (In-memory similarity lookup)                     |
+-------------------------------------------------------------------------+

+-------------------------------------------------------------------------+
|                      Online Query Execution Flow                        |
|                                                                         |
|  User Query (Streamlit Chat UI)                                         |
|          |                                                              |
|          v                                                              |
|  [Embedding Computation] -> [FAISS Similarity Search (Top-k = 3)]       |
|          |                                                              |
|          v                                                              |
|  [Context Aggregation & Prompt Injection]                               |
|          |                                                              |
|          v                                                              |
|  [Groq Inference Engine: Llama-3-8B-8192]                              |
|  (Conditioned with strict negative-constraint system prompt)            |
|          |                                                              |
|          v                                                              |
|  Grounded Output / Structured Fallback Response                         |
+-------------------------------------------------------------------------+
```

---

## Technical Specifications

| Component | Implementation | Rationale |
| :--- | :--- | :--- |
| **User Interface** | Streamlit | Lightweight reactive web interface supporting interactive conversational turns |
| **Pipeline Framework** | LangChain Core | Manages document parsing, vector indexing, prompt templates, and output parsing |
| **Dense Vector Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | Executes locally via HuggingFace embeddings; zero external API latency or costs |
| **Vector Store** | FAISS (Facebook AI Similarity Search) | High-performance in-memory index executing exact L2/cosine similarity scans |
| **Inference Hardware** | Groq LPU (Language Processing Unit) | Sub-second token generation using `llama3-8b-8192` for conversational responsiveness |
| **Chunking Strategy** | Recursive Character Splitter | 500 character chunks with 50 character overlap to preserve semantic continuity |

---

## Setup & Execution

### Prerequisites

- Python 3.10 or higher
- A Groq API key from [Groq Console](https://console.groq.com/)

### 1. Repository Setup

```bash
git clone https://github.com/Ma-nas/tapnex.git
cd tapnex
```

### 2. Environment Configuration

```bash
# Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
```

Edit `.env` and configure your credentials:

```dotenv
GROQ_API_KEY=gsk_your_groq_api_key_here
```

### 3. Application Launch

```bash
streamlit run app.py
```

The Streamlit interface opens automatically in your default browser at `http://localhost:8501`.

---

## Grounding & Hallucination Prevention

To prevent generative drift and ensure reliability in mission-critical event operations, the system prompt strictly constrains the reasoning space:

- The model is instructed to answer queries based **strictly and exclusively** on the top-3 retrieved passages from `knowledge_base.txt`.
- For queries falling outside documented topics (e.g., questions regarding external platforms, hypothetical scenarios, or general trivia), the agent deterministically returns:
  > *"I'm sorry, I couldn't find that information in the TapNex documentation."*
- Numerical assertions such as offline refund windows, NFC frequency standards, and transaction limits are preserved directly from authoritative source documentation.

---

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for terms.
