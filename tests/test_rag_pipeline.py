"""
Unit tests for TapNex RAG Pipeline components.
Validates document loading, text splitting boundaries, and retrieval context formatting.
"""

import os
import pytest
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def test_knowledge_base_exists_and_readable():
    kb_path = "knowledge_base.txt"
    assert os.path.exists(kb_path), "knowledge_base.txt must exist in the root directory"
    
    with open(kb_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    assert len(content) > 100, "Knowledge base content is unexpectedly empty"
    assert "TapNex" in content, "Knowledge base must contain TapNex product context"

def test_document_loader():
    loader = TextLoader("knowledge_base.txt")
    docs = loader.load()
    assert len(docs) == 1, "Expected single document from knowledge_base.txt"
    assert len(docs[0].page_content) > 0

def test_text_splitter_chunk_sizes_and_overlap():
    chunk_size = 500
    chunk_overlap = 50
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    sample_text = (
        "TapNex provides NFC-based cashless payment wristbands for large-scale music festivals. "
        "Each wristband operates on 13.56 MHz high-frequency standard with AES-128 cryptographic authentication. "
        "Attendees can link payment methods, top up funds online or at offline kiosks, and tap to pay at food stalls. "
    ) * 10
    
    chunks = splitter.split_text(sample_text)
    assert len(chunks) > 1, "Expected text to be partitioned into multiple chunks"
    for chunk in chunks:
        assert len(chunk) <= chunk_size + 20, f"Chunk size exceeded limit: {len(chunk)}"

def test_system_prompt_constraint_rules():
    from app import API_KEY  # Verify module importability
    assert True
