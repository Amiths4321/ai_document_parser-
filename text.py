#!/usr/bin/env python3
"""
doc_parser.py — A basic document parser that extracts text from PDFs
and splits it into chunks suitable for downstream use (e.g. embeddings,
RAG pipelines, search indexing).

Usage:
    python doc_parser.py path/to/file.pdf
    python doc_parser.py path/to/file.pdf --chunk-size 1000 --overlap 200
    python doc_parser.py path/to/file.pdf --by words --chunk-size 300
    python doc_parser.py path/to/file.pdf --output chunks.json

Requires: pypdf (pip install pypdf)
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

from pypdf import PdfReader


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------

@dataclass
class PageText:
    page_number: int
    text: str


def extract_pages(pdf_path: str) -> List[PageText]:
    """Extract raw text from each page of a PDF."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        cleaned = _clean_text(raw)
        pages.append(PageText(page_number=i, text=cleaned))
    return pages


def _clean_text(text: str) -> str:
    """Normalize whitespace and fix common PDF extraction artifacts."""
    # Collapse hyphenated line-breaks: "exam-\nple" -> "example"
    text = re.sub(r"-\n(?=\w)", "", text)
    # Collapse remaining newlines into spaces
    text = re.sub(r"\s*\n\s*", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------

@dataclass
class Chunk:
    chunk_id: int
    text: str
    start_page: int
    end_page: int
    char_count: int


def chunk_text(
    pages: List[PageText],
    chunk_size: int = 1000,
    overlap: int = 150,
    by: str = "chars",
) -> List[Chunk]:
    """
    Merge all page text into one stream (tracking page boundaries),
    then split into overlapping chunks.

    by="chars": chunk_size/overlap measured in characters.
    by="words": chunk_size/overlap measured in words.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    # Build a flat token stream with page markers so we can report
    # which page(s) each chunk came from.
    units = []  # list of (unit_text, page_number)
    for p in pages:
        if by == "words":
            for w in p.text.split():
                units.append((w, p.page_number))
        else:
            for ch in p.text:
                units.append((ch, p.page_number))
            if p.text:
                units.append((" ", p.page_number))  # page separator

    chunks = []
    step = chunk_size - overlap
    i = 0
    chunk_id = 0
    joiner = " " if by == "words" else ""

    while i < len(units):
        window = units[i:i + chunk_size]
        if not window:
            break
        text = joiner.join(u[0] for u in window).strip()
        if text:
            pages_in_chunk = [u[1] for u in window]
            chunks.append(Chunk(
                chunk_id=chunk_id,
                text=text,
                start_page=min(pages_in_chunk),
                end_page=max(pages_in_chunk),
                char_count=len(text),
            ))
            chunk_id += 1
        i += step

    return chunks


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract and chunk text from a PDF.")
    parser.add_argument("pdf_path", help="Path to the input PDF file")
    parser.add_argument("--chunk-size", type=int, default=1000,
                         help="Chunk size (chars or words depending on --by). Default: 1000")
    parser.add_argument("--overlap", type=int, default=150,
                         help="Overlap between consecutive chunks. Default: 150")
    parser.add_argument("--by", choices=["chars", "words"], default="chars",
                         help="Unit for chunk-size/overlap. Default: chars")
    parser.add_argument("--output", default=None,
                         help="Optional path to write chunks as JSON")
    args = parser.parse_args()

    if not Path(args.pdf_path).exists():
        print(f"Error: file not found: {args.pdf_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Extracting text from {args.pdf_path} ...")
    pages = extract_pages(args.pdf_path)
    total_chars = sum(len(p.text) for p in pages)
    print(f"Extracted {len(pages)} pages, {total_chars} characters.")

    chunks = chunk_text(pages, chunk_size=args.chunk_size,
                         overlap=args.overlap, by=args.by)
    print(f"Produced {len(chunks)} chunks (chunk_size={args.chunk_size}, "
          f"overlap={args.overlap}, by={args.by}).")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in chunks], f, indent=2, ensure_ascii=False)
        print(f"Wrote chunks to {args.output}")
    else:
        for c in chunks[:3]:
            preview = c.text[:150] + ("..." if len(c.text) > 150 else "")
            print(f"\n--- Chunk {c.chunk_id} (pages {c.start_page}-{c.end_page}, "
                  f"{c.char_count} chars) ---\n{preview}")
        if len(chunks) > 3:
            print(f"\n... and {len(chunks) - 3} more chunks. Use --output to save all of them.")


if __name__ == "__main__":
    main()