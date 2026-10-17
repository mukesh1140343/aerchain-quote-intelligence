
"""
Aerchain Quote Intelligence API
Python 3.11+
Provider: Groq
"""

import os
import json
import csv
import io
from pathlib import Path
from typing import Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from openpyxl import load_workbook
from docx import Document
from pypdf import PdfReader


# --------------------------------------------------
# Configuration
# --------------------------------------------------

app = FastAPI(title="Aerchain Quote Intelligence API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "null"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
DATA_DIR = Path(__file__).parent.parent / "data"


def get_client():
    """Create a Groq client using the current environment key."""
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY is not configured in the server environment.",
        )

    return OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )


class AnalystRequest(BaseModel):
    question: str
    comparison_data: list[dict[str, Any]]
    assumptions: dict[str, Any] = {}


# --------------------------------------------------
# Document parsing
# --------------------------------------------------

def read_upload(filename: str, raw: bytes) -> str:
    suffix = Path(filename).suffix.lower()

    if suffix in {".txt", ".md", ".json"}:
        return raw.decode("utf-8", errors="replace")

    if suffix == ".csv":
        return raw.decode("utf-8-sig", errors="replace")

    if suffix == ".xlsx":
        workbook = load_workbook(
            io.BytesIO(raw),
            data_only=True,
            read_only=True,
        )
        chunks = []

        for worksheet in workbook.worksheets:
            chunks.append(f"SHEET: {worksheet.title}")

            for row in worksheet.iter_rows(values_only=True):
                values = [
                    str(value) if value is not None else ""
                    for value in row
                ]
                if any(values):
                    chunks.append(" | ".join(values))

        return "\n".join(chunks)

    if suffix == ".docx":
        document = Document(io.BytesIO(raw))
        chunks = [
            paragraph.text
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        ]

        for table in document.tables:
            for row in table.rows:
                chunks.append(
                    " | ".join(
                        cell.text.replace("\n", " / ")
                        for cell in row.cells
                    )
                )

        return "\n".join(chunks)

    if suffix == ".pdf":
        reader = PdfReader(io.BytesIO(raw))
        pages = []

        for index, page in enumerate(reader.pages):
            text = page.extract_text()
            if not text:
                text = (
                    "[No selectable text found; "
                    "this page may require OCR.]"
                )
            pages.append(f"PAGE {index + 1}:\n{text}")

        return "\n\n".join(pages)

    raise HTTPException(
        status_code=415,
        detail=(
            f"Unsupported file type: {suffix}. Supported formats: "
            "TXT, CSV, MD, JSON, XLSX, DOCX and text-based PDF."
        ),
    )


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "provider": "Groq",
        "model": MODEL,
        "api_key_configured": bool(os.getenv("GROQ_API_KEY")),
    }


# --------------------------------------------------
# Sample comparison data
# --------------------------------------------------

@app.get("/sample-data")
def sample_data():
    path = DATA_DIR / "vendor_quotes_150_rows.csv"

    if not path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Sample data file not found: {path}",
        )

    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


# --------------------------------------------------
# AI quote extraction
# --------------------------------------------------

@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    """Extract quote fields from a vendor response using Groq."""

    client = get_client()
    filename = file.filename or "upload.txt"
    raw = await file.read()

    source_text = read_upload(filename, raw)

    if not source_text.strip():
        raise HTTPException(
            status_code=422,
            detail="No readable text found in the uploaded file.",
        )

    # Limit unusually large documents for this prototype.
    if len(source_text) > 18000:
        source_text = (
            source_text[:18000]
            + "\n[Document truncated to fit prototype limits.]"
        )

    schema = {
        "vendor_name": "string|null",
        "currency": "string|null",
        "validity_days": "number|null",
        "payment_terms": "string|null",
        "freight_policy": "string|null",
        "discount_policy": "string|null",
        "line_items": [
            {
                "line_item_id": "string|null",
                "description": "string",
                "quoted_quantity": "number|null",
                "quoted_unit": "string|null",
                "unit_price": "number|null",
                "currency": "string|null",
                "lead_time_days": "number|null",
                "notes": "string|null",
                "evidence_quote": "string",
                "confidence": "high|medium|low",
            }
        ],
        "unresolved_questions": ["string"],
    }

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You extract vendor quote data for enterprise "
                        "procurement. Extract only facts in the source. "
                        "Never invent prices, quantities, units or terms. "
                        "Use null for unknown values. Include a short "
                        "verbatim evidence_quote for every extracted line. "
                        "Mark ambiguous or conflicting values as low "
                        "confidence. Do not fabricate missing line items. "
                        "Return only a valid JSON object matching the schema."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Return data matching this schema:\n"
                        f"{json.dumps(schema)}\n\n"
                        f"SOURCE FILE: {filename}\n\n"
                        f"SOURCE CONTENT:\n{source_text}"
                    ),
                },
            ],
            temperature=0,
            max_tokens=4000,
            response_format={"type": "json_object"},
        )

        result = json.loads(
            response.choices[0].message.content or "{}"
        )

    except Exception as error:
        print(f"Groq extraction error: {error}")
        raise HTTPException(
            status_code=502,
            detail=(
                "Groq extraction failed. Check the server Terminal "
                "for the technical error."
            ),
        )

    result["_source_file"] = filename
    result["_extraction_model"] = MODEL
    result["_note"] = (
        "Verify extracted values against the source before "
        "making procurement decisions."
    )

    return result


# --------------------------------------------------
# AI procurement analyst
# --------------------------------------------------

@app.post("/analyze")
def analyze(req: AnalystRequest):
    """Answer procurement questions using compact comparison data."""

    client = get_client()

    # Keep useful fields while reducing unnecessary prompt tokens.
    relevant_words = (
        "vendor", "supplier", "line", "item", "description",
        "price", "currency", "unit", "quote", "status",
        "missing", "review", "lead", "freight", "discount",
        "payment", "validity", "quality", "quantity", "term",
        "note", "confidence", "evidence", "risk", "score",
        "id",
    )

    compact_rows = []

    for row in req.comparison_data:
        compact = {
            key: value
            for key, value in row.items()
            if any(word in key.lower() for word in relevant_words)
            and value is not None
            and value != ""
        }

        if not compact:
            compact = dict(list(row.items())[:6])

        compact_rows.append(compact)

    # Fit the prompt within a modest token budget.
    # Preserve the original row order and report if rows are omitted.
    selected_rows = []
    max_data_chars = 14000

    for row in compact_rows:
        candidate = selected_rows + [row]
        encoded = json.dumps(
            candidate,
            ensure_ascii=False,
            separators=(",", ":"),
        )

        if len(encoded) > max_data_chars:
            break

        selected_rows.append(row)

    rows_omitted = len(compact_rows) - len(selected_rows)

    data_text = json.dumps(
        selected_rows,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    assumptions_text = json.dumps(
        req.assumptions,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    system_prompt = (
        "You are a cautious enterprise procurement analyst. "
        "Answer only from the supplied comparison data and approved "
        "assumptions. Never invent quotes or treat missing values as zero. "
        "Distinguish quoted facts from assumptions and calculations. "
        "Cite vendor and line item IDs when available. Flag missing quotes, "
        "currency mismatches, unit mismatches, freight uncertainty and "
        "quality risks. Do not decide the award for the buyer. "
        "If information is incomplete, state the limitation."
    )

    user_prompt = (
        f"BUYER QUESTION:\n{req.question}\n\n"
        f"COMPARISON DATA ({len(selected_rows)} rows supplied):\n"
        f"{data_text}\n\n"
        f"BUYER-APPROVED ASSUMPTIONS:\n{assumptions_text}\n\n"
        f"Rows omitted to meet request limits: {rows_omitted}. "
        "Do not make claims about omitted rows. Explain if the "
        "available subset is insufficient to answer fully."
    )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=1200,
        )

        answer = response.choices[0].message.content or ""

        return {
            "answer": answer,
            "provider": "Groq",
            "model": MODEL,
            "rows_received": len(req.comparison_data),
            "rows_analyzed": len(selected_rows),
            "rows_omitted": rows_omitted,
        }

    except Exception as error:
        print(f"Groq analyst error: {error}")
        raise HTTPException(
            status_code=502,
            detail=(
                "The AI analyst request failed. Check the server "
                "Terminal for the exact Groq error."
            ),
        )
