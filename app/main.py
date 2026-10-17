"""Aerchain Quote Intelligence API: FastAPI + Groq."""
import csv, io, json, os
from pathlib import Path
from typing import Any
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI
from openpyxl import load_workbook
from docx import Document
from pypdf import PdfReader

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR.parent / "data"
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
FRONTEND_ORIGINS = [
    "https://aerchain.netlify.app",
    "https://magenta-cat-044395.netlify.app",
]
app = FastAPI(title="Aerchain Quote Intelligence API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS + ["http://localhost:5173", "http://127.0.0.1:5173", "null"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalystRequest(BaseModel):
    question: str
    comparison_data: list[dict[str, Any]] = Field(default_factory=list)
    assumptions: dict[str, Any] = Field(default_factory=dict)

def client() -> OpenAI:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise HTTPException(503, "AI is not configured. Add GROQ_API_KEY in Render environment variables.")
    return OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")

def ask_model(system: str, prompt: str, max_tokens: int = 1400) -> str:
    try:
        response = client().chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=max_tokens,
        )
        answer = response.choices[0].message.content
        if not answer:
            raise HTTPException(502, "The AI model returned an empty answer. Please retry.")
        return answer
    except HTTPException:
        raise
    except Exception as exc:
        msg = str(exc)
        if any(term in msg.lower() for term in ["rate_limit", "tokens per minute", "tpm"]):
            raise HTTPException(429, "Groq rate limit reached. Wait a minute and try again.")
        raise HTTPException(502, f"AI request failed: {msg[:300]}")

def read_upload(filename: str, raw: bytes) -> str:
    ext = Path(filename).suffix.lower()
    if ext in {".txt", ".md", ".json", ".csv"}:
        return raw.decode("utf-8-sig", errors="replace")
    if ext == ".xlsx":
        wb = load_workbook(io.BytesIO(raw), data_only=True, read_only=True)
        out = []
        for ws in wb.worksheets:
            out.append(f"SHEET: {ws.title}")
            for row in ws.iter_rows(values_only=True):
                values = [str(v) if v is not None else "" for v in row]
                if any(values): out.append(" | ".join(values))
        return "\n".join(out)
    if ext == ".docx":
        doc = Document(io.BytesIO(raw))
        out = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                out.append(" | ".join(c.text.replace("\n", " / ") for c in row.cells))
        return "\n".join(out)
    if ext == ".pdf":
        pdf = PdfReader(io.BytesIO(raw))
        return "\n\n".join(f"PAGE {i+1}:\n{p.extract_text() or '[No selectable text found]'}" for i, p in enumerate(pdf.pages))
    raise HTTPException(415, f"Unsupported file type: {ext}. Supported: TXT, CSV, MD, JSON, XLSX, DOCX, PDF.")

def compact_rows(rows: list[dict[str, Any]], question: str, max_chars: int = 7800):
    fields = ["line_item_id", "description", "specification", "required_unit", "required_quantity", "vendor_id", "vendor_name", "quoted", "quoted_unit", "quoted_unit_price", "currency", "discount_pct", "freight_pct", "lead_time_days", "quality_questionnaire_pass", "confidence", "notes", "source_document"]
    q = question.lower()
    if any(w in q for w in ["missing", "unquoted", "not quoted"]):
        priority = [r for r in rows if str(r.get("quoted", "")).lower() not in ("true", "1", "yes")]
    elif any(w in q for w in ["compare", "comparable", "unit", "currency", "risk", "verify", "award"]):
        priority = [r for r in rows if str(r.get("quoted_unit", "")).lower() != str(r.get("required_unit", "")).lower() or str(r.get("currency", "")).upper() not in ("INR", "") or str(r.get("confidence", "")).lower() in ("low", "medium") or bool(r.get("notes")) or str(r.get("quoted", "")).lower() not in ("true", "1", "yes")]
    else:
        priority = []
    ordered = priority + [r for r in rows if r not in priority]
    result, used = [], 0
    for row in ordered:
        item = {k: row.get(k) for k in fields if k in row}
        encoded = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
        if used + len(encoded) > max_chars: continue
        result.append(item); used += len(encoded) + 1
        if len(result) >= 55: break
    return result, max(0, len(rows) - len(result))

@app.get("/")
def root():
    return {"service": "Aerchain Quote Intelligence API", "status": "ok", "health": "/health", "sample_data": "/sample-data", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok", "provider": "Groq", "model": MODEL, "api_key_configured": bool(os.getenv("GROQ_API_KEY"))}

@app.get("/sample-data")
def sample_data():
    path = DATA_DIR / "vendor_quotes_150_rows.csv"
    if not path.exists():
        raise HTTPException(500, f"Sample data file not found at {path}. Check Render Root Directory and repository files.")
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

@app.post("/analyze")
def analyze(req: AnalystRequest):
    if not req.question.strip(): raise HTTPException(422, "Please provide a question.")
    rows, omitted = compact_rows(req.comparison_data, req.question)
    system = ("You are a cautious enterprise procurement analyst. Answer in clear Markdown using only the supplied data and buyer assumptions. Never invent quotes or treat missing prices as zero. Distinguish facts from assumptions. Do not claim currency/unit normalization has occurred unless the data proves it. Cite vendor names and line item IDs where possible. Do not decide the award; explain trade-offs. The supplied records may be a subset, so do not claim you reviewed omitted records.")
    prompt = f"QUESTION:\n{req.question}\n\nBUYER ASSUMPTIONS:\n{json.dumps(req.assumptions, ensure_ascii=False)}\n\nCOMPARISON DATA ({len(rows)} rows supplied):\n{json.dumps(rows, ensure_ascii=False, separators=(',', ':'))}\n\nIf data is insufficient for a definitive answer, state what must be checked."
    answer = ask_model(system, prompt, 1500)
    return {"answer": answer, "model": MODEL, "rows_received": len(req.comparison_data), "rows_analyzed": len(rows), "rows_omitted": omitted}

@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    filename = file.filename or "upload.txt"
    source = read_upload(filename, await file.read())
    if not source.strip(): raise HTTPException(422, "No readable text found in the uploaded file.")
    source = source[:12000]
    schema = {"vendor_name": "string|null", "currency": "string|null", "validity_days": "number|null", "payment_terms": "string|null", "freight_policy": "string|null", "discount_policy": "string|null", "line_items": [{"line_item_id": "string|null", "description": "string", "quoted_quantity": "number|null", "quoted_unit": "string|null", "unit_price": "number|null", "currency": "string|null", "lead_time_days": "number|null", "notes": "string|null", "evidence_quote": "string", "confidence": "high|medium|low"}], "unresolved_questions": ["string"]}
    prompt = f"Extract only facts present in this procurement quote. Never infer missing prices, quantities, units, or terms. Use null for unknown values, include a short evidence quote for each line item, and mark ambiguous values low confidence. Return JSON only matching this schema:\n{json.dumps(schema)}\n\nFILE: {filename}\nSOURCE:\n{source}"
    answer = ask_model("You accurately extract procurement quote data. Return one valid JSON object only, without Markdown fences.", prompt, 1800)
    try:
        result = json.loads(answer.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip())
    except Exception:
        raise HTTPException(502, "The AI model did not return valid JSON. Please retry.")
    result["_source_file"] = filename
    result["_extraction_model"] = MODEL
    result["_note"] = "Verify extracted values against source evidence before procurement use."
    return result
