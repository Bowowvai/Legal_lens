#!/usr/bin/env python3
"""
Test script to check Legal BERT installation and functionality
"""
from fastapi import FastAPI, UploadFile, File, Form, Header
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os
import io
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()  # Load variables from .env if present

# Lightweight extractive + abstractive hybrid using transformers (optional) or pure extractive fallback.
TRANSFORMER_ERROR = None
LEGAL_BERT_ERROR = None
LEGAL_MODEL_NAME = None

try:
    # First import the base transformers module
    import transformers

    # Enable more verbose output for debugging
    transformers.logging.set_verbosity_info()

    # Then import specific components
    from transformers import AutoTokenizer, AutoModel

    # Import pipeline separately as it's the one causing issues
    try:
        from transformers import pipeline
        HAS_PIPELINE = True
    except ImportError as e:
        print(f"⚠️ Transformers pipeline not available: {e}")
        print("💡 Will use alternative summarization methods")
        HAS_PIPELINE = False

    HAS_TRANSFORMERS = True
    print("✅ Transformers imported successfully")
except Exception as e:
    HAS_TRANSFORMERS = False
    HAS_LEGAL_BERT = False
    TRANSFORMER_ERROR = str(e)
    print(f"❌ Transformers failed to import: {e}")

# Legal BERT loading using our working implementation
LEGAL_MODEL_NAME = None
if HAS_TRANSFORMERS:
    if os.environ.get("LEGAL_BERT_OPT_IN", "1") == "0":
        HAS_LEGAL_BERT = False
        LEGAL_BERT_ERROR = "Disabled by env LEGAL_BERT_OPT_IN=0"
        print("ℹ️ Skipping LegalBERT load due to env LEGAL_BERT_OPT_IN=0")
    else:
        try:
            print("📥 Loading LegalBERT model using working implementation...")
            
            # Import our working Legal BERT implementation
            from legal_bert_working import LegalBERTAnalyzer
            
            # Initialize the analyzer
            legal_bert_analyzer = LegalBERTAnalyzer()
            HAS_LEGAL_BERT = legal_bert_analyzer.is_loaded
            LEGAL_MODEL_NAME = legal_bert_analyzer.model_name
            
            if HAS_LEGAL_BERT:
                print(f"✅ Legal BERT model loaded successfully: {LEGAL_MODEL_NAME}")
            else:
                LEGAL_BERT_ERROR = "Model loading failed, using rule-based fallback"
                print("⚠️ Legal BERT model loading failed, continuing with rule-based analysis")
                
        except Exception as e:
            HAS_LEGAL_BERT = False
            LEGAL_BERT_ERROR = str(e)
            print(f"❌ Legal BERT initialization error: {e}")
            print("🔄 Continuing without legal model")
else:
    HAS_LEGAL_BERT = False
    LEGAL_BERT_ERROR = "Transformers not available"

app = FastAPI(title="Themis Summarizer API")

# CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

summarizer = None

# Initialize chatbot (OpenAI-compatible); optional dependency
try:
    from chatbot_service import ChatbotService
    CHATBOT = ChatbotService()
    HAS_CHATBOT = CHATBOT.available
except Exception:
    CHATBOT = None
    HAS_CHATBOT = False

def get_summarizer():
    global summarizer
    if summarizer is not None:
        return summarizer
    
    if HAS_TRANSFORMERS and HAS_PIPELINE:
        try:
            # Small model to keep memory reasonable; you can change this later.
            summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
            print("✅ Summarization pipeline created successfully")
        except Exception as e:
            print(f"❌ Error creating summarization pipeline: {e}")
            print("💡 Will use fallback summarization method")
            summarizer = None
    else:
        print("⚠️ Transformers pipeline not available, using fallback summarization")
        summarizer = None
    
    return summarizer

class SummarizeTextIn(BaseModel):
    text: str
    sentences: Optional[int] = 6

@app.post("/summarize-text")
async def summarize_text(payload: SummarizeTextIn):
    text = (payload.text or "").strip()
    if not text:
        return {"summary": ""}

    # If the text is long, chunk it.
    def chunk_by_sentences(t: str, target: int = 1800, max_len: int = 2400):
        import re
        sents = re.split(r"(?<=[.!?])\s+", " ".join(t.split()))
        chunks, cur = [], ""
        for s in sents:
            if len(cur + " " + s) > max_len:
                if cur:
                    chunks.append(cur.strip())
                cur = s
            elif len(cur + " " + s) > target:
                cur = (cur + " " + s).strip()
                chunks.append(cur)
                cur = ""
            else:
                cur = (cur + " " + s).strip()
        if cur:
            chunks.append(cur)
        return chunks or [t]

    chunks = chunk_by_sentences(text)

    smz = get_summarizer()
    if smz is None:
        # Extractive fallback: take top longest sentences
        import re
        sents = re.split(r"(?<=[.!?])\s+", text)
        sents = [s for s in sents if len(s.split()) > 4]
        best = sorted(sents, key=lambda s: min(len(s), 200), reverse=True)[: payload.sentences or 6]
        return {"summary": " ".join(best)}

    partial = []
    for c in chunks:
        out = smz(c, max_length=180, min_length=60, do_sample=False)
        partial.append(out[0]["summary_text"])  # type: ignore
    if len(partial) == 1:
        return {"summary": partial[0]}
    joined = " ".join(partial)
    out = smz(joined, max_length=200, min_length=80, do_sample=False)
    return {"summary": out[0]["summary_text"]}

@app.get("/health")
async def health():
    """Health check endpoint with detailed status of all components"""
    return {
        "status": "ok",
        "gemini": bool(os.environ.get("GEMINI_API_KEY")),
        "ocr_available": HAS_OCR_DEPS,
        "pymupdf": 'HAS_PYMUPDF' in globals() and HAS_PYMUPDF,
        "tesseract": 'HAS_TESSERACT' in globals() and HAS_TESSERACT,
        "transformers": HAS_TRANSFORMERS,
        "pipeline": HAS_PIPELINE if "HAS_PIPELINE" in globals() else False,
        "legal_bert": HAS_LEGAL_BERT,
        "legal_model_name": LEGAL_MODEL_NAME,
    "chatbot": bool(HAS_CHATBOT),
        "transformers_error": TRANSFORMER_ERROR,
        "legal_bert_error": LEGAL_BERT_ERROR,
    }

@app.get("/chatbot/health")
async def chatbot_health():
    return {"chatbot": bool(HAS_CHATBOT)}

# -----------------------------
# Contract-style text analyzer
# -----------------------------
import re
from datetime import datetime

def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def extract_contract_fields(text: str):
    t = text or ""
    fields = {
        "document_type": None,
        "effective_date": None,
        "borrower": None,
        "lender": None,
        "borrower_address": None,
        "lender_address": None,
        "start_date_first_payment": None,
        "end_date_last_payment": None,
        "loan_amount": None,
        "interest_rate": None,
        "late_fee": None,
        "payment_method": None,
    }

    # Enhanced document type detection using legal BERT if available
    if HAS_LEGAL_BERT:
        # Use our working Legal BERT implementation for enhanced analysis
        try:
            # Get comprehensive analysis from Legal BERT
            analysis = legal_bert_analyzer.analyze_document(t)
            
            # Extract document type from analysis
            if analysis.get('document_type', {}).get('specific_type'):
                fields["document_type"] = analysis['document_type']['specific_type']
            
            # Extract additional entities if available
            entities = analysis.get('key_entities', {})
            if entities.get('parties'):
                # Use first few parties as borrower/lender if not already set
                if not fields.get('borrower') and len(entities['parties']) > 0:
                    fields['borrower'] = entities['parties'][0]
                if not fields.get('lender') and len(entities['parties']) > 1:
                    fields['lender'] = entities['parties'][1]
            
            # Extract amounts if available
            if entities.get('amounts') and not fields.get('loan_amount'):
                # Look for the largest amount as loan amount
                amounts = []
                for amount_str in entities['amounts']:
                    try:
                        # Extract numeric value from amount string
                        import re
                        num_match = re.search(r'[\d,]+\.?\d*', amount_str.replace('$', '').replace(',', ''))
                        if num_match:
                            amounts.append(float(num_match.group().replace(',', '')))
                    except:
                        continue
                
                if amounts:
                    max_amount = max(amounts)
                    fields['loan_amount'] = f"${max_amount:,.2f}"
                    
        except Exception as e:
            print(f"⚠️ Legal BERT analysis failed: {e}")
            # Continue with fallback approach

    # Fallback to original regex approach
    if not fields["document_type"]:
        m = re.search(r"\b(loan agreement|promissory note|service agreement|nda|non-disclosure agreement)\b", t, re.I)
        if m:
            fields["document_type"] = m.group(1).title()
        else:
            # Heuristic: first heading
            top = (t.splitlines() or [""])[0][:80]
            if len(top.split()) <= 6:
                fields["document_type"] = top.title()

    # Key terms - only extract if not blank template placeholders
    def grab(label):
        pat = rf"{label}\s*[:\-]?\s*(.+?)\n"
        m = re.search(pat, t, re.I)
        if m:
            val = _clean(m.group(1))
            # Skip template placeholders
            if val and not re.match(r'^_+$', val) and len(val) > 3:
                return val
        return None

    fields["start_date_first_payment"] = grab("Start Date of the First Payment")
    fields["end_date_last_payment"] = grab("End Date of the Last Payment")
    fields["loan_amount"] = grab("Loan Amount")
    fields["interest_rate"] = grab("Interest Rate")
    fields["late_fee"] = grab("Late Fee")
    fields["payment_method"] = grab("Payment Method")

    return fields

class AnalyzeTextIn(BaseModel):
    text: str

@app.post("/analyze-text")
async def analyze_text(payload: AnalyzeTextIn):
    text = (payload.text or "").strip()
    if not text:
        return {"summary": "", "structured": {}}
    structured = extract_contract_fields(text)
    # Prefer Gemini to format a crisp summary
    summary = summarize_with_gemini(
        (
            "Please generate a concise legal-style summary for the following document.\n" \
            "Focus on:\n" \
            "- Document type and purpose\n" \
            "- Parties involved (if identifiable)\n" \
            "- Key terms that are filled in (ignore blank template fields like '____')\n" \
            "- Important dates if present\n" \
            "- Overall legal significance\n\n" \
            f"Document text: {text[:8000]}"
        )
    )
    if not summary:
        # Fallback to template
        parts = [
            f"Document: {structured.get('document_type') or 'Unspecified'}.",
        ]
        if structured.get('borrower') or structured.get('lender'):
            parts.append(
                "Parties: "
                + ", ".join(
                    [
                        f"Borrower: {structured.get('borrower')}" if structured.get('borrower') else None,
                        f"Lender: {structured.get('lender')}" if structured.get('lender') else None,
                    ]
                    if any([structured.get('borrower'), structured.get('lender')]) else []
                )
            )
        schedule = []
        if structured.get('start_date_first_payment'):
            schedule.append(f"First Payment Start: {structured['start_date_first_payment']}")
        if structured.get('end_date_last_payment'):
            schedule.append(f"Last Payment End: {structured['end_date_last_payment']}")
        if schedule:
            parts.append("Schedule: " + ", ".join(schedule))
        if structured.get('loan_amount'):
            parts.append(f"Amount: {structured['loan_amount']}")
        if structured.get('interest_rate'):
            parts.append(f"Interest: {structured['interest_rate']}")
        summary = " ".join(parts)
    return {"summary": summary, "structured": structured}

# -----------------------------
# PDF Processing
# -----------------------------

# Check for OCR dependencies
try:
    import pytesseract
    from PIL import Image
    HAS_OCR_DEPS = True
except ImportError:
    HAS_OCR_DEPS = False

# Check for PyMuPDF
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

# Check for Tesseract
try:
    import pytesseract
    pytesseract.get_tesseract_version()
    HAS_TESSERACT = True
except Exception:
    HAS_TESSERACT = False

def extract_pdf_text_pypdf(data: bytes) -> tuple[str, int, int]:
    """Extract text from PDF using pypdf (fallback when PyMuPDF not available)"""
    try:
        from pypdf import PdfReader
        import io
        
        pdf_file = io.BytesIO(data)
        reader = PdfReader(pdf_file)
        
        texts = []
        pages_count = len(reader.pages)
        
        for page in reader.pages:
            text = page.extract_text()
            texts.append(text)
        
        text = "\n".join(texts).strip()
        return text, pages_count, 0  # pypdf doesn't easily count images
    except Exception as e:
        raise ImportError(f"pypdf extraction failed: {e}")

def extract_pdf_text_native(data: bytes) -> tuple[str, int, int]:
    """Extract text from PDF using PyMuPDF (native text extraction)"""
    if not HAS_PYMUPDF:
        raise ImportError("PyMuPDF not available")

    doc = fitz.open(stream=data, filetype="pdf")
    texts = []
    total_images = 0
    pages_count = len(doc)

    for page_num in range(pages_count):
        page = doc.load_page(page_num)
        text = page.get_text()
        texts.append(text)

        # Count images on this page
        image_list = page.get_images()
        total_images += len(image_list)

    text = "\n".join(texts).strip()
    doc.close()
    return text, pages_count, total_images

def ocr_pdf_to_text(data: bytes, lang: str = "eng") -> tuple[str, int, int]:
    """Extract text from PDF using OCR (when native text extraction fails)"""
    if not HAS_OCR_DEPS:
        raise ImportError("OCR dependencies not available (install pillow+pytesseract)")

    import fitz
    from PIL import Image
    import io

    doc = fitz.open(stream=data, filetype="pdf")
    texts = []
    total_images = 0
    pages_count = len(doc)

    for page_num in range(pages_count):
        page = doc.load_page(page_num)

        # Convert page to image
        pix = page.get_pixmap()
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))

        # Extract text using OCR
        text = pytesseract.image_to_string(img, lang=lang)
        texts.append(text)
        total_images += 1

    text = "\n".join(texts).strip()
    doc.close()
    return text, pages_count, total_images

def summarize_with_gemini(text: str, api_key: Optional[str] = None) -> Optional[str]:
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        prompt = (
            "You are a legal document summarization assistant. Create a clear, well-organized summary using ONLY bullet points.\n\n"
            "FORMATTING REQUIREMENTS:\n"
            "- Use bullet points (•) for ALL content\n"
            "- Start with document type and date\n"
            "- Organize into clear sections with bold headings\n"
            "- Keep each bullet point concise (1-2 lines max)\n"
            "- Use sub-bullets for details when needed\n"
            "- DO NOT use paragraphs or long text blocks\n"
            "- DO NOT wrap output in code fences or backticks\n\n"
            "CONTENT TO INCLUDE:\n"
            "• Document Type & Date\n"
            "• Parties Involved (names, roles, addresses)\n"
            "• Key Terms & Conditions\n"
            "• Financial/Property Details\n"
            "• Important Dates & Deadlines\n"
            "• Legal References (statutes, sections)\n"
            "• Outcome/Significance\n\n"
            f"Document text:\n{text[:120000]}"
        )
        resp = model.generate_content(prompt)
        return getattr(resp, "text", None) or (resp.candidates[0].content.parts[0].text if resp and resp.candidates else None)
    except Exception:
        return None

@app.post("/analyze-pdf")
async def analyze_pdf(
    file: UploadFile = File(...),
    gemini_api_key: Optional[str] = Form(default=None),
    lang: Optional[str] = Form(default="eng"),
):
    data = await file.read()
    if not data:
        return {"error": "Empty file uploaded", "text": "", "summary": ""}
    
    # Try multiple extraction methods in order of preference
    used_ocr = False
    text, pages, images = "", 0, 0
    extraction_error = None
    
    # Method 1: Try PyMuPDF (most feature-complete)
    if HAS_PYMUPDF:
        try:
            text, pages, images = extract_pdf_text_native(data)
        except Exception as e:
            extraction_error = f"PyMuPDF failed: {e}"
    
    # Method 2: Try pypdf (lightweight fallback)
    if not text or len(text.strip()) < 30:
        try:
            text, pages, images = extract_pdf_text_pypdf(data)
            extraction_error = None  # Success!
        except Exception as e:
            if extraction_error:
                extraction_error += f" | pypdf failed: {e}"
            else:
                extraction_error = f"pypdf failed: {e}"
    
    # Method 3: Try OCR as last resort
    if (not text or len(text.strip()) < 30) and HAS_OCR_DEPS:
        try:
            text, pages, images = ocr_pdf_to_text(data, lang=lang or "eng")
            used_ocr = True
            extraction_error = None  # Success!
        except Exception as e:
            if extraction_error:
                extraction_error += f" | OCR failed: {e}"
            else:
                extraction_error = f"OCR failed: {e}"
    
    # If all methods failed, return error
    if not text or len(text.strip()) < 10:
        return {
            "error": f"PDF text extraction failed: {extraction_error or 'No text found'}",
            "text": text,
            "summary": "",
            "pages": pages,
            "images": images,
            "used_ocr": used_ocr
        }

    summary = summarize_with_gemini(text, api_key=gemini_api_key)
    # Add structured meta line using lightweight extractors (leverages LegalBERT where available)
    try:
        structured = extract_contract_fields(text)
    except Exception:
        structured = {}
    if not summary:
        # Fallback to transformers/extractive
        try:
            smz = get_summarizer()
            if smz is not None:
                out = smz(text, max_length=220, min_length=80, do_sample=False)
                summary = out[0]["summary_text"]
        except Exception:
            summary = None
        if not summary:
            # Extractive backup
            import re
            sents = re.split(r"(?<=[.!?])\s+", text)
            sents = [s for s in sents if len(s.split()) > 4]
            best = sorted(sents, key=lambda s: min(len(s), 200), reverse=True)[: 8]
            summary = " ".join(best)

    # Append a compact meta footer to help the UI render quick facts
    if structured:
        meta_parts = []
        for k in [
            "document_type",
            "start_date_first_payment",
            "end_date_last_payment",
            "loan_amount",
            "interest_rate",
            "late_fee",
            "payment_method",
        ]:
            v = structured.get(k)
            if v:
                meta_parts.append(f"{k}: {v}")
        if meta_parts:
            summary = (summary or "").rstrip() + "\n\n— " + " | ".join(meta_parts)

    # Enhanced response with Legal BERT analysis if available
    response_data = {
        "text": text, 
        "summary": summary, 
        "pages": pages, 
        "images": images, 
        "used_ocr": used_ocr,
        "structured_data": structured
    }
    
    # Add Legal BERT analysis if available
    if HAS_LEGAL_BERT and len(text.strip()) > 50:
        try:
            # Analyze with Legal BERT (use first 3000 chars to avoid token limits)
            bert_analysis = legal_bert_analyzer.analyze_document(text[:3000])
            response_data["legal_bert_analysis"] = {
                "document_type": bert_analysis.get('document_type', {}),
                "key_entities": bert_analysis.get('key_entities', {}),
                "legal_terms": bert_analysis.get('legal_terms', []),
                "risk_factors": bert_analysis.get('risk_factors', []),
                "confidence": bert_analysis.get('confidence', 'unknown')
            }
        except Exception as e:
            print(f"⚠️ Legal BERT analysis on PDF failed: {e}")
            response_data["legal_bert_analysis"] = {"error": str(e)}
    
    return response_data


@app.post("/chatbot/ask")
async def chatbot_ask(payload: dict, x_openai_api_key: Optional[str] = Header(default=None)):
    """Answer a user question using the context of the current summary and/or source text.

    Always returns an answer: uses GPT when available, falls back to a local extractive method otherwise.
    """
    import re as _re
    question = (payload.get("question") or "").strip()
    summary = (payload.get("summary") or "").strip()
    source_text = (payload.get("source_text") or "").strip()
    if not question:
        return {"error": "question is required"}

    # Try GPT via ChatbotService if available
    svc = CHATBOT
    if svc is None:
        try:
            from chatbot_service import ChatbotService  # lazy import
            svc = ChatbotService()
        except Exception:
            svc = None
    if svc is not None:
        try:
            ans = svc.answer(question, summary=summary, source_text=source_text, api_key_override=x_openai_api_key)
            if ans:
                return {"answer": ans}
        except Exception:
            pass

    # Local extractive fallback (no external APIs)
    context = ((summary or "") + "\n\n" + (source_text or "")).strip()
    if not context:
        return {"answer": "No context available. Generate a summary or paste text first."}
    sents = _re.split(r"(?<=[.!?])\s+", context)
    sents = [s.strip() for s in sents if s and len(s.split()) > 3]
    q_terms = [w.lower() for w in _re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", question) if len(w) > 2] or [w.lower() for w in _re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", question)]
    def _score(sent: str) -> float:
        ls = sent.lower()
        return sum(2.0 if t in ls else 0.0 for t in q_terms) + min(len(sent)/200.0, 1.0)
    top = " ".join(sorted(sents, key=_score, reverse=True)[:2]).strip()
    return {"answer": top or "I couldn't find that in the current summary."}

@app.post("/legal-bert-analyze")
async def legal_bert_analyze(payload: AnalyzeTextIn):
    """
    Analyze text using Legal BERT for enhanced legal document understanding
    """
    text = (payload.text or "").strip()
    if not text:
        return {"error": "No text provided"}
    
    if not HAS_LEGAL_BERT:
        return {
            "error": "Legal BERT not available",
            "fallback_analysis": extract_contract_fields(text)
        }
    
    try:
        # Use our working Legal BERT implementation
        analysis = legal_bert_analyzer.analyze_document(text)
        
        # Add comparison capabilities if needed
        comparison_info = {
            "model_loaded": legal_bert_analyzer.is_loaded,
            "model_name": legal_bert_analyzer.model_name,
            "confidence": analysis.get('confidence', 'unknown')
        }
        
        return {
            "success": True,
            "analysis": analysis,
            "model_info": comparison_info
        }
        
    except Exception as e:
        return {
            "error": f"Legal BERT analysis failed: {str(e)}",
            "fallback_analysis": extract_contract_fields(text)
        }

@app.post("/compare-documents")
async def compare_documents(payload: dict):
    """
    Compare two legal documents using Legal BERT embeddings
    """
    text1 = payload.get("text1", "").strip()
    text2 = payload.get("text2", "").strip()
    
    if not text1 or not text2:
        return {"error": "Both text1 and text2 are required"}
    
    if not HAS_LEGAL_BERT:
        return {"error": "Legal BERT not available for document comparison"}
    
    try:
        comparison = legal_bert_analyzer.compare_documents(text1, text2)
        return {
            "success": True,
            "comparison": comparison
        }
    except Exception as e:
        return {
            "error": f"Document comparison failed: {str(e)}"
        }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
