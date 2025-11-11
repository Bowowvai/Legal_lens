from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import Optional, List, Tuple
import math


def _split_sentences(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    # naive sentence splitter
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


@dataclass
class ChatbotConfig:
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: str = os.getenv("OPENAI_MODEL", "gpt-5-chat")


class ChatbotService:
    def __init__(self, config: Optional[ChatbotConfig] = None):
        self.config = config or ChatbotConfig(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            model=os.getenv("OPENAI_MODEL", "gpt-5-chat"),
        )
        self._client = None
        self._available = False
        self._embedder = None
        self._embed_available = False
        try:
            from openai import OpenAI  # type: ignore
            # Create client if env key exists; otherwise remain unavailable
            if self.config.api_key:
                if self.config.base_url:
                    self._client = OpenAI(api_key=self.config.api_key, base_url=self.config.base_url)
                else:
                    self._client = OpenAI(api_key=self.config.api_key)
                self._available = True
        except Exception:
            # openai package not installed or init failed
            self._available = False

        # Try to set up a local embedder for retrieval (optional but improves accuracy)
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            # Small, fast model; good balance for Q&A retrieval
            self._embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            self._embed_available = True
        except Exception:
            self._embed_available = False

    @property
    def available(self) -> bool:
        return bool(self._available)

    def answer(self, question: str, *, summary: str = "", source_text: str = "", api_key_override: Optional[str] = None) -> str:
        """Answer a question using the provided context.

        Summary-first retrieval: we first analyze the SUMMARY to extract facts/sections, then use these
        blocks (and the raw summary text) as the main knowledge base. Source text is optional and used
        only as a tiebreaker. Then:
        - If GPT client is available, ask it to answer with citations [C#].
        - Otherwise, return a concise extractive answer from the top blocks.
        """
        question = (question or "").strip()
        summary = (summary or "").strip()
        context = summary if summary else ((source_text or "").strip())
        if not question:
            return "Please provide a question."
        if not context:
            return "No context available. Generate a summary or paste text first."

        # Intent shortcuts (e.g., just names, dates, amounts)
        # Amounts intent must return only the monetary value with minimal context
        if self._is_amounts_intent(question):
            amount_answer = self._extract_best_amount_answer(question, (summary + "\n" + source_text).strip())
            if amount_answer:
                return amount_answer
        # Dates intent must be answered with a single, professional line.
        if self._is_dates_intent(question):
            date_answer = self._extract_best_date_answer(question, (summary + "\n" + source_text).strip())
            if date_answer:
                return date_answer
        if self._is_names_intent(question):
            names = self._extract_people((summary + "\n" + source_text).strip())
            if names:
                return ", ".join(names)

        # Build retrieval context (summary-first)
        fact_blocks = self._summary_fact_blocks(summary) if summary else []
        sum_chunks = self._chunk_context(summary) if summary else []
        # optional: a few coarse chunks from source text to assist
        src_chunks = self._chunk_context(source_text, target=900)[:5] if source_text else []
        all_blocks = fact_blocks + sum_chunks + src_chunks
        selected = self._select_chunks(question, all_blocks, k=6)
        rag_context = self._format_citation_blocks(selected)

        if self._available:
            try:
                # optional per-request key (header value forwarded by API layer)
                client = self._client
                if api_key_override:
                    from openai import OpenAI  # type: ignore
                    if self.config.base_url:
                        client = OpenAI(api_key=api_key_override, base_url=self.config.base_url)
                    else:
                        client = OpenAI(api_key=api_key_override)

                system = (
                    "You are a precise legal assistant. Your answers must be complete, accurate, and never truncated. "
                    "Format rules:"
                    "1. For amounts: 'Spousal support amount: $50,000' or 'Settlement amount: $75,000'"
                    "2. For dates: 'Agreement date: 29 August 2025' or 'Effective date: 1 September 2025'"
                    "3. For names: 'Parties: John Smith, Jane Smith' or 'Petitioner: John Smith'"
                    "4. For other questions: Give a single complete sentence with all relevant details."
                    "\nNever truncate numbers, names, or sentences. Use the exact values from the context."
                    "\nUse ONLY information from the provided context blocks."
                    "\nIf information is not found, respond exactly: Not found in the provided context."
                )
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"Context blocks (facts first, then summary excerpts):\n{rag_context}\n\nQuestion: {question}\n\nOutput rules: Direct answer only; if multiple items, use a comma-separated list; no citations in the final text."},
                ]
                # Prefer chat.completions for broad compatibility
                resp = client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=600,
                )
                txt = (resp.choices[0].message.content or "").strip()
                if txt:
                    return txt
            except Exception:
                pass  # fall through to heuristic fallback

        # Heuristic extractive fallback over top chunks
        joined = " ".join(t for _, t in selected) or context
        ctx_sentences = _split_sentences(joined)
        if not ctx_sentences:
            return "Not found in the provided context."
        q_terms = [w.lower() for w in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", question) if len(w) > 2] or [w.lower() for w in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", question)]
        def score(sent: str) -> float:
            s = sent.lower()
            return sum(2.0 if t in s else 0.0 for t in q_terms) + min(len(sent)/200.0, 1.0)
        ranked = sorted(ctx_sentences, key=score, reverse=True)
        top = " ".join(ranked[:3]).strip()
        if not top:
            return "Not found in the provided context."
        # Post-process for minimality
        if self._is_amounts_intent(question):
            # Try to extract amount from the most relevant text
            amount_answer = self._extract_best_amount_answer(question, top)
            if amount_answer:
                return amount_answer
        if self._is_dates_intent(question):
            # Try to extract a date from the most relevant text
            date_answer = self._extract_best_date_answer(question, top)
            if date_answer:
                return date_answer
        if self._is_names_intent(question):
            people = self._extract_people(top)
            if people:
                return ", ".join(people)
        return self._trim_minimal(top, max_words=25)

    # ---- Retrieval helpers ----
    def _chunk_context(self, context: str, *, target: int = 800, overlap: int = 120) -> List[Tuple[str, str]]:
        # Split by paragraphs, then window into target sized pieces with overlap
        paras = [p.strip() for p in re.split(r"\n\s*\n+", context) if p.strip()]
        chunks: List[Tuple[str, str]] = []
        idx = 1
        for p in paras:
            if len(p) <= target:
                chunks.append((f"C{idx}", p))
                idx += 1
            else:
                # windowed slicing by sentences to avoid cutting mid-sentence
                sentences = _split_sentences(p)
                current_chunk = ""
                for sent in sentences:
                    if len(current_chunk + " " + sent) <= target:
                        current_chunk = (current_chunk + " " + sent).strip()
                    else:
                        if current_chunk:
                            chunks.append((f"C{idx}", current_chunk))
                            idx += 1
                        current_chunk = sent
                if current_chunk:
                    chunks.append((f"C{idx}", current_chunk))
                    idx += 1
        # cap number of chunks to avoid huge prompts
        return chunks[:40]

    def _summary_fact_blocks(self, summary: str) -> List[Tuple[str, str]]:
        """Extract structured fact blocks from a Markdown-like summary.

        We parse headings, bullet lists, and key:value lines to create compact blocks that are easier to retrieve.
        Returns a list of (cid, text) small blocks, with stable ordering.
        """
        if not summary:
            return []
        lines = [l.rstrip() for l in summary.splitlines()]
        blocks: List[str] = []
        cur: List[str] = []
        def flush():
            nonlocal cur
            t = " ".join(s.strip() for s in cur if s.strip()).strip()
            if t:
                blocks.append(t)
            cur = []
        def is_heading(s: str) -> bool:
            s2 = s.strip()
            return s2.startswith('#') or (s2.endswith(':') and 3 <= len(s2) <= 120) or bool(re.match(r"^[A-Z][A-Za-z0-9 \-]{2,80}$", s2))
        for ln in lines:
            if not ln.strip():
                flush(); continue
            if is_heading(ln):
                flush(); cur.append(ln.strip(': ').strip()); flush(); continue
            # key:value
            m = re.match(r"^\s*([A-Za-z][A-Za-z0-9 \-/]{2,60})\s*[:\-]\s*(.+)$", ln)
            if m:
                flush(); blocks.append(f"{m.group(1).strip()}: {m.group(2).strip()}"); continue
            # bullets or numbered lists
            if re.match(r"^\s*[-*]\s+.+$", ln) or re.match(r"^\s*\d+\.\s+.+$", ln):
                cur.append(ln.strip()); continue
            cur.append(ln)
        flush()
        # normalize and limit
        cleaned = []
        for b in blocks:
            t = re.sub(r"\s+", " ", b).strip()
            if t and len(t) >= 6:
                cleaned.append(t)
        cleaned = cleaned[:50]
        # label as C1, C2, ...
        return [(f"C{i+1}", t) for i, t in enumerate(cleaned)]

    def _select_chunks(self, question: str, chunks: List[Tuple[str, str]], k: int = 5) -> List[Tuple[str, str]]:
        if not chunks:
            return []

        # First detect intent to optimize retrieval
        ql = question.lower()
        amount_intent = any(w in ql for w in ["amount", "money", "pay", "payment", "support", "alimony"])
        date_intent = any(w in ql for w in ["date", "when", "signed", "effective"])
        name_intent = any(w in ql for w in ["who", "name", "party", "parties", "petitioner", "respondent"])

        # Helper to check if chunk contains complete info
        def has_complete_info(text: str) -> bool:
            if amount_intent:
                return bool(re.search(r'\$\d[\d,]*', text))
            if date_intent:
                return bool(re.search(r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:day\s+of\s+)?(?:January|February|March|April|May|June|July|August|September|October|November|December)\s*,?\s*\d{4}', text, re.I))
            if name_intent:
                return bool(re.search(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}', text))
            return True

        if self._embed_available and self._embedder is not None:
            try:
                texts = [c[1] for c in chunks]
                qv = self._embedder.encode([question], normalize_embeddings=True)[0]
                cvs = self._embedder.encode(texts, normalize_embeddings=True)
                def cos(a, b):
                    return float((a @ b).item()) if hasattr(a, "__matmul__") else sum(x*y for x,y in zip(a,b))
                # Score by embedding similarity but boost chunks with complete information
                scored = [(cos(qv, v) * (2.0 if has_complete_info(chunks[i][1]) else 1.0), chunks[i]) 
                         for i, v in enumerate(cvs)]
                scored.sort(key=lambda x: x[0], reverse=True)
                selected = [c for _, c in scored[:k]]
                # If we don't have complete info in top chunks, look further
                if not any(has_complete_info(c[1]) for c in selected):
                    for _, c in scored[k:]:
                        if has_complete_info(c[1]):
                            selected.append(c)
                            break
                return selected
            except Exception:
                pass

        # Keyword fallback scoring with intent awareness
        terms = [w.lower() for w in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", question) if len(w) > 2]
        if not terms:
            terms = [w.lower() for w in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", question)]
        
        def kw_score(chunk: Tuple[str, str]) -> float:
            text = chunk[1].lower()
            # Base score from keyword matches
            score = sum(2.0 if t in text else 0.0 for t in terms)
            # Boost for having complete information
            if has_complete_info(chunk[1]):
                score *= 2.0
            # Small boost for longer context
            score += min(len(text)/1000.0, 0.5)
            return score

        ranked = sorted(chunks, key=kw_score, reverse=True)
        selected = ranked[:k]
        # Again, ensure we have complete info
        if not any(has_complete_info(c[1]) for c in selected):
            for c in ranked[k:]:
                if has_complete_info(c[1]):
                    selected.append(c)
                    break
        return selected

    def _format_citation_blocks(self, selected: List[Tuple[str, str]]) -> str:
        if not selected:
            return ""
        return "\n\n".join([f"[{cid}]:\n{txt}" for cid, txt in selected])

    # ---- Intent and light NLP helpers ----
    def _is_amounts_intent(self, q: str) -> bool:
        ql = (q or "").lower()
        return any(w in ql for w in [
            "amount", "cost", "fee", "price", "money", "payment", "sum", "dollar", "support", "alimony", "settlement", "loan", "debt"
        ]) and not any(w in ql for w in ["name", "date", "address", "clause", "section", "why", "how", "explain"])

    def _is_dates_intent(self, q: str) -> bool:
        ql = (q or "").lower()
        return any(w in ql for w in [
            "date", "when", "effective", "signed", "signing", "executed", "execution", "commencement", "start", "ending", "termination", "expire"
        ]) and not any(w in ql for w in ["name", "amount", "address", "clause", "section", "why", "how", "explain"]) 

    def _is_names_intent(self, q: str) -> bool:
        ql = (q or "").lower()
        return any(w in ql for w in ["name", "names", "people", "parties", "who", "signatories", "participants", "person"]) and not any(
            w in ql for w in ["date", "amount", "address", "clause", "section", "summary", "explain", "why"]
        )

    def _extract_people(self, text: str) -> List[str]:
        t = (text or "")
        if not t:
            return []
        names: List[str] = []
        # Common legal role patterns
        role_pat = re.findall(r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})\s*\((?:Petitioner|Respondent|Plaintiff|Defendant|Party)\)", t)
        for n in role_pat:
            names.append(n.strip())
        # 'between X ... and Y ...' pattern
        m = re.search(r"between\s+([^\n,()]+?)\s*\(.*?\)\s*(?:,?\s*and\s+([^\n,()]+?)\s*\(.*?\))?", t, flags=re.I)
        if m:
            for g in [m.group(1), m.group(2) if m.lastindex and m.lastindex >= 2 else None]:
                if g:
                    names.append(g.strip())
        # Capitalized name heuristic (2–3 words)
        cap_names = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b", t)
        stop = {"Marital", "Settlement", "Agreement", "Petitioner", "Respondent", "Plaintiff", "Defendant", "Nagar", "Bengaluru", "Karnataka", "HSR", "Layout", "Block"}
        for n in cap_names:
            w = n.strip()
            if any(ch.isdigit() for ch in w):
                continue
            if w.split()[0] in stop or w.split()[-1] in stop:
                continue
            names.append(w)
        # Deduplicate preserve order
        seen = set()
        out: List[str] = []
        for n in names:
            key = n.lower()
            if key not in seen and 2 <= len(n.split()) <= 4:
                seen.add(key); out.append(n)
        return out[:10]

    def _trim_minimal(self, text: str, *, max_words: int = 25) -> str:
        t = (text or "").strip()
        if not t:
            return t

        # Split into sentences
        sents = _split_sentences(t)
        if not sents:
            return t

        # Helper to check if sentence has complete information
        def has_important_info(s: str) -> bool:
            return bool(
                # Has monetary amount
                re.search(r'\$\d[\d,]*', s) or
                # Has date
                re.search(r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:day\s+of\s+)?(?:January|February|March|April|May|June|July|August|September|October|November|December)\s*,?\s*\d{4}', s, re.I) or
                # Has proper name
                re.search(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}', s) or
                # Has important legal terms
                any(term in s.lower() for term in ["agree", "settle", "payment", "support", "alimony", "sign"])
            )

        # Try to find a single complete sentence with important info
        for sent in sents:
            if has_important_info(sent):
                return sent

        # If no single sentence has what we need, take the first sentence
        # that mentions anything relevant
        t = sents[0]
        words = t.split()
        if len(words) <= max_words:
            return t

        # If we must trim, try to keep complete phrases
        phrases = re.split(r'[,;]\s+', t)
        result = []
        word_count = 0
        for phrase in phrases:
            phrase_words = phrase.split()
            if word_count + len(phrase_words) <= max_words:
                result.append(phrase)
                word_count += len(phrase_words)
            else:
                break
        
        if result:
            return ", ".join(result)
        
        # Last resort: return first max_words if we couldn't split nicely
        return " ".join(words[:max_words])

    # ---- Date extraction helpers ----
    def _extract_best_amount_answer(self, question: str, text: str) -> Optional[str]:
        amounts = self._extract_amounts_with_labels(text)
        if not amounts:
            return None

        ql = (question or "").lower()
        # Preference by question intent
        def intent_label():
            if any(k in ql for k in ["spousal", "alimony", "support"]):
                return "Spousal support amount"
            if any(k in ql for k in ["loan", "principal", "borrowed"]):
                return "Loan amount"
            if any(k in ql for k in ["settlement", "lump sum"]):
                return "Settlement amount"
            if any(k in ql for k in ["fee", "cost", "charge"]):
                return "Fee amount"
            if any(k in ql for k in ["total", "full", "all"]):
                return "Total amount"
            return "Amount"

        prefer = intent_label()
        # Score amounts by relevance to question
        def score_amount(item: Tuple[Optional[str], str, Optional[str]]) -> float:
            label, amount, context = item
            score = 0.0
            
            # Exact label match gets highest priority
            if label and prefer.lower() in label.lower():
                score += 10.0
            
            # Context relevance
            if context:
                ctx_lower = context.lower()
                # Question term matches
                score += sum(2.0 for term in ql.split() if term in ctx_lower)
                # Intent term matches
                if any(k in ctx_lower for k in ["spousal", "alimony", "support"]) and "support" in ql:
                    score += 5.0
                if any(k in ctx_lower for k in ["settle", "settlement"]) and "settlement" in ql:
                    score += 5.0
                if any(k in ctx_lower for k in ["loan", "borrow"]) and "loan" in ql:
                    score += 5.0

            # Slight preference for larger amounts as they're often more significant
            try:
                value = float(amount.replace('$', '').replace(',', ''))
                score += min(value / 100000.0, 1.0)  # Small boost for larger amounts
            except:
                pass
            
            return score

        # Get full context and sort by relevance
        amounts_with_context = [(label, amount, self._get_amount_context(text, amount)) 
                              for label, amount in amounts]
        ranked = sorted(amounts_with_context, key=score_amount, reverse=True)
        
        if not ranked:
            return None

        label, amount, context = ranked[0]
        final_label = prefer if prefer else (label or "Amount")

        # If we have relevant context, include it
        if context and len(context) < 100:  # Keep it concise
            return f"{final_label}: {amount} ({context.strip()})"
        return f"{final_label}: {amount}"

    def _get_amount_context(self, text: str, amount: str) -> Optional[str]:
        """Extract a brief context snippet around an amount."""
        if not text or not amount:
            return None
        
        # Find the amount in text
        idx = text.find(amount)
        if idx == -1:
            return None
        
        # Get surrounding sentence
        start = max(0, idx - 100)
        end = min(len(text), idx + 100)
        context = text[start:end]
        
        # Try to trim to sentence boundaries
        sentences = _split_sentences(context)
        if sentences:
            for sent in sentences:
                if amount in sent:
                    return sent
        
        return context

    def _extract_amounts_with_labels(self, text: str) -> List[Tuple[Optional[str], str]]:
        t = (text or "").strip()
        if not t:
            return []
        results: List[Tuple[Optional[str], str]] = []
        # Build a lowercase mirror for context matching
        tl = t.lower()
        
        # Patterns for monetary amounts
        # 1) $50,000 or $50000 or $50.000
        pat1 = re.compile(r"\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)")
        # 2) 50,000 dollars or 50000 dollars  
        pat2 = re.compile(r"(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*dollars?", re.I)
        # 3) USD 50,000 or USD50,000
        pat3 = re.compile(r"USD\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)", re.I)

        def add_with_label(start_idx: int, end_idx: int, amount_str: str):
            near = tl[max(0, start_idx-80):min(len(tl), end_idx+80)]
            label: Optional[str] = None
            if any(k in near for k in ["spousal", "alimony", "support"]):
                label = "Spousal support amount"
            elif any(k in near for k in ["loan", "principal", "borrowed"]):
                label = "Loan amount"
            elif any(k in near for k in ["settlement", "lump sum"]):
                label = "Settlement amount" 
            elif any(k in near for k in ["fee", "cost", "charge"]):
                label = "Fee amount"
            results.append((label, f"${amount_str}"))

        for m in pat1.finditer(t):
            amount = m.group(1)
            add_with_label(m.start(), m.end(), amount)
        for m in pat2.finditer(t):
            amount = m.group(1)
            add_with_label(m.start(), m.end(), amount)
        for m in pat3.finditer(t):
            amount = m.group(1)
            add_with_label(m.start(), m.end(), amount)

        # Deduplicate and sort by amount (largest first)
        seen = set()
        uniq: List[Tuple[Optional[str], str]] = []
        for item in results:
            amount_key = item[1].replace('$', '').replace(',', '')
            if amount_key not in seen:
                seen.add(amount_key)
                uniq.append(item)
        
        # Sort by numeric value (largest first)
        def amount_value(item):
            try:
                return float(item[1].replace('$', '').replace(',', ''))
            except:
                return 0
        uniq.sort(key=amount_value, reverse=True)
        return uniq[:5]

    # ---- Date extraction helpers ----
    def _extract_best_date_answer(self, question: str, text: str) -> Optional[str]:
        dates = self._extract_dates_with_labels(text)
        if not dates:
            return None
        ql = (question or "").lower()
        # Preference by question intent
        def intent_label():
            if any(k in ql for k in ["effective", "commencement", "start"]):
                return "Effective date"
            if any(k in ql for k in ["sign", "execut", "made on", "execution"]):
                return "Agreement date"
            if any(k in ql for k in ["termination", "expire", "end", "ending"]):
                return "Termination date"
            return "Agreement date"
        prefer = intent_label()
        # Rank: exact label match first, then any, then earliest in text
        ranked = sorted(dates, key=lambda d: (0 if (d[0] and prefer.split()[0].lower() in d[0].lower()) else 1))
        label, norm = ranked[0]
        final_label = prefer if prefer else (label or "Agreement date")
        return f"{final_label}: {norm}"

    def _extract_dates_with_labels(self, text: str) -> List[Tuple[Optional[str], str]]:
        t = (text or "").strip()
        if not t:
            return []
        results: List[Tuple[Optional[str], str]] = []
        # Build a lowercase mirror for context matching
        tl = t.lower()
        # Patterns for dates
        month_names = r"January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
        # 1) "this 29th day of August, 2025"
        pat1 = re.compile(rf"\b(?:this\s+)?(\d{{1,2}})(?:st|nd|rd|th)?\s+day\s+of\s+({month_names})\,?\s+(\d{{4}})\b", re.I)
        # 2) 29th August 2025 or 29 August 2025
        pat2 = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({month_names})\,?\s+(\d{{4}})\b", re.I)
        # 3) August 29, 2025
        pat3 = re.compile(rf"\b({month_names})\s+(\d{{1,2}})(?:st|nd|rd|th)?\,?\s+(\d{{4}})\b", re.I)
        # 4) 2025-08-29
        pat4 = re.compile(r"\b(\d{4})[\/-](\d{1,2})[\/-](\d{1,2})\b")
        # 5) 29/08/2025 or 29-08-2025
        pat5 = re.compile(r"\b(\d{1,2})[\/-](\d{1,2})[\/-](\d{2,4})\b")

        def month_to_int(m: str) -> int:
            m = m.lower()[:3]
            mp = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,"jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12}
            return mp.get(m, 1)

        def norm(d: int, m: int, y: int) -> str:
            # Return like '29 August 2025'
            months = ["January","February","March","April","May","June","July","August","September","October","November","December"]
            mname = months[max(1, min(12, m)) - 1]
            return f"{d} {mname} {y}"

        # Helper to add result with nearby label
        def add_with_label(start_idx: int, end_idx: int, d: int, m: int, y: int):
            near = tl[max(0, start_idx-60):min(len(tl), end_idx+40)]
            label: Optional[str] = None
            if any(k in near for k in ["effective", "commencement", "start"]):
                label = "Effective date"
            elif any(k in near for k in ["sign", "execut", "made on"]):
                label = "Agreement date"
            elif any(k in near for k in ["termination", "expire", "end", "ending"]):
                label = "Termination date"
            results.append((label, norm(d, m, y)))

        for m in pat1.finditer(t):
            d, mon, y = int(m.group(1)), m.group(2), int(m.group(3))
            add_with_label(m.start(), m.end(), d, month_to_int(mon), y)
        for m in pat2.finditer(t):
            d, mon, y = int(m.group(1)), m.group(2), int(m.group(3))
            add_with_label(m.start(), m.end(), d, month_to_int(mon), y)
        for m in pat3.finditer(t):
            mon, d, y = m.group(1), int(m.group(2)), int(m.group(3))
            add_with_label(m.start(), m.end(), d, month_to_int(mon), y)
        for m in pat4.finditer(t):
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            add_with_label(m.start(), m.end(), d, mo, y)
        for m in pat5.finditer(t):
            d, mo, y = int(m.group(1)), int(m.group(2)), m.group(3)
            y = int(y) + 2000 if len(y) == 2 else int(y)
            add_with_label(m.start(), m.end(), d, mo, y)

        # Deduplicate
        seen = set()
        uniq: List[Tuple[Optional[str], str]] = []
        for item in results:
            key = (item[0] or "" , item[1])
            if key not in seen:
                seen.add(key)
                uniq.append(item)
        return uniq[:5]
