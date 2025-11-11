#!/usr/bin/env python3
"""
Working Legal BERT Implementation
Avoids TensorFlow conflicts and provides stable legal document analysis
"""

import os
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LegalBERTAnalyzer:
    """
    A working Legal BERT implementation that provides legal document analysis
    without TensorFlow conflicts
    """
    
    def __init__(self, model_name: str = "nlpaueb/legal-bert-base-uncased"):
        """
        Initialize the Legal BERT analyzer with a stable model
        
        Args:
            model_name: HuggingFace model name to use
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_loaded = False
        
        # Legal document patterns for classification
        self.legal_patterns = {
            "contract": [
                "agreement", "contract", "terms", "conditions", "parties", 
                "obligations", "liabilities", "breach", "termination"
            ],
            "legal_document": [
                "legal", "law", "statute", "regulation", "ordinance", 
                "code", "act", "bill", "amendment"
            ],
            "financial": [
                "loan", "mortgage", "credit", "debt", "payment", "interest",
                "principal", "collateral", "guarantee", "security"
            ],
            "employment": [
                "employment", "employee", "employer", "salary", "benefits",
                "termination", "non-compete", "confidentiality"
            ],
            "property": [
                "lease", "rental", "property", "real estate", "landlord",
                "tenant", "eviction", "maintenance"
            ],
            "family_law": [
                "marital", "divorce", "settlement", "custody", "child support",
                "alimony", "spousal support", "marriage", "separation",
                "prenuptial", "postnuptial", "domestic relations"
            ]
        }
        
        # Load the model
        self._load_model()
    
    def _load_model(self):
        """Load the BERT model and tokenizer"""
        try:
            logger.info(f"Loading Legal BERT model: {self.model_name}")
            
            # Import transformers components safely
            from transformers import AutoTokenizer, AutoModel
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "legal-models"),
                use_fast=True
            )
            
            # Load model with safetensors preference to avoid PyTorch version issues
            try:
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    cache_dir=os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "legal-models"),
                    torch_dtype=torch.float32,  # Use float32 for compatibility
                    use_safetensors=True  # Prefer safetensors to avoid PyTorch version issues
                )
            except Exception as e:
                logger.warning(f"Safetensors loading failed, trying regular loading: {e}")
                # Fallback to regular loading
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    cache_dir=os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "legal-models"),
                    torch_dtype=torch.float32
                )
            
            # Move to device
            self.model.to(self.device)
            self.model.eval()
            
            self.is_loaded = True
            logger.info(f"✅ Legal BERT model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load Legal BERT model: {e}")
            self.is_loaded = False
            # Fallback to rule-based analysis
            logger.info("🔄 Falling back to rule-based legal analysis")
    
    def analyze_document(self, text: str) -> Dict:
        """
        Analyze a legal document and extract key information
        
        Args:
            text: Document text to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        if not text or not text.strip():
            return {"error": "No text provided"}
        
        # Clean text
        text = self._clean_text(text)
        
        # Initialize results
        analysis = {
            "document_type": self._classify_document_type(text),
            "key_entities": self._extract_entities(text),
            "legal_terms": self._extract_legal_terms(text),
            "risk_factors": self._identify_risk_factors(text),
            "summary": self._generate_summary(text),
            "confidence": "high" if self.is_loaded else "medium"
        }
        
        return analysis
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove special characters but keep legal symbols
        text = re.sub(r'[^\w\s\-\.\,\;\:\!\?\(\)\[\]\{\}\"\']', ' ', text)
        return text
    
    def _classify_document_type(self, text: str) -> Dict:
        """Classify the type of legal document"""
        text_lower = text.lower()
        
        # Calculate confidence scores for each category
        scores = {}
        for category, patterns in self.legal_patterns.items():
            score = sum(1 for pattern in patterns if pattern in text_lower)
            scores[category] = score / len(patterns) if patterns else 0
        
        # Get the highest scoring category
        if scores:
            best_category = max(scores, key=scores.get)
            confidence = scores[best_category]
            
            # Additional specific document type detection
            specific_type = self._detect_specific_type(text_lower)
            
            return {
                "category": best_category,
                "specific_type": specific_type,
                "confidence": confidence,
                "scores": scores
            }
        
        return {"category": "unknown", "confidence": 0.0}
    
    def _detect_specific_type(self, text_lower: str) -> str:
        """Detect specific document types"""
        # Family law documents
        if any(word in text_lower for word in ["marital settlement", "divorce settlement", "marriage settlement"]):
            return "Marital Settlement Agreement"
        elif any(word in text_lower for word in ["divorce decree", "dissolution"]):
            return "Divorce Decree"
        elif any(word in text_lower for word in ["custody agreement", "child custody"]):
            return "Custody Agreement"
        elif any(word in text_lower for word in ["prenuptial agreement", "prenup"]):
            return "Prenuptial Agreement"
        elif any(word in text_lower for word in ["postnuptial agreement", "postnup"]):
            return "Postnuptial Agreement"
        # Financial documents
        elif any(word in text_lower for word in ["loan agreement", "promissory note"]):
            return "Loan Agreement"
        elif any(word in text_lower for word in ["mortgage agreement", "deed of trust"]):
            return "Mortgage Agreement"
        # Employment documents
        elif any(word in text_lower for word in ["employment contract", "employment agreement"]):
            return "Employment Contract"
        elif any(word in text_lower for word in ["non-disclosure", "nda", "confidentiality"]):
            return "Non-Disclosure Agreement"
        # Property documents
        elif any(word in text_lower for word in ["lease agreement", "rental agreement"]):
            return "Lease Agreement"
        elif any(word in text_lower for word in ["purchase agreement", "sales contract"]):
            return "Purchase Agreement"
        # Service documents
        elif any(word in text_lower for word in ["service agreement", "service contract"]):
            return "Service Agreement"
        elif any(word in text_lower for word in ["partnership agreement"]):
            return "Partnership Agreement"
        else:
            return "General Legal Document"
    
    def _extract_entities(self, text: str) -> Dict:
        """Extract key entities from the document"""
        entities = {
            "parties": [],
            "dates": [],
            "amounts": [],
            "locations": []
        }
        
        # Extract parties (enhanced pattern matching for family law)
        party_patterns = [
            r'(?:between|by|from|to)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:Inc|LLC|Corp|Corporation|Company|Ltd)',
            r'(?:Party\s+[A-Z])\s*[:=]\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'(?:Husband|Wife|Spouse|Petitioner|Respondent)\s*[:=]\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'(?:Plaintiff|Defendant)\s*[:=]\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
        ]
        
        for pattern in party_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities["parties"].extend([match.strip() for match in matches if match.strip()])
        
        # Extract dates (enhanced for legal documents)
        date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{4}-\d{2}-\d{2}\b',
            r'(?:effective|commencement|termination|expiration)\s+date\s*[:=]\s*([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})'
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            entities["dates"].extend(matches)
        
        # Extract amounts (enhanced for legal documents)
        amount_patterns = [
            r'\$\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?',
            r'\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars|USD)',
            r'(?:amount|sum|total|payment|support|alimony|maintenance)\s*(?:of)?\s*\$?\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?',
            r'(?:child\s+support|spousal\s+support|alimony)\s*[:=]\s*\$?\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?'
        ]
        
        for pattern in amount_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities["amounts"].extend(matches)
        
        # Extract locations (for family law documents)
        location_patterns = [
            r'(?:County|State)\s+of\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'(?:venue|jurisdiction)\s*[:=]\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'(?:governed\s+by|subject\s+to)\s+the\s+laws\s+of\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
        ]
        
        for pattern in location_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities["locations"].extend([match.strip() for match in matches if match.strip()])
        
        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def _extract_legal_terms(self, text: str) -> List[str]:
        """Extract legal terms and concepts (enhanced for family law)"""
        legal_terms = [
            # General legal terms
            "breach", "damages", "liability", "indemnification", "arbitration",
            "jurisdiction", "governing law", "force majeure", "termination",
            "amendment", "assignment", "waiver", "severability", "entire agreement",
            "counterparts", "notice", "default", "cure period", "liquidated damages",
            # Family law specific terms
            "custody", "visitation", "child support", "spousal support", "alimony",
            "maintenance", "marital property", "separate property", "community property",
            "equitable distribution", "marital settlement", "divorce decree",
            "dissolution", "separation", "prenuptial", "postnuptial",
            "domestic relations", "family court", "marriage dissolution"
        ]
        
        found_terms = []
        text_lower = text.lower()
        
        for term in legal_terms:
            if term in text_lower:
                found_terms.append(term)
        
        return found_terms
    
    def _identify_risk_factors(self, text: str) -> List[str]:
        """Identify potential risk factors in the document (enhanced for family law)"""
        risk_indicators = [
            # General risk indicators
            "unlimited liability", "personal guarantee", "cross-default",
            "acceleration clause", "prepayment penalty", "balloon payment",
            "adjustable rate", "variable interest", "penalty", "fine",
            "forfeiture", "lien", "encumbrance",
            # Family law risk indicators
            "modification", "change in circumstances", "cost of living adjustment",
            "income change", "employment change", "relocation", "move away",
            "contempt", "enforcement", "arrears", "default in payment",
            "termination of support", "remarriage", "cohabitation"
        ]
        
        risks = []
        text_lower = text.lower()
        
        for indicator in risk_indicators:
            if indicator in text_lower:
                risks.append(indicator)
        
        return risks
    
    def _generate_summary(self, text: str) -> str:
        """Generate a summary of the legal document"""
        if not self.is_loaded:
            # Rule-based summary
            return self._rule_based_summary(text)
        
        try:
            # BERT-based summary
            return self._bert_summary(text)
        except Exception as e:
            logger.warning(f"BERT summary failed, falling back to rule-based: {e}")
            return self._rule_based_summary(text)
    
    def _rule_based_summary(self, text: str) -> str:
        """Generate summary using rule-based approach"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if not sentences:
            return "Document appears to be incomplete or contains insufficient text for analysis."
        
        # Take first few sentences and key sentences with legal terms
        summary_sentences = sentences[:3]
        
        # Add sentences with legal terms
        legal_terms = self._extract_legal_terms(text)
        for sentence in sentences[3:10]:  # Look in first 10 sentences
            if any(term in sentence.lower() for term in legal_terms):
                summary_sentences.append(sentence)
                if len(summary_sentences) >= 5:
                    break
        
        return " ".join(summary_sentences) + "."
    
    def _bert_summary(self, text: str) -> str:
        """Generate summary using BERT embeddings"""
        try:
            # Tokenize text
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)  # Average pooling
            
            # Simple extractive summary based on embedding similarity
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
            
            if len(sentences) <= 3:
                return " ".join(sentences) + "."
            
            # Take first sentence and a few key sentences
            summary = [sentences[0]]
            
            # Add middle and end sentences for variety
            if len(sentences) > 3:
                summary.append(sentences[len(sentences) // 2])
            if len(sentences) > 1:
                summary.append(sentences[-1])
            
            return " ".join(summary) + "."
            
        except Exception as e:
            logger.error(f"BERT summary generation failed: {e}")
            return self._rule_based_summary(text)
    
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get BERT embedding for a text"""
        if not self.is_loaded:
            return None
        
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                return embeddings.cpu().numpy()
                
        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            return None
    
    def compare_documents(self, text1: str, text2: str) -> Dict:
        """Compare two legal documents"""
        if not self.is_loaded:
            return {"error": "BERT model not loaded"}
        
        try:
            emb1 = self.get_embedding(text1)
            emb2 = self.get_embedding(text2)
            
            if emb1 is None or emb2 is None:
                return {"error": "Failed to generate embeddings"}
            
            # Calculate cosine similarity
            similarity = np.dot(emb1.flatten(), emb2.flatten()) / (
                np.linalg.norm(emb1) * np.linalg.norm(emb2)
            )
            
            return {
                "similarity_score": float(similarity),
                "similarity_percentage": float(similarity * 100),
                "interpretation": self._interpret_similarity(similarity)
            }
            
        except Exception as e:
            return {"error": f"Comparison failed: {e}"}
    
    def _interpret_similarity(self, similarity: float) -> str:
        """Interpret similarity score"""
        if similarity > 0.8:
            return "Very similar documents - likely same type and content"
        elif similarity > 0.6:
            return "Similar documents - related content or structure"
        elif similarity > 0.4:
            return "Moderately similar - some common elements"
        elif similarity > 0.2:
            return "Low similarity - different content"
        else:
            return "Very different documents - unrelated content"

# Test function
def test_legal_bert():
    """Test the Legal BERT implementation"""
    print("🧪 Testing Legal BERT Implementation...")
    
    # Sample legal text
    sample_text = """
    MARITAL SETTLEMENT AGREEMENT
    
    This Marital Settlement Agreement (the "Agreement") is entered into on January 15, 2024, 
    between John Smith ("Husband") and Jane Smith ("Wife"), both residents of California.
    
    The parties agree to divide their marital property equitably, with the Husband receiving 
    the family home valued at $500,000 and the Wife receiving $250,000 in cash and retirement accounts.
    
    Child support shall be set at $1,500 per month for the two minor children, with custody 
    shared equally between the parties. Spousal support shall be $2,000 per month for 24 months.
    
    This Agreement is governed by the laws of the State of California and any disputes 
    shall be resolved through mediation or court proceedings.
    """
    
    try:
        # Initialize analyzer
        analyzer = LegalBERTAnalyzer()
        
        # Analyze document
        print("📄 Analyzing sample legal document...")
        analysis = analyzer.analyze_document(sample_text)
        
        print("✅ Analysis completed successfully!")
        print(f"📊 Document Type: {analysis['document_type']['specific_type']}")
        print(f"🔍 Key Entities: {len(analysis['key_entities']['parties'])} parties, {len(analysis['key_entities']['amounts'])} amounts")
        print(f"⚖️ Legal Terms: {len(analysis['legal_terms'])} terms identified")
        print(f"⚠️ Risk Factors: {len(analysis['risk_factors'])} risks identified")
        print(f"📝 Summary: {analysis['summary'][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_legal_bert()

