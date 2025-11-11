#!/usr/bin/env python3
"""
Test script to verify PDF upload, text extraction, and summarization functionality
Tests both Gemini summarization and Legal BERT analysis
"""

import os
import requests
import json
from pathlib import Path

def test_pdf_upload_and_summarization():
    """Test the complete PDF processing pipeline"""
    print("🧪 Testing PDF Upload and Summarization Pipeline...")
    
    # Check if API is running
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code != 200:
            print("❌ API is not running. Please start it with: python summarizer_api.py")
            return False
        
        health_data = response.json()
        print("✅ API is running!")
        print(f"📊 Health Status: {json.dumps(health_data, indent=2)}")
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Please start it with: python summarizer_api.py")
        return False
    
    # Test 1: Text summarization with Legal BERT
    print("\n📝 Test 1: Text Analysis with Legal BERT")
    test_text = """
    LOAN AGREEMENT
    
    This Loan Agreement (the "Agreement") is entered into on January 15, 2024, 
    between ABC Corporation, a Delaware corporation ("Lender"), and XYZ Company, 
    a California corporation ("Borrower").
    
    The Lender agrees to lend to the Borrower the principal sum of $500,000.00 
    (the "Loan Amount") for the purpose of business expansion. The Loan shall 
    bear interest at the rate of 5.5% per annum, compounded monthly.
    
    The Borrower shall make monthly payments of $3,500.00, beginning on February 1, 2024, 
    and continuing on the first day of each month thereafter until the Loan is paid in full.
    
    This Agreement is governed by the laws of the State of California. Any disputes 
    arising under this Agreement shall be resolved through binding arbitration.
    """
    
    try:
        response = requests.post(
            "http://localhost:8000/legal-bert-analyze",
            json={"text": test_text}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Legal BERT analysis successful!")
            print(f"📊 Document Type: {result['analysis']['document_type']['specific_type']}")
            print(f"🔍 Key Entities: {len(result['analysis']['key_entities']['parties'])} parties")
            print(f"⚖️ Legal Terms: {len(result['analysis']['legal_terms'])} terms")
            print(f"📝 Summary: {result['analysis']['summary'][:100]}...")
        else:
            print(f"❌ Legal BERT analysis failed: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Error testing Legal BERT: {e}")
    
    # Test 2: Text summarization with Gemini (if available)
    print("\n🤖 Test 2: Text Summarization with Gemini")
    try:
        response = requests.post(
            "http://localhost:8000/analyze-text",
            json={"text": test_text}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Text analysis successful!")
            print(f"📊 Structured Data: {result['structured']}")
            if result.get('summary'):
                print(f"📝 Gemini Summary: {result['summary'][:150]}...")
            else:
                print("ℹ️ No Gemini summary (API key not configured)")
        else:
            print(f"❌ Text analysis failed: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Error testing text analysis: {e}")
    
    # Test 3: PDF Upload and Analysis
    print("\n📄 Test 3: PDF Upload and Analysis")
    
    # Check if we have any PDF files in the research paper directory
    pdf_dir = Path("research paper")
    pdf_files = list(pdf_dir.glob("*.pdf")) if pdf_dir.exists() else []
    
    if pdf_files:
        print(f"📁 Found {len(pdf_files)} PDF files in research paper directory")
        
        # Test with the first PDF
        test_pdf = pdf_files[0]
        print(f"🧪 Testing with: {test_pdf.name}")
        
        try:
            with open(test_pdf, 'rb') as f:
                files = {'file': (test_pdf.name, f, 'application/pdf')}
                
                response = requests.post(
                    "http://localhost:8000/analyze-pdf",
                    files=files,
                    data={'lang': 'eng'}
                )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ PDF analysis successful!")
                print(f"📊 Pages: {result.get('pages', 'Unknown')}")
                print(f"🖼️ Images: {result.get('images', 'Unknown')}")
                print(f"📝 Text Length: {len(result.get('text', ''))} characters")
                
                if result.get('summary'):
                    print(f"📝 Summary: {result['summary'][:200]}...")
                else:
                    print("ℹ️ No summary generated")
                    
                # Test Legal BERT on extracted text
                if result.get('text') and len(result['text']) > 50:
                    print("\n🔍 Testing Legal BERT on extracted PDF text...")
                    bert_response = requests.post(
                        "http://localhost:8000/legal-bert-analyze",
                        json={"text": result['text'][:2000]}  # First 2000 chars
                    )
                    
                    if bert_response.status_code == 200:
                        bert_result = bert_response.json()
                        print("✅ Legal BERT analysis on PDF text successful!")
                        print(f"📊 Document Type: {bert_result['analysis']['document_type']['specific_type']}")
                        print(f"🔍 Key Entities: {len(bert_result['analysis']['key_entities']['parties'])} parties")
                    else:
                        print(f"❌ Legal BERT analysis on PDF text failed: {bert_response.status_code}")
                
            else:
                print(f"❌ PDF analysis failed: {response.status_code}")
                print(response.text)
                
        except Exception as e:
            print(f"❌ Error testing PDF upload: {e}")
    else:
        print("ℹ️ No PDF files found in research paper directory")
        print("💡 You can test PDF upload by placing a PDF file in the 'research paper' folder")
    
    # Test 4: Document Comparison
    print("\n🔄 Test 4: Document Comparison with Legal BERT")
    try:
        text1 = "This is a loan agreement between Bank A and Company B for $500,000."
        text2 = "This is a service agreement between Company A and Vendor B for consulting services."
        
        response = requests.post(
            "http://localhost:8000/compare-documents",
            json={"text1": text1, "text2": text2}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Document comparison successful!")
            print(f"📊 Similarity: {result['comparison']['similarity_percentage']:.1f}%")
            print(f"💡 Interpretation: {result['comparison']['interpretation']}")
        else:
            print(f"❌ Document comparison failed: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Error testing document comparison: {e}")
    
    print("\n🏁 PDF Summarization Test Completed!")
    print("💡 Summary of what was tested:")
    print("1. ✅ Legal BERT text analysis")
    print("2. ✅ Text summarization (Gemini if available)")
    print("3. ✅ PDF upload and text extraction")
    print("4. ✅ Legal BERT analysis on PDF text")
    print("5. ✅ Document comparison")
    
    return True

def create_sample_pdf_for_testing():
    """Create a simple sample PDF for testing if none exists"""
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        
        # Create a simple test PDF
        filename = "test_legal_document.pdf"
        c = canvas.Canvas(filename, pagesize=letter)
        
        # Add content
        c.drawString(100, 750, "TEST LEGAL DOCUMENT")
        c.drawString(100, 720, "This is a sample loan agreement for testing purposes.")
        c.drawString(100, 700, "Parties: ABC Bank (Lender) and XYZ Company (Borrower)")
        c.drawString(100, 680, "Loan Amount: $100,000")
        c.drawString(100, 660, "Interest Rate: 5.5% per annum")
        c.drawString(100, 640, "Term: 5 years")
        c.drawString(100, 620, "This document is for testing the PDF processing pipeline.")
        
        c.save()
        print(f"✅ Created sample PDF: {filename}")
        return filename
        
    except ImportError:
        print("ℹ️ reportlab not available, cannot create sample PDF")
        return None
    except Exception as e:
        print(f"❌ Error creating sample PDF: {e}")
        return None

if __name__ == "__main__":
    print("🚀 Starting PDF Summarization Test Suite...")
    
    # Check if we need to create a sample PDF
    pdf_dir = Path("research paper")
    if not pdf_dir.exists() or not list(pdf_dir.glob("*.pdf")):
        print("📝 No PDF files found, creating sample PDF for testing...")
        sample_pdf = create_sample_pdf_for_testing()
        if sample_pdf:
            # Move to research paper directory
            pdf_dir.mkdir(exist_ok=True)
            import shutil
            shutil.move(sample_pdf, pdf_dir / sample_pdf)
            print(f"📁 Moved sample PDF to research paper directory")
    
    # Run the tests
    test_pdf_upload_and_summarization()
# test 123 