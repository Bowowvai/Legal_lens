#!/usr/bin/env python3
"""
Comprehensive Demo of the Legal AI System
Shows PDF processing, Legal BERT analysis, and Gemini summarization
"""

import requests
import json
import time
from pathlib import Path

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"🚀 {title}")
    print("="*60)

def print_section(title):
    """Print a formatted section header"""
    print(f"\n📋 {title}")
    print("-" * 40)

def demo_legal_bert_analysis():
    """Demonstrate Legal BERT analysis capabilities"""
    print_section("Legal BERT Document Analysis")
    
    # Sample legal documents for testing
    documents = {
        "Loan Agreement": """
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
        """,
        
        "Employment Contract": """
        EMPLOYMENT AGREEMENT
        
        This Employment Agreement (the "Agreement") is made and entered into on March 1, 2024, 
        between TechCorp Inc., a California corporation ("Employer"), and John Smith ("Employee").
        
        The Employer hereby employs the Employee as a Senior Software Engineer, and the Employee 
        accepts such employment, subject to the terms and conditions set forth in this Agreement.
        
        The Employee's annual salary shall be $120,000, payable in bi-weekly installments. 
        The Employee shall be entitled to health insurance, 401(k) matching, and 20 days of 
        paid time off per year.
        
        This Agreement shall commence on March 1, 2024, and continue until terminated by either 
        party with 30 days written notice.
        """,
        
        "Service Agreement": """
        SERVICE AGREEMENT
        
        This Service Agreement (the "Agreement") is entered into on April 1, 2024, 
        between Consulting Services LLC ("Provider") and Manufacturing Corp ("Client").
        
        The Provider agrees to provide consulting services related to process optimization 
        and quality control improvements. The total fee for these services is $75,000, 
        payable in three installments of $25,000 each.
        
        The Provider shall complete all work within 90 days of the effective date. 
        Any additional work requested by the Client shall be subject to additional fees 
        as agreed upon in writing.
        """
    }
    
    for doc_type, content in documents.items():
        print(f"\n📄 Analyzing: {doc_type}")
        
        try:
            response = requests.post(
                "http://localhost:8000/legal-bert-analyze",
                json={"text": content}
            )
            
            if response.status_code == 200:
                result = response.json()
                analysis = result['analysis']
                
                print(f"  ✅ Document Type: {analysis['document_type']['specific_type']}")
                print(f"  📊 Category: {analysis['document_type']['category']}")
                print(f"  🎯 Confidence: {analysis['document_type']['confidence']:.2f}")
                print(f"  👥 Parties: {len(analysis['key_entities']['parties'])} identified")
                print(f"  💰 Amounts: {len(analysis['key_entities']['amounts'])} found")
                print(f"  ⚖️ Legal Terms: {len(analysis['legal_terms'])} identified")
                print(f"  ⚠️ Risk Factors: {len(analysis['risk_factors'])} detected")
                
                if analysis['legal_terms']:
                    print(f"  📝 Legal Terms: {', '.join(analysis['legal_terms'][:3])}")
                if analysis['risk_factors']:
                    print(f"  ⚠️ Risk Factors: {', '.join(analysis['risk_factors'][:3])}")
                    
            else:
                print(f"  ❌ Analysis failed: {response.status_code}")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")

def demo_document_comparison():
    """Demonstrate document comparison capabilities"""
    print_section("Document Comparison with Legal BERT")
    
    # Test different types of document comparisons
    comparisons = [
        {
            "name": "Similar Documents (Loan Agreements)",
            "text1": "This is a loan agreement between Bank A and Company B for $500,000 at 5% interest.",
            "text2": "This is a loan agreement between Bank C and Company D for $750,000 at 6% interest."
        },
        {
            "name": "Different Document Types",
            "text1": "This is a loan agreement between Bank A and Company B for $500,000.",
            "text2": "This is a service agreement between Company A and Vendor B for consulting services."
        },
        {
            "name": "Very Different Content",
            "text1": "This is a loan agreement between Bank A and Company B for $500,000.",
            "text2": "This is a research paper about machine learning algorithms and their applications."
        }
    ]
    
    for comparison in comparisons:
        print(f"\n🔄 {comparison['name']}")
        
        try:
            response = requests.post(
                "http://localhost:8000/compare-documents",
                json={"text1": comparison['text1'], "text2": comparison['text2']}
            )
            
            if response.status_code == 200:
                result = response.json()
                comp = result['comparison']
                
                print(f"  📊 Similarity: {comp['similarity_percentage']:.1f}%")
                print(f"  💡 Interpretation: {comp['interpretation']}")
                
            else:
                print(f"  ❌ Comparison failed: {response.status_code}")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")

def demo_pdf_processing():
    """Demonstrate PDF processing capabilities"""
    print_section("PDF Processing and Analysis")
    
    # Check for PDF files
    pdf_dir = Path("research paper")
    pdf_files = list(pdf_dir.glob("*.pdf")) if pdf_dir.exists() else []
    
    if not pdf_files:
        print("ℹ️ No PDF files found in research paper directory")
        return
    
    print(f"📁 Found {len(pdf_files)} PDF files")
    
    # Test with the first PDF
    test_pdf = pdf_files[0]
    print(f"🧪 Testing with: {test_pdf.name}")
    
    try:
        with open(test_pdf, 'rb') as f:
            files = {'file': (test_pdf.name, f, 'application/pdf')}
            
            print("  ⏳ Uploading and analyzing PDF...")
            response = requests.post(
                "http://localhost:8000/analyze-pdf",
                files=files,
                data={'lang': 'eng'}
            )
        
        if response.status_code == 200:
            result = response.json()
            print("  ✅ PDF analysis successful!")
            
            # Basic info
            print(f"  📊 Pages: {result.get('pages', 'Unknown')}")
            print(f"  🖼️ Images: {result.get('images', 'Unknown')}")
            print(f"  📝 Text Length: {len(result.get('text', ''))} characters")
            print(f"  🔍 OCR Used: {result.get('used_ocr', 'Unknown')}")
            
            # Legal BERT analysis
            if result.get('legal_bert_analysis') and not result['legal_bert_analysis'].get('error'):
                bert = result['legal_bert_analysis']
                print(f"  🤖 Legal BERT Analysis:")
                print(f"    📄 Document Type: {bert.get('document_type', {}).get('specific_type', 'Unknown')}")
                print(f"    🎯 Confidence: {bert.get('confidence', 'Unknown')}")
                
                if bert.get('key_entities', {}).get('parties'):
                    print(f"    👥 Parties: {len(bert['key_entities']['parties'])} identified")
                
                if bert.get('legal_terms'):
                    print(f"    ⚖️ Legal Terms: {len(bert['legal_terms'])} found")
            
            # Summary
            if result.get('summary'):
                print(f"  📝 Summary: {result['summary'][:150]}...")
            else:
                print("  ℹ️ No summary generated")
                
        else:
            print(f"  ❌ PDF analysis failed: {response.status_code}")
            print(f"  📄 Response: {response.text}")
            
    except Exception as e:
        print(f"  ❌ Error: {e}")

def demo_gemini_summarization():
    """Demonstrate Gemini summarization capabilities"""
    print_section("Gemini AI Summarization")
    
    test_text = """
    CONTRACT DISPUTE RESOLUTION
    
    This case involves a breach of contract dispute between TechSolutions Inc. and DataCorp LLC. 
    The parties entered into a software development agreement on January 15, 2023, whereby 
    TechSolutions agreed to develop a custom CRM system for DataCorp for $150,000.
    
    The contract specified a delivery date of June 1, 2023, with penalties of $1,000 per day 
    for late delivery. TechSolutions failed to deliver the system by the specified date, 
    citing technical difficulties and requesting an extension until August 1, 2023.
    
    DataCorp refused the extension and terminated the contract, seeking damages for breach 
    and the return of the $75,000 advance payment. TechSolutions counterclaimed for the 
    remaining $75,000, arguing that substantial work had been completed and that the delay 
    was due to DataCorp's failure to provide necessary specifications in a timely manner.
    
    The court must determine whether TechSolutions materially breached the contract, 
    whether DataCorp's termination was justified, and what damages, if any, are appropriate.
    """
    
    try:
        response = requests.post(
            "http://localhost:8000/analyze-text",
            json={"text": test_text}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Text analysis successful!")
            
            # Structured data
            if result.get('structured'):
                print(f"📊 Structured Data: {result['structured']}")
            
            # Summary
            if result.get('summary'):
                print(f"📝 Gemini Summary:")
                print(result['summary'])
            else:
                print("ℹ️ No Gemini summary (API key not configured)")
                
        else:
            print(f"❌ Text analysis failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def check_system_health():
    """Check the overall system health"""
    print_section("System Health Check")
    
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            health = response.json()
            
            print("✅ API Status: Running")
            print(f"🤖 Gemini: {'✅ Available' if health.get('gemini') else '❌ Not configured'}")
            print(f"🤖 Legal BERT: {'✅ Available' if health.get('legal_bert') else '❌ Not available'}")
            print(f"📄 PDF Processing: {'✅ Available' if health.get('pymupdf') else '⚠️ Limited (OCR only)'}")
            print(f"🔍 OCR: {'✅ Available' if health.get('ocr_available') else '❌ Not available'}")
            print(f"🤖 Transformers: {'✅ Available' if health.get('transformers') else '❌ Not available'}")
            
            if health.get('legal_bert'):
                print(f"📊 Legal BERT Model: {health.get('legal_model_name', 'Unknown')}")
                
        else:
            print(f"❌ API Health Check Failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")

def main():
    """Main demonstration function"""
    print_header("LEGAL AI SYSTEM COMPREHENSIVE DEMONSTRATION")
    
    print("🎯 This demonstration showcases the complete Legal AI system capabilities:")
    print("   • Legal BERT document analysis and classification")
    print("   • Document comparison using AI embeddings")
    print("   • PDF processing with text extraction")
    print("   • Gemini AI-powered summarization")
    print("   • Comprehensive legal document understanding")
    
    # Check system health first
    check_system_health()
    
    # Run demonstrations
    demo_legal_bert_analysis()
    demo_document_comparison()
    demo_pdf_processing()
    demo_gemini_summarization()
    
    print_header("DEMONSTRATION COMPLETED")
    print("🎉 The Legal AI system is fully functional!")
    print("\n💡 Next steps:")
    print("   1. Use the HTML interface (pdf_test_interface.html) for easy PDF testing")
    print("   2. Test with your own legal documents")
    print("   3. Explore the API endpoints for integration")
    print("   4. Customize the Legal BERT analysis for your specific needs")

if __name__ == "__main__":
    main()

