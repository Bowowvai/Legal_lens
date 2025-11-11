#!/usr/bin/env python3
"""
Quick test of the /analyze-pdf endpoint
"""
import requests
import json

# Test with the MaritalSettlementAgreement.pdf file
pdf_path = "/Users/vaibhav/Desktop/5th sem/Law/MaritalSettlementAgreement.pdf"

print("Testing /analyze-pdf endpoint...")
print(f"PDF file: {pdf_path}")
print()

try:
    with open(pdf_path, 'rb') as f:
        files = {'file': ('MaritalSettlementAgreement.pdf', f, 'application/pdf')}
        data = {'lang': 'eng'}
        
        print("Sending request to http://localhost:8000/analyze-pdf...")
        response = requests.post('http://localhost:8000/analyze-pdf', files=files, data=data)
        
        print(f"Status Code: {response.status_code}")
        print()
        
        if response.status_code == 200:
            result = response.json()
            print("=" * 80)
            print("SUCCESS! Response received:")
            print("=" * 80)
            print()
            
            # Display key information
            print(f"📄 Pages: {result.get('pages', 'N/A')}")
            print(f"🖼️  Images: {result.get('images', 'N/A')}")
            print(f"📝 Text Length: {len(result.get('text', ''))} characters")
            print(f"🔍 OCR Used: {result.get('used_ocr', False)}")
            print()
            
            # Summary
            summary = result.get('summary')
            if summary:
                print("📋 SUMMARY:")
                print("-" * 80)
                print(summary[:500] + ("..." if len(summary) > 500 else ""))
                print()
            else:
                print("⚠️  No summary generated")
                print()
            
            # Legal BERT Analysis
            if 'legal_bert_analysis' in result:
                bert = result['legal_bert_analysis']
                if not bert.get('error'):
                    print("🤖 LEGAL BERT ANALYSIS:")
                    print("-" * 80)
                    print(f"Document Type: {bert.get('document_type', {}).get('specific_type', 'N/A')}")
                    print(f"Category: {bert.get('document_type', {}).get('category', 'N/A')}")
                    print(f"Confidence: {bert.get('document_type', {}).get('confidence', 'N/A')}")
                    
                    parties = bert.get('key_entities', {}).get('parties', [])
                    if parties:
                        print(f"Parties: {', '.join(parties[:3])}")
                    print()
            
            # Structured Data
            if result.get('structured_data'):
                print("🏗️  STRUCTURED DATA:")
                print("-" * 80)
                for key, value in list(result['structured_data'].items())[:5]:
                    if value:
                        print(f"  {key}: {value}")
                print()
            
            # Text preview
            text_preview = result.get('text', '')[:300]
            if text_preview:
                print("📄 TEXT PREVIEW (first 300 chars):")
                print("-" * 80)
                print(text_preview)
                print()
            
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
except FileNotFoundError:
    print(f"❌ PDF file not found: {pdf_path}")
except requests.exceptions.ConnectionError:
    print("❌ Could not connect to backend server at http://localhost:8000")
    print("   Make sure the backend is running!")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
