#!/usr/bin/env python3
"""
Test script to check Legal BERT installation and functionality
"""
import os
import time
import sys

print("💼 Testing Legal BERT integration...")

# Test 1: Check if PyTorch is available
try:
    import torch
    print(f"✅ PyTorch version: {torch.__version__}")
    torch_available = True
except ImportError as e:
    print(f"❌ PyTorch not available: {e}")
    print("💡 Try: pip install torch")
    torch_available = False
    sys.exit(1)

# Test 2: Check if transformers is available
try:
    # Import the base transformers module first
    import transformers
    print(f"✅ Transformers base module version: {transformers.__version__}")
    
    # Then try importing specific components
    try:
        from transformers import AutoTokenizer, AutoModel
        print("✅ AutoTokenizer and AutoModel imported successfully")
        
        # Also test pipeline but don't fail if it's not available
        try:
            from transformers import pipeline
            print("✅ Pipeline module available")
            pipeline_available = True
        except ImportError as e:
            print(f"⚠️ Pipeline module not available: {e}")
            print("💡 This is not critical for Legal BERT")
            pipeline_available = False
            
        transformers_available = True
    except ImportError as e:
        print(f"❌ AutoTokenizer/AutoModel not available: {e}")
        print("💡 This indicates a problem with transformers installation")
        transformers_available = False
        sys.exit(1)
except ImportError as e:
    print(f"❌ Transformers base module not available: {e}")
    print("💡 Try: pip install transformers")
    transformers_available = False
    sys.exit(1)

# Test 3: Check if we can load our working Legal BERT implementation
if torch_available and transformers_available:
    try:
        print("📥 Testing our working Legal BERT implementation...")
        
        # Import our working implementation
        from legal_bert_working import LegalBERTAnalyzer
        
        # Initialize the analyzer
        print("🔧 Initializing Legal BERT analyzer...")
        analyzer = LegalBERTAnalyzer()
        
        if analyzer.is_loaded:
            print("✅ Legal BERT model loaded successfully!")
            print(f"📊 Model: {analyzer.model_name}")
            print(f"🖥️ Device: {analyzer.device}")
            
            # Test 4: Try a simple inference
            test_text = "This contract is between Party A and Party B for the sale of goods."
            print(f"🧪 Testing with sample text: '{test_text}'")
            
            # Analyze the document
            analysis = analyzer.analyze_document(test_text)
            
            print("✅ Document analysis successful!")
            print(f"📊 Document Type: {analysis['document_type']['specific_type']}")
            print(f"🔍 Key Entities: {len(analysis['key_entities']['parties'])} parties")
            print(f"⚖️ Legal Terms: {len(analysis['legal_terms'])} terms identified")
            print(f"⚠️ Risk Factors: {len(analysis['risk_factors'])} risks identified")
            print(f"📝 Summary: {analysis['summary'][:100]}...")
            print(f"🎯 Confidence: {analysis['confidence']}")
            
            # Test document comparison
            print("\n🧪 Testing document comparison...")
            text1 = "This is a loan agreement between Bank A and Company B."
            text2 = "This is a service agreement between Company A and Vendor B."
            
            comparison = analyzer.compare_documents(text1, text2)
            if 'error' not in comparison:
                print(f"✅ Document comparison successful!")
                print(f"📊 Similarity: {comparison['similarity_percentage']:.1f}%")
                print(f"💡 Interpretation: {comparison['interpretation']}")
            else:
                print(f"⚠️ Document comparison failed: {comparison['error']}")
            
            print("\n🎯 Legal BERT is properly configured and working!")
            
        else:
            print("⚠️ Legal BERT model not loaded, but analyzer initialized with rule-based fallback")
            print("💡 This is still functional for basic legal document analysis")
            
            # Test rule-based analysis
            test_text = "This contract is between Party A and Party B for the sale of goods."
            analysis = analyzer.analyze_document(test_text)
            print(f"✅ Rule-based analysis working: {analysis['document_type']['specific_type']}")
        
    except Exception as e:
        print(f"❌ Error with Legal BERT: {e}")
        print("💡 Troubleshooting tips:")
        print("   - Check internet connection (model is ~400MB)")
        print("   - Ensure you have enough disk space")
        print("   - Try restarting your Python environment")
        print("   - Check if the legal_bert_working.py file is in the same directory")
        sys.exit(1)

# Test 5: Check current API status
try:
    import requests
    print("\n🔍 Checking if API is running...")
    response = requests.get("http://localhost:8000/health")
    if response.status_code == 200:
        health_data = response.json()
        print(f"🌐 API Health Status: {health_data}")
        if health_data.get("legal_bert", False):
            print("✅ API shows Legal BERT is enabled and working")
        else:
            print("⚠️ API shows Legal BERT is not enabled")
            print("💡 You need to restart the API server for changes to take effect")
    else:
        print("⚠️ API not responding. You need to start the API server.")
except Exception as e:
    print(f"⚠️ Cannot check API: {e}")
    print("💡 The API server might not be running")

print("\n🏁 Legal BERT test completed!")
print("💡 Next steps:")
print("1. Restart your API server with: python summarizer_api.py")
print("2. Check the API health endpoint again to confirm Legal BERT is enabled")
print("3. Test the new /legal-bert-analyze endpoint for enhanced legal document analysis")
