#!/usr/bin/env python3
"""
Start the Themis Summarizer API with enhanced legal document understanding capabilities
"""
import os
import sys
import time
import subprocess

print("🚀 Starting Themis Summarizer API")

# Check if transformers and torch are installed
try:
    import transformers
    import torch
    print(f"✅ Transformers {transformers.__version__} and PyTorch {torch.__version__} available")
except ImportError:
    print("❌ Missing required packages")
    print("💡 Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers", "torch", "accelerate"])

# Check if the legal model is available in cache
cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "law-ai-inlegalbert")
if not os.path.exists(cache_dir) or len(os.listdir(cache_dir)) == 0:
    print("⚠️ InLegalBERT model not found in cache")
    print("💡 Running test_legal_bert.py to configure the model...")
    subprocess.check_call([sys.executable, "test_legal_bert.py"])
else:
    print("✅ InLegalBERT model found in cache")

# Start the API server
print("\n🚀 Starting API server...")
os.environ["TRANSFORMERS_VERBOSITY"] = "info"
os.execv(sys.executable, [sys.executable, "summarizer_api.py"])
