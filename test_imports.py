#!/usr/bin/env python3

print("Testing Python packages...")

# Test basic imports
try:
    import torch
    print(f"✅ PyTorch version: {torch.__version__}")
except ImportError:
    print("❌ PyTorch not installed")

try:
    import transformers
    print(f"✅ Transformers version: {transformers.__version__}")
except ImportError:
    print("❌ Transformers not installed")

try:
    import requests
    print("✅ Requests available")
except ImportError:
    print("❌ Requests not installed")
