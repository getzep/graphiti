#!/usr/bin/env python3

import sys
import os
sys.path.append('/c/pawelz-workspace/graphiti/server')

# Direct test of FastCoref
from fastcoref import FCoref
import spacy

def test_fastcoref_basic():
    print("Testing FastCoref basic functionality...")
    
    try:
        # Initialize model
        model = FCoref(device='cpu')
        
        # Test prediction
        text = "John went to the store. He bought milk."
        predictions = model.predict(texts=[text])
        
        print(f"Input: {text}")
        print(f"Predictions type: {type(predictions)}")
        print(f"Predictions: {predictions}")
        
        if predictions and len(predictions) > 0:
            result = predictions[0]
            print(f"Result type: {type(result)}")
            print(f"Result attributes: {dir(result)}")
            print(f"Text: {getattr(result, 'text', 'N/A')}")
            print(f"Clusters: {getattr(result, 'clusters', 'N/A')}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_fastcoref_basic()
