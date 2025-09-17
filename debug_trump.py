#!/usr/bin/env python3
"""Debug Trump detection logic."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def debug_trump_logic(raw_text: str):
    """Debug the Trump detection flow."""
    print(f"Input: '{raw_text}'")
    print(f"Lowercase: '{raw_text.lower()}'")
    
    # Main detection
    if 'trump' in raw_text.lower():
        print("✅ Trump detected - will call fetch_trump_specialized()")
        
        # Sub-detection within fetch_trump_specialized
        text_lower = raw_text.lower()
        if 'reuters' in text_lower:
            print("✅ Reuters detected - will call fetch_trump_from_source('reuters')")
        elif 'bbc' in text_lower:
            print("✅ BBC detected - will call fetch_trump_from_source('bbc')")
        elif 'cnn' in text_lower:
            print("✅ CNN detected - will call fetch_trump_from_source('cnn')")
        else:
            print("✅ No specific source - will use NYTTrumpFetch")
    else:
        print("❌ No Trump detected")
    
    print()

if __name__ == "__main__":
    test_cases = [
        "Reuters trump",
        "BBC trump", 
        "CNN trump",
        "Find trump articles",
        "trump news",
        "NYT technology"
    ]
    
    for case in test_cases:
        debug_trump_logic(case)