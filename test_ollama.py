#!/usr/bin/env python3
"""
Test script for Ollama + Moondream integration
"""

import os
import sys
import requests
import traceback
from PIL import Image

# Add the app directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_ollama_connection():
    """Test if Ollama is running and accessible"""
    print("Testing Ollama connection...")
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json()
            print("OK: Ollama is running")
            print(f"  Available models: {len(models.get('models', []))}")

            # Check if moondream is available
            model_names = [model['name'] for model in models.get('models', [])]
            if any('moondream' in name for name in model_names):
                print("OK: Moondream model found")
                return True
            else:
                print("ERROR: Moondream model not found")
                print("  Run: ollama pull moondream:latest")
                return False
        else:
            print(f"ERROR: Ollama responded with status: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to Ollama")
        print("  Make sure Ollama is running: ollama serve")
        return False
    except Exception as e:
        print(f"ERROR: Error checking Ollama: {e}")
        return False

def test_app_imports():
    """Test if app modules can be imported"""
    print("Testing app imports...")
    try:
        from app import query_ollama_api, analyze_image, image_to_base64
        print("OK: App modules imported successfully")
        return True
    except ImportError as e:
        print(f"ERROR: Failed to import app modules: {e}")
        return False

def test_image_analysis():
    """Test image analysis with Ollama"""
    print("Testing image analysis...")
    try:
        # Create a test image
        test_image = Image.new('RGB', (100, 100), color='blue')
        print("  Created test image")

        from app import analyze_image

        # Test the analyze_image function
        elements, summary = analyze_image(test_image)

        if elements and summary:
            print("OK: Image analysis completed successfully")
            print(f"  Summary length: {len(summary)} characters")
            return True
        else:
            print("ERROR: Image analysis returned empty results")
            return False

    except Exception as e:
        print(f"ERROR: Image analysis failed: {e}")
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests"""
    print("Testing Ollama + Moondream Integration\n")
    print("=" * 50)

    tests = [
        test_ollama_connection,
        test_app_imports,
        test_image_analysis,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
                print("PASSED")
            else:
                failed += 1
                print("FAILED")
        except Exception as e:
            print(f"CRASHED: {e}")
            failed += 1
        print("-" * 30)

    print(f"\nTest Summary:")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")

    if failed == 0:
        print("\nAll tests passed!")
        print("Ready to run the app: python app.py")
        return True
    else:
        print(f"\n{failed} test(s) failed!")
        print("Fix the issues above before running the app.")
        return False

if __name__ == "__main__":
    success = run_all_tests()

    if not success:
        sys.exit(1)

    print("\nReady for deployment!")