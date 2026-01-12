#!/usr/bin/env python
"""
Test script for boundary nudging predictions - no menu, no input() blocking.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import cv2

print("Loading model and processing image...")
from age_prediction import AgePrediction, AGE_MODEL_PATH
from dataset import IMAGE_SIZE

# Load model
ap = AgePrediction()
ap.load_model(AGE_MODEL_PATH)

# Test image path
test_image = r"C:\Users\Robin Rienks Hestad\Downloads\biometricsjulie.jpg"

if not os.path.exists(test_image):
    print(f"Image not found: {test_image}")
    print("Please update the test_image path in this script.")
    sys.exit(1)

print(f"\nTesting prediction with boundary nudging on: {test_image}")
print("="*60)

try:
    ap.predict_age(test_image)
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("="*60)
print("Test complete!")
