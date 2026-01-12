#!/usr/bin/env python
"""
Direct calibration and save script - no menu, no input() blocking.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("Initializing...")
from age_prediction import AgePrediction, AGE_MODEL_PATH

print("Loading model...")
ap = AgePrediction()
ap.load_model(AGE_MODEL_PATH)

print("Loading test dataset...")
ap.load_test_dataset()

print("Calibrating temperature...")
ap.cnn_model.calibrate_temperature()

print("\nSaving model with calibrated temperature...")
ap.save_model(AGE_MODEL_PATH)

print("\n" + "="*60)
print("✓ CALIBRATION COMPLETE AND SAVED")
print("="*60)
print(f"Temperature: {ap.cnn_model.temperature:.3f}")
print(f"Model: {AGE_MODEL_PATH}")
print("="*60)
print("\nReady for predictions!")
