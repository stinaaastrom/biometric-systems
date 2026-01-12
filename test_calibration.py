"""
Quick script to test temperature scaling calibration on the saved model.
"""
import sys
sys.path.append('src')

from age_prediction import AgePrediction, AGE_MODEL_PATH

print("Loading model and test data...")
ap = AgePrediction()
ap.load_model(AGE_MODEL_PATH)
ap.load_test_dataset()

print("\nRunning temperature calibration...")
ap.cnn_model.calibrate_temperature()

print("\nSaving model with calibrated temperature...")
ap.save_model(AGE_MODEL_PATH)

print("\n" + "="*60)
print("CALIBRATION COMPLETE AND SAVED")
print("="*60)
print("\nNext steps:")
print("  1. Test predictions with calibrated confidence:")
print("     python src/age_prediction.py -> option 3")
print(f"\nCalibrated temperature: {ap.cnn_model.temperature:.3f}")
print("="*60)
