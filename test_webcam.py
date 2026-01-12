#!/usr/bin/env python
"""
Webcam age prediction test - captures frames and saves predictions to disk.
Auto-captures 3 frames with 2-second intervals.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import cv2
import numpy as np
import time

print("Initializing webcam test...")
from age_prediction import AgePrediction, AGE_MODEL_PATH
from dataset import IMAGE_SIZE
from image_processing import ImageProcesser
from cnn_model import AGE_CLASSES

# Load model
print("Loading model...")
ap = AgePrediction()
ap.load_model(AGE_MODEL_PATH)
print("✓ Model ready\n")

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    sys.exit(1)

# Create output directory
output_dir = "webcam_predictions"
os.makedirs(output_dir, exist_ok=True)

print("="*60)
print("WEBCAM AGE PREDICTION TEST")
print("="*60)
print("Auto-capturing 3 frames with 2-second intervals...")
print("Look at the camera!")
print("="*60)

image_processor = ImageProcesser()

# Warm-up: skip first few frames
for _ in range(10):
    cap.read()

time.sleep(1)

for capture_num in range(1, 4):
    print(f"\nCapture {capture_num}/3 in 2 seconds...")
    time.sleep(2)
    
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        continue
    
    print("Captured! Processing...")
    
    try:
        # Save original frame
        frame_path = os.path.join(output_dir, f"capture_{capture_num}_original.jpg")
        cv2.imwrite(frame_path, frame)
        
        # Process image
        face_img = image_processor.detect_crop_faces(frame)
        if face_img is None:
            print("⚠ No face detected. Try adjusting position/lighting.")
            continue
        
        face_img = image_processor.image_enhancements(face_img)
        face_img_resized = cv2.resize(face_img, IMAGE_SIZE)
        
        # Save cropped face
        face_path = os.path.join(output_dir, f"capture_{capture_num}_face.jpg")
        cv2.imwrite(face_path, face_img_resized)
        
        # Normalize
        face_img_normalized = face_img_resized / 255.0
        
        # Simple TTA: original + flip
        aug_views = [face_img_normalized, np.flip(face_img_normalized, axis=1)]
        face_batch = np.stack(aug_views, axis=0).reshape(len(aug_views), IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
        
        # Predict
        pred_probs, pred_reg = ap.cnn_model.model.predict(face_batch, verbose=0)
        
        # Get mean predictions
        mean_probs = pred_probs.mean(axis=0)
        mean_reg_age = pred_reg.mean()
        
        # Apply boundary nudging
        mean_probs = ap.cnn_model.nudge_probabilities_by_regression(mean_probs, mean_reg_age, boundary_threshold=1.5)
        
        predicted_class = np.argmax(mean_probs)
        confidence = mean_probs[predicted_class] * 100
        
        # Get age range
        min_age, max_age = AGE_CLASSES[predicted_class]
        if max_age is None:
            age_range = f"{min_age}+"
        else:
            age_range = f"{min_age}-{max_age}"
        
        est_age = float(mean_reg_age)
        
        print(f"\n✓ PREDICTION #{capture_num}:")
        print(f"  Age class: {age_range}")
        print(f"  Confidence: {confidence:.1f}%")
        print(f"  Regression estimate: {est_age:.1f} years")
        
        # Reject warnings
        if confidence < 35.0:
            print(f"  ⚠ Low confidence - manual review recommended")
        
        # Show top-2
        top2_idx = np.argsort(mean_probs)[-2:][::-1]
        if len(AGE_CLASSES) > top2_idx[1]:
            alt_min, alt_max = AGE_CLASSES[top2_idx[1]]
            alt_range = f"{alt_min}+" if alt_max is None else f"{alt_min}-{alt_max}"
            alt_conf = mean_probs[top2_idx[1]] * 100
            print(f"  Alternative: {alt_range} ({alt_conf:.1f}%)")
        
        # Create result image with text overlay
        result_frame = frame.copy()
        cv2.putText(result_frame, f"Age: {age_range} ({confidence:.0f}%)", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(result_frame, f"Reg: {est_age:.1f} years", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        result_path = os.path.join(output_dir, f"capture_{capture_num}_result.jpg")
        cv2.imwrite(result_path, result_frame)
        print(f"  Saved: {result_path}")
        
    except Exception as e:
        print(f"✗ Error on capture {capture_num}: {e}")
        import traceback
        traceback.print_exc()

cap.release()
print("\n" + "="*60)
print(f"✓ Test complete! Results saved to: {output_dir}/")
print("="*60)
