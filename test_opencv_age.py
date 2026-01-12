"""
Simple age estimation using OpenCV's pre-built age/gender detection
Based on Caffe models trained on ImDB-UTKFace
"""

import os
import cv2
import numpy as np
import time
import sys

sys.path.insert(0, 'src')
from image_processing import ImageProcesser

# Age groups for OpenCV model
AGE_GROUPS = ['(0-2)', '(4-6)', '(8-13)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# OpenCV pre-trained model paths
MODEL_DIR = "models_pretrained"

def test_opencv_age_detector():
    """Test OpenCV's built-in age/gender detection"""
    
    print("="*60)
    print("OPENCV AGE/GENDER DETECTOR TEST")
    print("="*60)
    
    # Check if models exist
    gender_proto = os.path.join(MODEL_DIR, "gender_deploy.prototxt")
    gender_model = os.path.join(MODEL_DIR, "gender_net.caffemodel")
    age_proto = os.path.join(MODEL_DIR, "age_deploy.prototxt")
    age_model = os.path.join(MODEL_DIR, "age_net.caffemodel")
    
    # Try to download if not found
    models_missing = []
    if not os.path.exists(gender_proto):
        models_missing.append("gender_deploy.prototxt")
    if not os.path.exists(gender_model):
        models_missing.append("gender_net.caffemodel")
    if not os.path.exists(age_proto):
        models_missing.append("age_deploy.prototxt")
    if not os.path.exists(age_model):
        models_missing.append("age_net.caffemodel")
    
    if models_missing:
        print(f"✗ Missing models: {', '.join(models_missing)}")
        print("\nOpenCV Caffe models need to be downloaded from:")
        print("  https://github.com/yu4u/age-gender-estimation")
        print("\nDownload these files to models_pretrained/:")
        print("  - age_deploy.prototxt")
        print("  - age_net.caffemodel")
        print("  - gender_deploy.prototxt")  
        print("  - gender_net.caffemodel")
        return
    
    print("\nLoading OpenCV models...")
    try:
        age_net = cv2.dnn.readNet(age_proto, age_model)
        print("✓ Age model loaded")
    except Exception as e:
        print(f"✗ Failed to load age model: {e}")
        return
    
    # Initialize face detector
    print("Initializing face detector...")
    image_processor = ImageProcesser()
    
    # Create output directory
    os.makedirs("webcam_predictions", exist_ok=True)
    
    print("\n" + "="*60)
    print("Auto-capturing 3 frames with 2-second intervals...")
    print("Look at the camera!")
    print("="*60)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("✗ Cannot open webcam")
        return
    
    results = []
    
    for capture_num in range(1, 4):
        print(f"\nCapture {capture_num}/3 in 2 seconds...")
        time.sleep(2)
        
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print(f"✗ Failed to capture frame {capture_num}")
            continue
        
        print("Captured! Processing...")
        
        # Detect face
        face, (x1, y1, x2, y2) = image_processor.detect_and_crop_face(frame)
        
        if face is None:
            print(f"✗ No face detected in capture {capture_num}")
            continue
        
        # Convert to BGR for OpenCV
        face_bgr = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
        
        # Prepare blob for neural network
        blob = cv2.dnn.blobFromImage(face_bgr, 1, (227, 227), 
                                      [78.4263377603, 87.7689143744, 114.895847746],
                                      swapRB=False, crop=False)
        
        # Predict age
        age_net.setInput(blob)
        age_preds = age_net.forward()
        
        # Get top 2 predictions
        age_probs = age_preds[0]
        top_2_idx = np.argsort(age_probs)[::-1][:2]
        
        age_class = AGE_GROUPS[top_2_idx[0]]
        confidence = float(age_probs[top_2_idx[0]]) * 100
        second_best = AGE_GROUPS[top_2_idx[1]]
        second_confidence = float(age_probs[top_2_idx[1]]) * 100
        
        result = {
            'capture': capture_num,
            'age_class': age_class,
            'confidence': confidence,
            'second_best': second_best,
            'second_confidence': second_confidence
        }
        results.append(result)
        
        # Display result
        print(f"\n✓ PREDICTION #{capture_num}:")
        print(f"  Age group: {age_class}")
        print(f"  Confidence: {confidence:.1f}%")
        print(f"  Alternative: {second_best} ({second_confidence:.1f}%)")
        
        if confidence < 30:
            print(f"  ⚠ Low confidence - manual review recommended")
        
        # Save result image
        result_img = face.copy()
        cv2.putText(result_img, f"Age: {age_class}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(result_img, f"Conf: {confidence:.1f}%", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        result_path = f"webcam_predictions/opencv_capture_{capture_num}.jpg"
        cv2.imwrite(result_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
        print(f"  Saved: {result_path}")
    
    cap.release()
    
    # Summary
    print("\n" + "="*60)
    print("✓ Test complete!")
    print("="*60)
    
    if results:
        print("\nSummary:")
        print(f"  Most likely age: {results[0]['age_class']}")
        
        # Check consistency
        ages = [r['age_class'] for r in results]
        if len(set(ages)) == 1:
            print(f"  ✓ All 3 captures predicted same age class!")
        else:
            print(f"  Predictions varied:")
            for r in results:
                print(f"    - Capture {r['capture']}: {r['age_class']} ({r['confidence']:.1f}%)")
    
    print(f"\nResults saved to: webcam_predictions/")


if __name__ == "__main__":
    test_opencv_age_detector()
