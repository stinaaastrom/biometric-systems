"""
Test FairFace pre-trained model for age estimation on webcam
FairFace is optimized for diverse faces and selfies
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from urllib.request import urlopen
import time
from datetime import datetime

# Add src to path
import sys
sys.path.insert(0, 'src')

from image_processing import ImageProcesser

# FairFace model URLs
FAIRFACE_AGE_MODEL_URL = "https://github.com/dariusaffect/fairface/raw/master/fair_face_models/fairface_age.h5"
FAIRFACE_AGE_WEIGHTS_URL = "https://github.com/dariusaffect/fairface/raw/master/fair_face_models/fairface_age_weights.h5"

AGE_CLASSES = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']

def download_fairface_model():
    """Download FairFace age model"""
    model_path = "models/fairface_age.h5"
    
    if os.path.exists(model_path):
        print(f"FairFace model already downloaded: {model_path}")
        return model_path
    
    print("Downloading FairFace age model...")
    try:
        # Try downloading from direct source
        response = urlopen(FAIRFACE_AGE_MODEL_URL)
        with open(model_path, 'wb') as f:
            f.write(response.read())
        print(f"✓ Downloaded to {model_path}")
        return model_path
    except Exception as e:
        print(f"✗ Failed to download: {e}")
        print("Note: You can manually download from:")
        print("  https://github.com/dariusaffect/fairface/tree/master/fair_face_models")
        return None


def load_fairface_model(model_path):
    """Load FairFace model"""
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return None
    
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"✓ FairFace model loaded")
        return model
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return None


def preprocess_face(face_img):
    """Preprocess face for FairFace (224x224)"""
    if face_img is None:
        return None
    
    # Resize to 224x224
    face = cv2.resize(face_img, (224, 224))
    
    # Convert BGR to RGB
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    face = face.astype(np.float32) / 255.0
    
    return face


def predict_age_fairface(model, face_img):
    """Predict age using FairFace model"""
    if model is None or face_img is None:
        return None, None, None
    
    try:
        # Preprocess
        face = preprocess_face(face_img)
        if face is None:
            return None, None, None
        
        # Add batch dimension
        face_batch = np.expand_dims(face, axis=0)
        
        # Predict
        predictions = model.predict(face_batch, verbose=0)
        
        # Get age class probabilities
        age_probs = predictions[0]
        age_class_idx = np.argmax(age_probs)
        age_class = AGE_CLASSES[age_class_idx]
        confidence = float(age_probs[age_class_idx]) * 100
        
        return age_class, confidence, age_probs
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, None, None


def test_fairface_webcam():
    """Test FairFace on webcam captures"""
    
    print("="*60)
    print("FAIRFACE AGE PREDICTION TEST")
    print("="*60)
    
    # Download model
    model_path = download_fairface_model()
    if model_path is None:
        print("\n✗ Could not obtain FairFace model")
        print("Please download manually from:")
        print("  https://github.com/dariusaffect/fairface/tree/master/fair_face_models")
        return
    
    # Load model
    print("\nLoading FairFace model...")
    model = load_fairface_model(model_path)
    if model is None:
        print("✗ Failed to load model")
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
        
        # Predict age
        age_class, confidence, all_probs = predict_age_fairface(model, face)
        
        if age_class is None:
            print(f"✗ Prediction failed for capture {capture_num}")
            continue
        
        # Get second-best prediction
        probs_sorted_idx = np.argsort(all_probs)[::-1]
        second_best = AGE_CLASSES[probs_sorted_idx[1]]
        second_confidence = float(all_probs[probs_sorted_idx[1]]) * 100
        
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
        print(f"  Age class: {age_class}")
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
        
        result_path = f"webcam_predictions/fairface_capture_{capture_num}.jpg"
        cv2.imwrite(result_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
        print(f"  Saved: {result_path}")
    
    cap.release()
    
    # Summary
    print("\n" + "="*60)
    print("✓ Test complete!")
    print("="*60)
    
    if results:
        print("\nSummary:")
        print(f"  Consistent prediction: {results[0]['age_class']}")
        
        # Check consistency
        ages = [r['age_class'] for r in results]
        if len(set(ages)) == 1:
            print(f"  ✓ All 3 captures predicted same age class!")
        else:
            print(f"  ⚠ Predictions varied:")
            for r in results:
                print(f"    - Capture {r['capture']}: {r['age_class']} ({r['confidence']:.1f}%)")
    
    print(f"\nResults saved to: webcam_predictions/")


if __name__ == "__main__":
    test_fairface_webcam()
