"""
Age Prediction Main Application

This module provides the main interface for training age prediction models
and making predictions on new images. Orchestrates dataset loading, model training,
evaluation, and prediction workflows.
"""

import os
import cv2
import numpy as np
from cnn_model import CNNModel
from dataset import IMAGE_SIZE, DatasetDownloader
from image_processing import ImageProcesser
from visualization import DatasetVisualizer

# Default path for saving/loading trained models
# Use fine-tuned model if available, otherwise fall back to original
FINETUNED_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'age_model_finetuned.keras')
ORIGINAL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'age_model.keras')
AGE_MODEL_PATH = ORIGINAL_PATH  # Use original - fine-tuning made it worse

# Confidence threshold for safe predictions (0-1 scale)
CONFIDENCE_THRESHOLD = 0.35



class AgePrediction:
    """
    Main application class for age prediction.
    """
    def __init__(self):
        """Initialize the age prediction application."""
        self.cnn_model = None

    def train_age_prediction_model(self):
        """Train a new CNN model on UTKFace and Facial Age datasets."""
        print("\n" + "="*60)
        print("STARTING AGE PREDICTION MODEL TRAINING")
        print("="*60)
        
        downloader = DatasetDownloader()
        X_train, X_test, age_train, age_test, gen_train, gen_test, etn_train, etn_test = downloader.generate_dataset()
        
        # Skip blocking visualization - proceed directly to training
        # visualizer = DatasetVisualizer(age_train, gen_train, etn_train)
        # visualizer.show_initial_visuals()
        
        self.cnn_model = CNNModel(X_train, age_train, X_test, age_test)
        self.cnn_model.build_cnn_model()
        
        # Store demographic data for later evaluation
        self.gen_test = gen_test
        self.etn_test = etn_test
        
        print("\n✓ Training complete!")

    def predict_age(self, img_path):
        """Predict and display age from an image file."""
        if self.cnn_model is None or self.cnn_model.model is None:
            raise ValueError("Model not trained or loaded. Train a model first or load an existing one.")
        try:
            # Load image
            frame = cv2.imread(img_path)
            if frame is None:
                print(f"Error: Could not load image from {img_path}")
                return
            
            # Process image
            image_processor = ImageProcesser()
            face_img = image_processor.detect_crop_faces(frame)
            if face_img is None:
                print("Error: Could not detect face in image")
                return
                
            face_img = image_processor.image_enhancements(face_img)
            face_img = cv2.resize(face_img, IMAGE_SIZE)
            
            # Normalize and build a TTA batch: original, flip, light center-crops
            face_img_normalized = face_img / 255.0

            def center_crop(img, scale=0.95):
                h, w, _ = img.shape
                ch, cw = int(h * scale), int(w * scale)
                y0 = (h - ch) // 2
                x0 = (w - cw) // 2
                crop = img[y0:y0+ch, x0:x0+cw]
                return cv2.resize(crop, (IMAGE_SIZE[1], IMAGE_SIZE[0]))

            aug_views = []
            aug_views.append(face_img_normalized)
            aug_views.append(np.flip(face_img_normalized, axis=1))
            cc = center_crop(face_img_normalized, scale=0.95)
            aug_views.append(cc)
            aug_views.append(np.flip(cc, axis=1))

            face_batch = np.stack(aug_views, axis=0).reshape(len(aug_views), IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
            
            # Predict: [class probabilities, regression age]
            pred_probs, pred_reg = self.cnn_model.model.predict(face_batch, verbose=0)
            
            # Ensure pred_probs is 2D (batch, classes)
            if len(pred_probs.shape) == 1:
                pred_probs = pred_probs.reshape(1, -1)
            
            # Get mean predictions across TTA
            mean_probs = pred_probs.mean(axis=0)
            mean_reg_age = pred_reg.mean()
            
            # Apply boundary nudging: use regression to adjust softmax near boundaries
            mean_probs = self.cnn_model.nudge_probabilities_by_regression(mean_probs, mean_reg_age, boundary_threshold=1.5)
            
            # Validate shape
            from cnn_model import AGE_CLASSES
            if len(mean_probs) != len(AGE_CLASSES):
                print(f"Error: Probability shape mismatch. Expected {len(AGE_CLASSES)} classes, got {len(mean_probs)}")
                return
            
            if len(mean_probs) < 2:
                print(f"Error: Invalid probability shape: {mean_probs.shape}")
                return
                
            predicted_class = np.argmax(mean_probs)
            
            # Clamp predicted_class to valid range
            predicted_class = min(predicted_class, len(AGE_CLASSES) - 1)
            
            confidence = mean_probs[predicted_class] * 100
            
            # HYBRID APPROACH: If confidence is low, use regression-binning instead
            # This handles domain gap when classification fails but regression works
            reg_class = self.cnn_model.map_age_to_class(mean_reg_age)
            use_regression = confidence < (CONFIDENCE_THRESHOLD * 100)  # Use regression if conf < 35%
            
            if use_regression:
                print(f"  [Switching to regression-based prediction due to low confidence]")
                predicted_class = reg_class
                # Recompute confidence based on how far regression is from class boundaries
                min_age, max_age = AGE_CLASSES[predicted_class]
                if max_age is None:
                    # Open-ended class, higher confidence
                    confidence = 50.0
                else:
                    # Confidence inversely proportional to distance from class center
                    class_center = (min_age + max_age) / 2
                    distance = abs(mean_reg_age - class_center)
                    max_distance = (max_age - min_age) / 2
                    confidence = max(30.0, 60.0 * (1 - distance / max_distance))

            # Optional: low-confidence reject hint
            # Show top-2 classes to aid manual review when confidence is low
            if len(mean_probs) >= 2:
                top2_idx = np.argsort(mean_probs)[-2:][::-1]
                top2_conf = mean_probs[top2_idx] * 100
            else:
                top2_idx = np.array([predicted_class, 0])
                top2_conf = np.array([confidence / 100.0, 0])
            
            # Convert class to age range
            from cnn_model import AGE_CLASSES
            min_age, max_age = AGE_CLASSES[predicted_class]
            if max_age is None:
                age_range = f"{min_age}+"
                predicted_age_value = min_age + 5  # Use middle of range for display
            else:
                age_range = f"{min_age}-{max_age}"
                predicted_age_value = (min_age + max_age) / 2
            
            print(f"\nPredicted age class: {age_range} (confidence: {confidence:.1f}%)")
            
            # Also report regression estimate
            est_age = float(mean_reg_age)
            print(f"Estimated age: {predicted_age_value:.0f} years (reg: {est_age:.1f})")
            
            # Enhanced reject policy: flag if confidence low OR regression near boundary
            reject_reasons = []
            
            if (confidence / 100.0) < CONFIDENCE_THRESHOLD:
                reject_reasons.append("low confidence")
            
            # Check if regression is near class boundary (± 1 year)
            if max_age is not None and abs(est_age - max_age) < 1.0:
                reject_reasons.append(f"regression near upper boundary ({est_age:.1f} ≈ {max_age})")
            if min_age > 0 and abs(est_age - min_age) < 1.0:
                reject_reasons.append(f"regression near lower boundary ({est_age:.1f} ≈ {min_age})")
            
            if reject_reasons:
                print(f"⚠ Review recommended: {', '.join(reject_reasons)}")
                if len(AGE_CLASSES) > top2_idx[1]:
                    alt_min, alt_max = AGE_CLASSES[top2_idx[1]]
                    alt_range = f"{alt_min}+" if alt_max is None else f"{alt_min}-{alt_max}"
                    print(f"  Alternative: {alt_range} ({top2_conf[1]*100:.1f}%)")
            
            # Display result
            DatasetVisualizer.display_prediction(face_img, predicted_age_value)
        
        except Exception as e:
            print(f"✗ Prediction error: {e}")
            import traceback
            traceback.print_exc()
    
    def save_model(self, filepath=None):
        """Save the trained model to disk."""
        if self.cnn_model is None:
            raise ValueError("No model to save. Train a model first.")
        
        # Determine which path to use
        if not filepath:
            filepath = AGE_MODEL_PATH
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # If filepath is None, let CNNModel use its default
        if filepath != AGE_MODEL_PATH:
            self.cnn_model.save_model(filepath)
        else:
            self.cnn_model.save_model()
    
    def load_test_dataset(self):
        """Load the test dataset for evaluation purposes."""
        print("\nLoading test dataset for evaluation...")
        downloader = DatasetDownloader()
        X_train, X_test, age_train, age_test, gen_train, gen_test, etn_train, etn_test = downloader.generate_dataset()
        
        # Update CNNModel's test data
        if self.cnn_model:
            self.cnn_model.X_test = X_test
            self.cnn_model.age_test = age_test
        
        # Store demographic data for evaluation
        self.gen_test = gen_test
        self.etn_test = etn_test
        print(f"✓ Test dataset loaded: {len(X_test)} samples")
    
    def load_model(self, filepath=None):
        """Load a previously trained model from disk."""
        import numpy as np
        
        # Determine which path to use
        if not filepath:
            filepath = AGE_MODEL_PATH
        
        # Check if file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Model file not found: {filepath}\n"
                f"Please train and save a model first (options 1 then 5), "
                f"or provide the correct path to an existing model."
            )
        
        # Create placeholder CNNModel (won't be used for training)
        self.cnn_model = CNNModel(
            np.zeros((1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),  # Placeholder
            np.zeros(1),
            np.zeros((1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
            np.zeros(1)
        )
        self.cnn_model.load_model(filepath)
        print("✓ Model ready for predictions")
    
    def program_interface(self):
        """Run the interactive command-line menu for the age prediction system."""
        print("\n" + "="*60)
        print("AGE PREDICTION SYSTEM")
        print("="*60)
        
        while True:
            print("\n" + "-"*60)
            print("MENU OPTIONS:")
            print("-"*60)
            print("1. Train new age prediction model")
            print("2. Load existing model from file")
            print("3. Make prediction on an image")
            print("4. Evaluate model on test data")
            print("5. Save current model")
            print("6. Calibrate confidence (temperature scaling)")
            print("7. Quit")
            print("-"*60)
            
            action = input("\nSelect option (1-7): ").strip()

            try:
                if action == '1':
                    self.train_age_prediction_model()
                    
                elif action == '2':
                    filepath = input(f"\nEnter model path (press Enter for default: {AGE_MODEL_PATH}): ").strip()
                    self.load_model(filepath if filepath else None)
                    
                elif action == '3':
                    if self.cnn_model is None or self.cnn_model.model is None:
                        print("\n⚠ No model loaded. Please train or load a model first.")
                        continue
                    img_path = input("\nEnter image path: ").strip()
                    if os.path.exists(img_path):
                        self.predict_age(img_path)
                    else:
                        print(f"\n✗ Error: File not found: {img_path}")
                        
                elif action == '4':
                    if self.cnn_model is None or self.cnn_model.model is None:
                        print("\n⚠ No model loaded. Please train a model first (option 1).")
                        continue
                    
                    # Check if we have real test data (not just placeholders)
                    if len(self.cnn_model.X_test) == 1 and self.cnn_model.X_test[0].sum() == 0:
                        print("\n No test data loaded. Loading test dataset...")
                        self.load_test_dataset()
                    
                    # Pass demographic data if available
                    gen_test = getattr(self, 'gen_test', None)
                    etn_test = getattr(self, 'etn_test', None)
                    self.cnn_model.evaluate_model_performance(gen_test, etn_test)
                    
                elif action == '5':
                    if self.cnn_model is None:
                        print("\n⚠ No model to save. Train a model first.")
                        continue
                    filepath = input(f"\nEnter save path (press Enter for default: {AGE_MODEL_PATH}): ").strip()
                    self.save_model(filepath if filepath else None)
                    
                elif action == '6':
                    if self.cnn_model is None or self.cnn_model.model is None:
                        print("\n⚠ No model loaded. Please train or load a model first.")
                        continue
                    
                    # Check if we have test data
                    if len(self.cnn_model.X_test) == 1 and self.cnn_model.X_test[0].sum() == 0:
                        print("\n⚠ No test data loaded. Loading test dataset...")
                        self.load_test_dataset()
                    
                    # Run temperature calibration
                    self.cnn_model.calibrate_temperature()
                    print("\n✓ Temperature calibration complete. Model will use calibrated probabilities.")
                    print("  Don't forget to save the model (option 5) to preserve calibration!")
                    
                elif action == '7' or action.lower() == 'quit':
                    print("\nGoodbye!")
                    break
                    
                else:
                    print("\n✗ Invalid option. Please select 1-7.")
                    
            except Exception as e:
                print(f"\n✗ Error: {e}")
