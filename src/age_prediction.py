"""
Age Prediction Main Application

This module provides the main interface for training age prediction models
and making predictions on new images. Orchestrates dataset loading, model training,
evaluation, and prediction workflows.
"""

import os
import cv2
from cnn_model import CNNModel
from dataset import IMAGE_SIZE, DatasetDownloader
from image_processing import ImageProcesser
from visualization import DatasetVisualizer

# Default path for saving/loading trained models
AGE_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'age_model.keras')



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
        
        visualizer = DatasetVisualizer(age_train, gen_train, etn_train)
        visualizer.show_initial_visuals()
        
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
        
        # Normalize and reshape for model input
        face_img_normalized = face_img / 255.0
        face_img_normalized = face_img_normalized.reshape(1, 64, 64, 3)
        
        # Predict age
        predicted_age_value = self.cnn_model.model.predict(face_img_normalized, verbose=0)[0][0]
        
        print(f"\nPredicted age: {predicted_age_value:.1f} years")
        
        # Display result
        DatasetVisualizer.display_prediction(face_img, predicted_age_value)
    
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
            print("6. Quit")
            print("-"*60)
            
            action = input("\nSelect option (1-6): ").strip()

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
                    
                elif action == '6' or action.lower() == 'quit':
                    print("\nGoodbye!")
                    break
                    
                else:
                    print("\n✗ Invalid option. Please select 1-6.")
                    
            except Exception as e:
                print(f"\n✗ Error: {e}")
