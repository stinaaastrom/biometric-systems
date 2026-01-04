"""
Age Prediction Main Application

This module provides the main interface for training age prediction models
and making predictions on new images. Orchestrates dataset loading, model training,
evaluation, and prediction workflows.
"""

import os
import cv2
import numpy as np
from cnn_model import CNNModel, AGE_CLASSES
from dataset_downloader import DatasetDownloader
from dataset_processor import DatasetProcessor
from image_processing import ImageProcesser
from visualization import DatasetVisualizer
from constants import IMAGE_SIZE
from data_generator import AGE_NORMALIZATION_FACTOR

# Default path for saving/loading trained models
AGE_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models', 'age_model.keras')



class AgePrediction:
    """
    Main application class for age prediction.
    """
    def __init__(self):
        """Initialize the age prediction application."""
        self.cnn_model = None

    def train_age_prediction_model(self):
        """Train a new CNN model on UTKFace and Facial Age datasets using generators."""
        print("\n" + "="*60)
        print("STARTING AGE PREDICTION MODEL TRAINING")
        print("="*60)

        # Ensure datasets exist, then preprocess and build generators
        downloader = DatasetDownloader()  # download + folder management
        processor = DatasetProcessor(downloader.dataset_utkface, downloader.dataset_facial_age)

        # 1. Preprocess and save all images (without augmentation)
        processor.preprocess_and_save(output_dir='data/preprocessed_faces')

        # 2. Build generators (augmentation happens in the generator)
        train_gen, test_gen, train_paths, train_ages, train_genders, train_etnicity, test_paths, test_ages, test_genders, test_etnicity = processor.generate_dataset(batch_size=32)

        # Visualize initial dataset distribution
        visualizer = DatasetVisualizer(train_ages, train_genders, train_etnicity)
        visualizer.show_initial_visuals()

        # Create and train the model with generators
        self.cnn_model = CNNModel(
            train_generator=train_gen,
            test_generator=test_gen,
            gen_test=test_genders,
            etn_test=test_etnicity
        )

        # Train model - use 20 epochs for faster training
        self.cnn_model.build_cnn_model(epochs_stage1=10, epochs_stage2=20)

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
        #face_img = image_processor.detect_crop_faces(frame)
        face_img = frame
        if face_img is None:
            print("Error: Could not detect face in image")
            return
            
        face_img = image_processor.image_enhancements(face_img)
        face_img = cv2.resize(face_img, IMAGE_SIZE)  # Use 224x224 (IMAGE_SIZE)
        
        # Normalize and reshape for model input
        face_img_normalized = face_img.astype(np.float32) / 255.0
        face_img_normalized = face_img_normalized.reshape(1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
        
        # Predict age as a scalar (regression head outputs shape (1, 1))
        # Model outputs normalized age (0-1 range), so denormalize it
        predicted_age_normalized = float(self.cnn_model.model.predict(face_img_normalized, verbose=0)[0][0])
        predicted_age_value = predicted_age_normalized * AGE_NORMALIZATION_FACTOR
        predicted_class = self.cnn_model.find_age_class(predicted_age_value)
        min_age, max_age = AGE_CLASSES[predicted_class] if predicted_class is not None else (None, None)
        age_range = f"{min_age}-{max_age}" if max_age else f"{min_age}+" if min_age is not None else "unknown"

        print(f"\nEstimated age: {predicted_age_value:.1f} years")
        if predicted_class is not None:
            print(f"Age range bucket: {age_range}")
        
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
    
    def load_test_dataset(self):
        """Load the test dataset for evaluation purposes."""
        print("\nLoading test dataset for evaluation...")
        downloader = DatasetDownloader()
        processor = DatasetProcessor(downloader.dataset_utkface, downloader.dataset_facial_age)
        train_gen, test_gen, train_paths, train_ages, train_genders, train_etnicity, test_paths, test_ages, test_genders, test_etnicity = processor.generate_dataset()
        
        # Update CNNModel's test data
        if self.cnn_model:
            self.cnn_model.age_test = test_ages
            self.cnn_model.gen_test = test_genders
            self.cnn_model.etn_test = test_etnicity
            self.cnn_model.test_generator = test_gen
        
        print(f"✓ Test dataset loaded: {len(test_paths)} samples")
    
    def load_model(self, filepath=None):
        """Load a previously trained model from disk."""
        
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
        
        # Create placeholder CNNModel for loading
        self.cnn_model = CNNModel()
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
                    
                    # Check if we have test data loaded
                    if self.cnn_model.X_test is None and self.cnn_model.test_generator is None:
                        print("\nNo test data loaded. Loading test dataset...")
                        self.load_test_dataset()
                    
                    # Evaluate the model
                    self.cnn_model.evaluate_model_performance()
                    
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
