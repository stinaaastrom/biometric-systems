# Standard library imports
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import Input, Lambda



# Check available GPUs
print("Available GPUs:", tf.config.list_physical_devices('GPU'))

# TensorFlow/Keras imports
from keras.layers import (
    Input, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D, Rescaling,
    Conv2D, AveragePooling2D
)
from keras.models import Model, load_model, Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical

from constants import AGE_CLASSES
from evaluator import Evaluation
from data_generator import AGE_NORMALIZATION_FACTOR


# Default path for model files (absolute path)
import os
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models', 'age_model.keras')


class CNNModel:
    
    def __init__(self, train_generator=None, test_generator=None, X_test=None, age_test=None, 
                 gen_test=None, etn_test=None, X_train=None, age_train=None):
        """Initialize CNNModel with generators or arrays.
        
        Args:
            train_generator: Keras Sequence for training data
            test_generator: Keras Sequence for test data (optional)
            X_test: Numpy array of test images (for non-generator validation)
            age_test: Numpy array of test ages (for non-generator validation)
            gen_test: Numpy array of test genders (for evaluation)
            etn_test: Numpy array of test ethnicities (for evaluation)
            X_train: Numpy array of training images (deprecated, for backward compatibility)
            age_train: Numpy array of training ages (deprecated, for backward compatibility)
        """
        self.train_generator = train_generator
        self.test_generator = test_generator
        self.X_test = X_test
        self.age_test = age_test
        self.gen_test = gen_test
        self.etn_test = etn_test
        # Legacy support
        self.X_train = X_train
        self.age_train = age_train
        self.model = None
        self.history = None

    @staticmethod
    def find_age_class(predicted_age):
        """
        Returns the index of the class an age belongs to.
        Ages between classes are rounded to the nearest class.
        """
        if isinstance(predicted_age, np.ndarray):
            # Only scalar arrays are allowed; no one-hot handling
            if predicted_age.size == 1:
                predicted_age = float(predicted_age)
            else:
                raise ValueError(f"find_age_class: predicted_age must be a scalar value, got shape {predicted_age.shape}")
        else:
            predicted_age = float(predicted_age)

        # Handle ages below the first class
        if predicted_age < AGE_CLASSES[0][0]:
            return 0

        # Find the best matching class
        for idx, (min_age, max_age) in enumerate(AGE_CLASSES):
            if max_age is None:
                # Last class (75+)
                if predicted_age >= min_age:
                    return idx
            elif min_age <= predicted_age <= max_age:
                # Age falls directly in this class
                return idx
            elif idx < len(AGE_CLASSES) - 1:
                # Check if age is between this class and the next
                next_min = AGE_CLASSES[idx + 1][0]
                if max_age < predicted_age < next_min:
                    # Age is in the gap - assign to closer class
                    midpoint = (max_age + next_min) / 2
                    return idx if predicted_age < midpoint else idx + 1

        # Fallback: should not happen, but return last class if nothing matched
        return len(AGE_CLASSES) - 1




    def build_cnn_model(
            self,
            epochs_stage1=10,
            epochs_stage2=25,
            lr_stage1=1e-3,
            lr_stage2=1e-5,
            fine_tune_percent=0.3
        ):
        """
        Build and train EfficientNet age REGRESSION model in two stages:
        1. Feature extractor (freeze backbone)
        2. Fine-tuning (unfreeze last % of backbone)
        """

        inputs = Input(shape=(224, 224, 3))
        x = Lambda(preprocess_input)(inputs)

        base_model = EfficientNetB2(
            include_top=False,
            input_tensor=x,
            weights="imagenet"
        )

        x = base_model.output
        x = GlobalAveragePooling2D()(x)

        # Wider and deeper head for increased model capacity
        x = Dense(512, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        
        x = Dense(256, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        x = Dense(128, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        #  Regression output (sigmoid constrains output to [0, 1] range)
        output = Dense(1, activation="sigmoid")(x)

        model = Model(inputs=base_model.input, outputs=output)

        # -------------------
        # Stage 1 – freeze backbone
        # -------------------
        for layer in base_model.layers:
            layer.trainable = False

        model.compile(
            optimizer=Adam(learning_rate=lr_stage1),
            loss="mse",  # MSE works better with sigmoid for normalized regression
            metrics=["mae"]
        )

        model.summary()

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=2,
                min_lr=1e-6
            )
        ]

        print("\n--- Stage 1: Feature extractor (regression) ---")
        self.history = model.fit(
            self.train_generator,
            validation_data=self.test_generator,
            epochs=epochs_stage1,
            callbacks=callbacks
        )

        # -------------------
        # Stage 2 – fine-tuning
        # -------------------
        n_layers = len(base_model.layers)
        n_unfreeze = int(n_layers * fine_tune_percent)

        for layer in base_model.layers[-n_unfreeze:]:
            # Unfreeze layer but keep BatchNorm layers frozen to prevent distribution shift
            if not isinstance(layer, BatchNormalization):
                layer.trainable = True

        model.compile(
            optimizer=Adam(learning_rate=lr_stage2),
            loss="mse",  # MSE works better with sigmoid for normalized regression
            metrics=["mae"]
        )

        print(f"\n--- Stage 2: Fine-tuning last {fine_tune_percent*100:.0f}% ---")
        self.history_finetune = model.fit(
            self.train_generator,
            validation_data=self.test_generator,
            epochs=epochs_stage2,
            callbacks=callbacks
        )

        self.model = model



    def evaluate_model_performance(self, test_generator=None, gen_test=None, etn_test=None):
        """
        Evaluate model performance using the Evaluation class.
        
        Uses generator-based evaluation for memory efficiency.
        
        Provides:
        - Overall accuracy and per-class accuracy
        - Classification report
        - Confusion matrix
        - Demographic analysis
        
        Returns:
            (accuracy_percent, correct_count, incorrect_count)
        """
        # Use provided generator or fall back to instance variable
        if test_generator is None:
            test_generator = self.test_generator
        
        if test_generator is None:
            raise ValueError("No test_generator available for evaluation")
        
        # Create evaluator instance
        evaluator = Evaluation(AGE_CLASSES, self.find_age_class)
        
        # Let evaluator handle all prediction and evaluation logic
        accuracy, correct, incorrect = evaluator.evaluate(
            model=self.model,
            test_generator=test_generator,
            history=self.history if hasattr(self, 'history') else None,
            gen_test=gen_test,
            etn_test=etn_test
        )
        
        return accuracy, correct, incorrect

    
    def save_model(self, filepath=None):
        """Save the trained model and test data to files"""
        if self.model is None:
            print("No model to save. Train the model first.")
            return
        
        if filepath is None:
            filepath = DEFAULT_MODEL_PATH
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
        
        # Also save test data for later evaluation (if available)
        test_data_path = filepath.replace('.keras', '_testdata.npz')
        if self.X_test is not None and self.age_test is not None:
            try:
                np.savez_compressed(
                    test_data_path,
                    X_test=self.X_test,
                    age_test=self.age_test
                )
                print(f"Test data saved to {test_data_path}")
            except Exception as e:
                print(f"Warning: Could not save test data: {e}")
        else:
            print("Note: No test data to save (using generator-based validation)")
    
    def load_model(self, filepath=None):
        """Load a trained model and test data from files"""
        if filepath is None:
            filepath = DEFAULT_MODEL_PATH
        
        # Custom objects for Lambda layer with preprocess_input
        custom_objects = {
            'preprocess_input': preprocess_input
        }
        
        try:
            self.model = load_model(filepath, custom_objects=custom_objects)
            print(f"✓ Model loaded from {filepath}")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            print("  Trying alternative loading method...")
            try:
                # Fallback: try loading without custom objects
                self.model = load_model(filepath)
                print(f"✓ Model loaded (without custom objects)")
            except Exception as e2:
                print(f"✗ Failed to load model: {e2}")
                return None
        
        # Also load test data if it exists
        test_data_path = filepath.replace('.keras', '_testdata.npz')
        if os.path.exists(test_data_path):
            try:
                test_data = np.load(test_data_path, allow_pickle=True)
                self.X_test = test_data['X_test']
                self.age_test = test_data['age_test']
                print(f"✓ Test data loaded from {test_data_path} ({len(self.age_test)} samples)")
            except Exception as e:
                print(f"⚠️  Warning: Could not load test data: {e}")
        else:
            print(f"ℹ️  Test data file not found at {test_data_path}")
        
        return self.model