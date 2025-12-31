# Standard library imports
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
from tensorflow.keras.applications import EfficientNetB0
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
        Returns the index of the class an age belongs to,
        or None if age is outside all defined ranges.
        Handles both scalar ålder och one-hot-vektor (t.ex. [0 0 0 0 0 1 0 0]).
        """
        if isinstance(predicted_age, np.ndarray):
            # Om det är en one-hot-vektor (längd 8)
            if predicted_age.ndim == 1 and predicted_age.shape[0] == len(AGE_CLASSES):
                return int(np.argmax(predicted_age))
            # Om det är en array med ett element
            if predicted_age.size == 1:
                predicted_age = float(predicted_age)
            else:
                raise ValueError(f"find_age_class: predicted_age måste vara en scalar eller one-hot-vektor, fick shape {predicted_age.shape}")

        for idx, (min_age, max_age) in enumerate(AGE_CLASSES):
            if max_age is None:
                if predicted_age >= min_age:
                    return idx
            elif min_age <= predicted_age <= max_age:
                return idx
        return None




    def build_cnn_model(self, epochs_stage1=10, epochs_stage2=25, lr_stage1=1e-3, lr_stage2=1e-5, fine_tune_percent=0.3):
        """
        Bygger och tränar EfficientNet-modellen i två steg:
        1. Feature extractor (frys backbone, träna topplager)
        2. Fine-tuning (tina sista 20-30% av backbone)
        """
    
        

        inputs = Input(shape=(224, 224, 3))
        x = Lambda(preprocess_input)(inputs)
        base_model = EfficientNetB0(include_top=False, input_tensor=x, weights="imagenet")
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.3)(x)
        x = Dense(128, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)

        output = Dense(8, activation="softmax")(x)
        model = Model(inputs=base_model.input, outputs=output)

        # Steg 1: Frys hela backbone
        for layer in base_model.layers:
            layer.trainable = False

        model.compile(optimizer=Adam(learning_rate=lr_stage1), loss="categorical_crossentropy", metrics=["accuracy"])
        model.summary()

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)
        ]

        if self.train_generator is not None and self.test_generator is not None:
            print("\n--- Steg 1: Feature extractor (frys backbone, träna topplager) ---")
            self.history = model.fit(
                self.train_generator,
                validation_data=self.test_generator,
                epochs=epochs_stage1,
                callbacks=callbacks
            )
        else:
            raise ValueError("Ingen träningsdata tillgänglig.")

        # Steg 2: Fine-tuning (tina sista 20-30% av backbone)
        n_layers = len(base_model.layers)
        n_unfreeze = int(n_layers * fine_tune_percent)
        for layer in base_model.layers[-n_unfreeze:]:
            layer.trainable = True

        model.compile(optimizer=Adam(learning_rate=lr_stage2), loss="categorical_crossentropy", metrics=["accuracy"])

        print(f"\n--- Steg 2: Fine-tuning (tina sista {fine_tune_percent*100:.0f}% av backbone) ---")
        self.history_finetune = model.fit(
            self.train_generator,
            validation_data=self.test_generator,
            epochs=epochs_stage2,
            callbacks=callbacks
        )

        self.model = model



    from keras.saving import register_keras_serializable

    @staticmethod
    @register_keras_serializable()
    def age_mae(y_true, y_pred):
        class_centers = tf.constant([1, 4, 8, 13, 18, 30, 50, 70], dtype=tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_true_age = tf.reduce_sum(y_true * class_centers, axis=1)
        y_pred_age = tf.reduce_sum(y_pred * class_centers, axis=1)
        return tf.reduce_mean(tf.abs(y_true_age - y_pred_age))


            


    def evaluate_model_performance(self, gen_test=None, etn_test=None):
        """
        Evaluate model performance by comparing age class predictions.
        Supports both array-based test data and Keras generators.
        """

        # 1. PREDICTIONS + TRUE LABELS
        if self.X_test is not None and isinstance(self.X_test, np.ndarray) and self.X_test.ndim >= 3:
            print("Making predictions on array-based test data...")
            predictions = self.model.predict(self.X_test)
            predicted_classes = np.argmax(predictions, axis=1)
            actual_classes = np.argmax(self.age_test, axis=1)
            valid_mask = actual_classes != None
            predicted_classes = predicted_classes[valid_mask]
            actual_classes = actual_classes[valid_mask].astype(int)
        elif self.test_generator is not None:
            print("Making predictions on generator-based test data...")
            predictions = self.model.predict(self.test_generator)
            # test_generator.labels är one-hot
            actual_classes = np.argmax(self.test_generator.labels, axis=1)
            predicted_classes = np.argmax(predictions, axis=1)
            valid_mask = actual_classes != None
            predicted_classes = predicted_classes[valid_mask]
            actual_classes = actual_classes[valid_mask].astype(int)
        else:
            raise ValueError("No test data available (X_test or test_generator required).")

        # Säkerhetskontroll
        if len(predicted_classes) != len(actual_classes):
            raise ValueError(
                f"Length mismatch: predicted={len(predicted_classes)}, actual={len(actual_classes)}"
            )

        # --------------------------------------------------
        # 2. DEMOGRAFISK FILTRERING (OM FINNS)
        # --------------------------------------------------
        if gen_test is not None:
            gen_test = np.asarray(gen_test)[:len(actual_classes)]
        if etn_test is not None:
            etn_test = np.asarray(etn_test)[:len(actual_classes)]

        # --------------------------------------------------
        # 3. ACCURACY
        # --------------------------------------------------
        correct_predictions = np.sum(predicted_classes == actual_classes)
        total_predictions = len(actual_classes)
        accuracy = (correct_predictions / total_predictions) * 100

        # --------------------------------------------------
        # 4. UTSKRIFT
        # --------------------------------------------------
        print("\n" + "=" * 55)
        print("MODEL EVALUATION RESULTS – AGE CLASS ACCURACY")
        print("=" * 55)
        print(f"Total predictions: {total_predictions}")
        print(f"Correct predictions: {correct_predictions}")
        print(f"Incorrect predictions: {total_predictions - correct_predictions}")
        print(f"Accuracy: {accuracy:.2f}%\n")

        print("Age Classes:")
        for idx, (min_age, max_age) in enumerate(AGE_CLASSES):
            label = f"{min_age}+" if max_age is None else f"{min_age}-{max_age}"
            print(f"  Class {idx}: {label}")

        # --------------------------------------------------
        # 5. PER-KLASS ACCURACY
        # --------------------------------------------------
        print("\nPer-class accuracy:")
        for class_idx in range(len(AGE_CLASSES)):
            mask = actual_classes == class_idx
            if np.any(mask):
                class_correct = np.sum(predicted_classes[mask] == class_idx)
                class_total = np.sum(mask)
                class_acc = (class_correct / class_total) * 100

                min_age, max_age = AGE_CLASSES[class_idx]
                label = f"{min_age}+" if max_age is None else f"{min_age}-{max_age}"
                print(
                    f"  Class {class_idx} ({label}): "
                    f"{class_acc:.2f}% ({class_correct}/{class_total})"
                )

        print("=" * 55 + "\n")

        # --------------------------------------------------
        # 6. TRÄNINGSHISTORIK
        # --------------------------------------------------
        if hasattr(self, "history") and self.history is not None:
            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            plt.plot(self.history.history["accuracy"], label="Train Accuracy")
            plt.plot(self.history.history["val_accuracy"], label="Val Accuracy")
            plt.title("Model Accuracy")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.grid(True)

            plt.subplot(1, 2, 2)
            plt.plot(self.history.history["loss"], label="Train Loss")
            plt.plot(self.history.history["val_loss"], label="Val Loss")
            plt.title("Model Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.show()

        return accuracy, correct_predictions, total_predictions - correct_predictions

    
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
        
        self.model = load_model(filepath)
        print(f"Model loaded from {filepath}")
        
        # Also load test data if it exists
        test_data_path = filepath.replace('.keras', '_testdata.npz')
        if os.path.exists(test_data_path):
            try:
                test_data = np.load(test_data_path, allow_pickle=True)
                self.X_test = test_data['X_test']
                self.age_test = test_data['age_test']
                print(f"Test data loaded from {test_data_path} ({len(self.age_test)} samples)")
            except Exception as e:
                print(f"Warning: Could not load test data: {e}")
        else:
            print(f"Warning: Test data file not found at {test_data_path}")
        
        return self.model