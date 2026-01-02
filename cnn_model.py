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
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
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

# Default path for model files (absolute path)
import os
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models', 'age_model.keras')


class EveryNEpochCheckpoint(Callback):
    """Saves model weights every Nth epoch using a formatted filepath."""

    def __init__(self, filepath, every_n_epochs=1):
        super().__init__()
        self.filepath = filepath
        self.every_n_epochs = max(1, every_n_epochs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # epoch is zero-based; add 1 for human-friendly numbering
        if (epoch + 1) % self.every_n_epochs != 0:
            return

        filepath = self.filepath.format(
            epoch=epoch + 1,
            val_loss=logs.get("val_loss", 0.0)
        )
        self.model.save_weights(filepath)
        print(f"\n✓ Saved checkpoint: {filepath}")

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




    def build_cnn_model(self, epochs_stage1=10, epochs_stage2=25, lr_stage1=1e-3, lr_stage2=1e-5, fine_tune_percent=0.3, checkpoint_dir='models/checkpoints'):
        """
        Bygger och tränar EfficientNet-modellen i två steg:
        1. Feature extractor (frys backbone, träna topplager)
        2. Fine-tuning (tina sista 20-30% av backbone)
        
        Stödjer automatisk TPU-detektering och distributed training.
        """
    
        # Skapa checkpoint-mapp om den inte finns
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Kolla om det finns en tidigare checkpoint att ladda (stage 1)
        latest_checkpoint = self._get_latest_checkpoint(checkpoint_dir, stage='stage1')

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
        
        # Ladda checkpoint om den finns
        initial_epoch_stage1 = 0
        if latest_checkpoint and 'stage1' in latest_checkpoint:
            print(f"\n✓ Laddar checkpoint: {latest_checkpoint}")
            model.load_weights(latest_checkpoint)
            # Extrahera epoch-nummer från filnamnet
            epoch_str = latest_checkpoint.split('epoch')[-1].split('-')[0]
            initial_epoch_stage1 = int(epoch_str)
            print(f"  Fortsätter från epoch {initial_epoch_stage1}")
        
        # Kompilera EFTER checkpoint laddats för att säkerställa fryst status
        model.compile(optimizer=Adam(learning_rate=lr_stage1), loss="categorical_crossentropy", metrics=["accuracy"])
        
        model.summary()

        # Checkpoint callback för Stage 1 - sparar var 10:e epoch
        checkpoint_stage1 = EveryNEpochCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'stage1_epoch{epoch:02d}-val_loss{val_loss:.4f}.weights.h5'),
            every_n_epochs=10
        )
        
        callbacks = [
            checkpoint_stage1,
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)
        ]

        if self.train_generator is not None and self.test_generator is not None:
            print("\n--- Steg 1: Feature extractor (frys backbone, träna topplager) ---")
            # Beräkna total epochs: om vi laddat epoch 10, kör till epoch 10+10=20
            target_epochs_stage1 = initial_epoch_stage1 + epochs_stage1
            print(f"Kör från epoch {initial_epoch_stage1} till {target_epochs_stage1}")
            self.history = model.fit(
                self.train_generator,
                validation_data=self.test_generator,
                epochs=target_epochs_stage1,
                initial_epoch=initial_epoch_stage1,
                callbacks=callbacks
            )
        else:
            raise ValueError("Ingen träningsdata tillgänglig.")

        # Steg 2: Fine-tuning (tina sista 20-30% av backbone)
        n_layers = len(base_model.layers)
        n_unfreeze = int(n_layers * fine_tune_percent)
        for layer in base_model.layers[-n_unfreeze:]:
            layer.trainable = True
        
        # Kolla om det finns checkpoint för stage 2
        latest_checkpoint_stage2 = self._get_latest_checkpoint(checkpoint_dir, stage='stage2')
        initial_epoch_stage2 = 0
        if latest_checkpoint_stage2:
            print(f"\n✓ Laddar Stage 2 checkpoint: {latest_checkpoint_stage2}")
            model.load_weights(latest_checkpoint_stage2)
            epoch_str = latest_checkpoint_stage2.split('epoch')[-1].split('-')[0]
            initial_epoch_stage2 = int(epoch_str)
            print(f"  Fortsätter från epoch {initial_epoch_stage2}")
        
        # Kompilera EFTER checkpoint laddats
        model.compile(optimizer=Adam(learning_rate=lr_stage2), loss="categorical_crossentropy", metrics=["accuracy"])
        
        # Checkpoint callback för Stage 2
        checkpoint_stage2 = EveryNEpochCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'stage2_epoch{epoch:02d}-val_loss{val_loss:.4f}.weights.h5'),
            every_n_epochs=10
        )
        
        callbacks_stage2 = [
            checkpoint_stage2,
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)
        ]

        print(f"\n--- Steg 2: Fine-tuning (tina sista {fine_tune_percent*100:.0f}% av backbone) ---")
        target_epochs_stage2 = initial_epoch_stage2 + epochs_stage2
        print(f"Kör från epoch {initial_epoch_stage2} till {target_epochs_stage2}")
        self.history_finetune = model.fit(
            self.train_generator,
            validation_data=self.test_generator,
            epochs=target_epochs_stage2,
            initial_epoch=initial_epoch_stage2,
            callbacks=callbacks_stage2
        )

        self.model = model



    def _get_latest_checkpoint(self, checkpoint_dir, stage=None):
        """Hitta senaste checkpoint-fil i checkpoint-mappen."""
        if not os.path.exists(checkpoint_dir):
            return None

        checkpoint_exts = ('.weights.h5', '.ckpt')
        checkpoints = []
        for f in os.listdir(checkpoint_dir):
            if not f.endswith(checkpoint_exts):
                continue
            if stage and stage not in f:
                continue
            checkpoints.append(os.path.join(checkpoint_dir, f))

        if not checkpoints:
            return None

        checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return checkpoints[0]

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


            


    def evaluate_model_performance(self, test_generator=None, X_test=None, age_test=None, gen_test=None, etn_test=None, top_n=10, output_dir=None):
        """
        Evaluate model performance using the Evaluation class.
        
        Supports both generator-based and array-based test data.
        
        Provides:
        - Overall accuracy and per-class accuracy
        - Classification report
        - Confusion matrix
        - Demographic analysis
        - Saves top-N worst predictions as images
        
        Args:
            test_generator: Keras Sequence generator for test data (optional, uses self.test_generator)
            X_test: Test images array (optional, uses self.X_test if not using generator)
            age_test: Test ages array (optional, uses self.age_test if not using generator)
            gen_test: Gender labels (optional)
            etn_test: Ethnicity labels (optional)
            top_n: Number of worst predictions to save as images (default 10)
            output_dir: Directory to save images (default: reports/evaluation_mistakes)
        
        Returns:
            (accuracy_percent, correct_count, incorrect_count)
        """
        # Use provided generator or fall back to instance variable
        if test_generator is None:
            test_generator = self.test_generator
        
        # Use generator if available (preferred - memory efficient)
        if test_generator is not None:
            print("Using generator-based evaluation (memory efficient)...")
            # Get predictions from generator
            predictions = self.model.predict(test_generator)
            # Generator labels are one-hot encoded
            age_test = np.argmax(test_generator.labels, axis=1)
            X_test = None  # Can't easily get images from generator for worst image saving
            
        else:
            # Fall back to array-based data
            if X_test is None:
                X_test = self.X_test
            if age_test is None:
                age_test = self.age_test
            
            # Get test data
            if X_test is not None and isinstance(X_test, np.ndarray):
                pass  # X_test is ready
            else:
                raise ValueError("No test data available (test_generator or X_test required)")
            
            # Get age test labels (convert from one-hot if needed)
            if age_test is not None:
                if age_test.ndim > 1:
                    # One-hot encoded - convert to class indices
                    age_test = np.argmax(age_test, axis=1)
            else:
                raise ValueError("age_test not available for evaluation")
            
            # Make predictions on array
            predictions = self.model.predict(X_test)
        
        # Use Evaluation class for comprehensive analysis
        evaluator = Evaluation(AGE_CLASSES, self.find_age_class)
        accuracy, correct, incorrect = evaluator.evaluate(
            self.model,
            X_test,
            age_test,
            history=self.history if hasattr(self, 'history') else None,
            gen_test=gen_test,
            etn_test=etn_test,
            top_n=top_n,
            output_dir=output_dir
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