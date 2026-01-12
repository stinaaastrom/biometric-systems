# Standard library imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Third-party imports
from sklearn.metrics import confusion_matrix



# TensorFlow/Keras imports
import tensorflow as tf
from keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
    GlobalAveragePooling2D,
    Rescaling,
    Lambda,
)
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam, AdamW
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical
from keras.applications.efficientnet import EfficientNetB1, preprocess_input
from keras.regularizers import L2

AGE_CLASSES = [(0,12), (13,17), (18,25), (26,35), (36,45), (46,60), (61,74), (75, None)]

# Default path for model files (absolute path)
import os
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'age_model.keras')

class CNNModel:
    
    def __init__(self, X_train, age_train, X_test, age_test):
        self.X_train = X_train
        self.age_train = age_train
        self.X_test = X_test
        self.age_test = age_test
        self.model = None
        self.history = None
        self.temperature = 1.0  # Temperature scaling parameter

    @staticmethod
    def find_age_class(predicted_age):
        """
        Returns the index of the class an age belongs to,
        or None if age is outside all defined ranges.
        """
        for idx, (min_age, max_age) in enumerate(AGE_CLASSES):
            if max_age is None:
                if predicted_age >= min_age:
                    return idx
            elif min_age <= predicted_age <= max_age:
                return idx
        return None

    @staticmethod
    def map_age_to_class(age_value):
        """Map a continuous age (float) to the discrete class index based on AGE_CLASSES.
        Clamps ages outside the defined ranges to the nearest class boundary.
        """
        try:
            age = float(age_value)
        except Exception:
            return 0

        # Handle NaN or non-finite values
        if not np.isfinite(age):
            return 0

        # Clamp to valid range
        min0 = AGE_CLASSES[0][0]
        last_idx = len(AGE_CLASSES) - 1
        last_min = AGE_CLASSES[last_idx][0]
        if age < min0:
            return 0
        if age >= last_min:
            return last_idx

        # Find matching class
        idx = CNNModel.find_age_class(age)
        return 0 if idx is None else idx

    @staticmethod
    def nudge_probabilities_by_regression(probs, predicted_age, boundary_threshold=1.5):
        """
        Soft combination: nudge softmax probabilities using regression output.
        If regression predicts an age near a class boundary, transfer some probability
        mass to the adjacent class to handle boundary confusion.
        
        Args:
            probs: Softmax probabilities (1D array of shape [8])
            predicted_age: Regression output (continuous age estimate)
            boundary_threshold: Distance from class boundary to apply nudging (in years)
        
        Returns:
            Adjusted probabilities (1D array of shape [8])
        """
        adjusted_probs = probs.copy()
        
        # Check if age is near any class boundary
        for class_idx in range(len(AGE_CLASSES)):
            min_age, max_age = AGE_CLASSES[class_idx]
            
            # Upper boundary (if not last class)
            if max_age is not None and max_age < AGE_CLASSES[-1][0] + 100:
                upper_bound = max_age + 0.5
                if abs(predicted_age - upper_bound) < boundary_threshold and class_idx < len(AGE_CLASSES) - 1:
                    # Age is near upper boundary: transfer some prob to next class
                    transfer = 0.15 * probs[class_idx]  # Transfer 15% of current class prob
                    adjusted_probs[class_idx] -= transfer
                    adjusted_probs[class_idx + 1] += transfer
            
            # Lower boundary (if not first class)
            if class_idx > 0:
                lower_bound = min_age - 0.5
                if abs(predicted_age - lower_bound) < boundary_threshold:
                    # Age is near lower boundary: transfer some prob to prev class
                    transfer = 0.15 * probs[class_idx]
                    adjusted_probs[class_idx] -= transfer
                    adjusted_probs[class_idx - 1] += transfer
        
        # Renormalize to ensure sum=1
        return adjusted_probs / np.sum(adjusted_probs)

    def apply_temperature_scaling(self, logits):
        """Apply temperature scaling to logits before softmax.
        
        Args:
            logits: Pre-softmax logits (numpy array)
        
        Returns:
            Calibrated probabilities after temperature scaling
        """
        scaled_logits = logits / self.temperature
        # Apply softmax manually
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    def calibrate_temperature(self, val_data=None, val_labels=None):
        """Calibrate temperature parameter using validation data to minimize NLL.
        
        Args:
            val_data: Validation images (if None, uses self.X_test)
            val_labels: Validation class indices (if None, computed from self.age_test)
        
        Returns:
            Optimal temperature value
        """
        if self.model is None:
            print("Error: No model loaded. Train or load a model first.")
            return 1.0
        
        # Use test data as validation if not provided
        if val_data is None:
            val_data = self.X_test
        if val_labels is None:
            val_labels = np.array([self.find_age_class(age) for age in self.age_test])
        
        print("\nCalibrating temperature scaling...")
        
        # Get logits from the model (we need pre-softmax values)
        # Extract the layer before final softmax
        from keras.models import Model
        logit_model = Model(inputs=self.model.input, 
                           outputs=self.model.get_layer('age_output').input)
        
        logits = logit_model.predict(val_data, verbose=0)
        
        # Optimize temperature using scipy
        from scipy.optimize import minimize
        
        def nll_loss(T):
            """Negative log likelihood loss for temperature T"""
            T = max(T[0], 0.01)  # Avoid division by zero
            scaled_logits = logits / T
            # Compute softmax
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            # Compute NLL
            log_probs = np.log(probs[np.arange(len(val_labels)), val_labels] + 1e-12)
            return -np.mean(log_probs)
        
        # Find optimal temperature
        result = minimize(nll_loss, [1.0], method='Nelder-Mead', 
                         options={'maxiter': 100, 'xatol': 0.001})
        
        optimal_T = max(result.x[0], 0.01)
        self.temperature = optimal_T
        
        # Report results
        before_nll = nll_loss([1.0])
        after_nll = nll_loss([optimal_T])
        
        print(f"Optimal temperature: {optimal_T:.3f}")
        print(f"NLL before calibration: {before_nll:.4f}")
        print(f"NLL after calibration: {after_nll:.4f}")
        print(f"Improvement: {((before_nll - after_nll) / before_nll * 100):.2f}%")
        
        # Show confidence distribution change
        probs_before = self.model.predict(val_data, verbose=0)[0]
        conf_before = np.max(probs_before, axis=1)
        
        scaled_logits = logits / optimal_T
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
        probs_after = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        conf_after = np.max(probs_after, axis=1)
        
        print(f"\nConfidence statistics:")
        print(f"Before - Mean: {np.mean(conf_before):.3f}, Std: {np.std(conf_before):.3f}")
        print(f"After  - Mean: {np.mean(conf_after):.3f}, Std: {np.std(conf_after):.3f}")
        
        return optimal_T

    def build_cnn_model(self):
        # Convert ages to class indices
        train_classes = np.array([self.find_age_class(age) for age in self.age_train])
        test_classes = np.array([self.find_age_class(age) for age in self.age_test])

        # Compute class weights to balance training
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight('balanced', classes=np.unique(train_classes), y=train_classes)
        class_weight_dict = {i: class_weights[i] for i in range(8)}
        
        # Boost weak classes (26-35, 36-45) and mildly increase adjacent ranges
        class_weight_dict[2] *= 1.35  # 18-25
        class_weight_dict[3] *= 1.6   # 26-35 (main problem class)
        class_weight_dict[4] *= 1.5   # 36-45 (consistently weak)
        class_weight_dict[5] *= 1.3   # 46-60
        class_weight_dict[6] *= 1.2   # 61-74
        
        print("\nClass weights:")
        for idx, weight in class_weight_dict.items():
            min_age, max_age = AGE_CLASSES[idx]
            label = f"{min_age}+" if max_age is None else f"{min_age}-{max_age}"
            print(f"  Class {idx} ({label}): {weight:.3f}")

        # Build Gaussian-smoothed label distributions (Label Distribution Learning)
        def gaussian_soft_labels(indices, num_classes, sigma=1.2):
            labels = np.zeros((len(indices), num_classes), dtype=np.float32)
            classes = np.arange(num_classes, dtype=np.float32)
            for n, i in enumerate(indices):
                d = classes - float(i)
                p = np.exp(-0.5 * (d / sigma) ** 2)
                p /= p.sum()
                labels[n] = p
            return labels

        y_train = gaussian_soft_labels(train_classes, 8, sigma=1.5)
        y_test = gaussian_soft_labels(test_classes, 8, sigma=1.5)

        # Input at 224x224 for EfficientNetB1
        input_layer = Input(shape=(224, 224, 3))
        # Our dataset is normalized to [0,1]; EfficientNet preprocess expects [0,255]
        x = Rescaling(255.0)(input_layer)
        # Note: Skip Lambda(preprocess_input) due to serialization issues
        # Data normalization is sufficient for EfficientNet

        # Backbone: EfficientNetB1 (Imagenet). Fallback to random init if weights unavailable.
        try:
            base = EfficientNetB1(include_top=False, weights='imagenet', input_tensor=x, pooling='avg')
        except Exception:
            base = EfficientNetB1(include_top=False, weights=None, input_tensor=x, pooling='avg')

        # Head with L2 regularization + higher dropout
        h = Dropout(0.5)(base.output)
        h = Dense(256, activation='relu', kernel_regularizer=L2(0.001))(h)
        h = Dropout(0.6)(h)
        # Classification output
        age_output = Dense(8, activation='softmax', kernel_regularizer=L2(0.001), name='age_output')(h)
        # Regression output (continuous age estimate)
        age_reg = Dense(1, activation='linear', name='age_reg')(h)

        self.model = Model(inputs=input_layer, outputs=[age_output, age_reg])

        # Callbacks with aggressive early stopping to prevent overfitting
        early_stop = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
        
        # Reshape ages for regression head (needs shape (n, 1))
        age_train_reg = self.age_train.reshape(-1, 1).astype(np.float32)
        age_test_reg = self.age_test.reshape(-1, 1).astype(np.float32)

        # Stage 1: freeze backbone with label smoothing + regularization
        base.trainable = False
        self.model.compile(
            optimizer=AdamW(learning_rate=3e-4, weight_decay=1e-4), 
            loss={
                'age_output': tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                'age_reg': tf.keras.losses.Huber()
            },
            loss_weights={'age_output': 0.6, 'age_reg': 0.4},
            metrics={'age_output': 'accuracy', 'age_reg': 'mae'}
        )
        self.model.summary()
        self.history = self.model.fit(
            self.X_train,
            {'age_output': y_train, 'age_reg': age_train_reg},
            validation_data=(self.X_test, {'age_output': y_test, 'age_reg': age_test_reg}),
            epochs=8,
            batch_size=32,
            callbacks=[reduce_lr]
        )

        # Stage 2: fine-tune backbone with label smoothing + regularization
        base.trainable = True
        self.model.compile(
            optimizer=AdamW(learning_rate=2e-4, weight_decay=1e-4), 
            loss={
                'age_output': tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                'age_reg': tf.keras.losses.Huber()
            },
            loss_weights={'age_output': 0.6, 'age_reg': 0.4},
            metrics={'age_output': 'accuracy', 'age_reg': 'mae'}
        )
        self.history = self.model.fit(
            self.X_train,
            {'age_output': y_train, 'age_reg': age_train_reg},
            validation_data=(self.X_test, {'age_output': y_test, 'age_reg': age_test_reg}),
            epochs=60,
            batch_size=32,
            callbacks=[early_stop, reduce_lr]
        )

    def evaluate_model_performance(self, gen_test=None, etn_test=None):
        """
        Evaluate model performance by comparing age class predictions
        """
        # Make predictions on test set: [class_probs, age_reg]
        pred_probs, pred_reg = self.model.predict(self.X_test)
        predicted_classes = np.argmax(pred_probs, axis=1)
        # Also compute class predictions from regression (binning)
        pred_reg_classes = np.array([self.map_age_to_class(a) for a in pred_reg.flatten()])
        
        # Convert actual ages to class indices
        actual_classes = np.array([self.find_age_class(age) for age in self.age_test])
        
        # Filter out None values (ages outside defined ranges)
        valid_mask = actual_classes != None
        predicted_classes = predicted_classes[valid_mask]
        actual_classes = actual_classes[valid_mask]
        
        # Also filter demographic data if provided
        if gen_test is not None:
            gen_test = np.array(gen_test)[valid_mask]
        if etn_test is not None:
            etn_test = np.array(etn_test)[valid_mask]
        
        # Calculate accuracy (softmax-based)
        correct_predictions = np.sum(predicted_classes == actual_classes)
        total_predictions = len(pred_probs)
        accuracy = (correct_predictions / total_predictions) * 100

        # Regression-binning exact accuracy
        reg_exact = np.sum(pred_reg_classes == actual_classes)
        reg_exact_acc = (reg_exact / total_predictions) * 100

        # Within-one-class accuracy for regression-binning
        within_one = np.sum(np.abs(pred_reg_classes - actual_classes) <= 1)
        within_one_acc = (within_one / total_predictions) * 100
        
        # Print results
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS - AGE CLASS ACCURACY")
        print("="*50)
        print(f"Total predictions: {total_predictions}")
        print(f"Correct age class predictions: {correct_predictions}")
        print(f"Incorrect age class predictions: {total_predictions - correct_predictions}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Regression-binning (exact): {reg_exact_acc:.2f}%")
        print(f"Regression-binning (±1 class): {within_one_acc:.2f}%")
        print("\nAge Classes:")
        for idx, (min_age, max_age) in enumerate(AGE_CLASSES):
            if max_age is None:
                print(f"  Class {idx}: {min_age}+")
            else:
                print(f"  Class {idx}: {min_age}-{max_age}")
        
        # Per-class accuracy (softmax)
        print("\nPer-class accuracy:")
        for class_idx in range(len(AGE_CLASSES)):
            class_mask = actual_classes == class_idx
            if np.sum(class_mask) > 0:
                class_correct = np.sum(predicted_classes[class_mask] == class_idx)
                class_total = np.sum(class_mask)
                class_accuracy = (class_correct / class_total) * 100
                min_age, max_age = AGE_CLASSES[class_idx]
                class_label = f"{min_age}+" if max_age is None else f"{min_age}-{max_age}"
                print(f"  Class {class_idx} ({class_label}): {class_accuracy:.2f}% ({class_correct}/{class_total})")
        
        # Confusion matrix (softmax)
        cm = confusion_matrix(actual_classes, predicted_classes)
        print("\nConfusion Matrix (rows=actual, cols=predicted):")
        print("     ", end="")
        for i in range(8):
            print(f"{i:4}", end="")
        print()
        for i in range(8):
            min_age, max_age = AGE_CLASSES[i]
            label = f"{min_age}+" if max_age is None else f"{min_age}-{max_age}"
            print(f"{i} ({label:>7}): ", end="")
            for j in range(8):
                print(f"{cm[i,j]:4}", end="")
            print()
        
        # Confusion matrix (regression-binning)
        cm_reg = confusion_matrix(actual_classes, pred_reg_classes)
        print("\nConfusion Matrix (Regression-binning):")
        print("     ", end="")
        for i in range(8):
            print(f"{i:4}", end="")
        print()
        for i in range(8):
            min_age, max_age = AGE_CLASSES[i]
            label = f"{min_age}+" if max_age is None else f"{min_age}-{max_age}"
            print(f"{i} ({label:>7}): ", end="")
            for j in range(8):
                print(f"{cm_reg[i,j]:4}", end="")
            print()
        print("="*50 + "\n")
        
        # Plot training history
        plt.figure(figsize=(12, 5))
        
        if self.history is not None and hasattr(self.history, 'history'):
            hist = self.history.history
            if ('accuracy' in hist and 'val_accuracy' in hist and 'loss' in hist and 'val_loss' in hist):
                plt.subplot(1, 2, 1)
                plt.plot(hist['accuracy'], label='Train Accuracy')
                plt.plot(hist['val_accuracy'], label='Val Accuracy')
                plt.title('Model Accuracy')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.legend()
                plt.grid(True)
                
                plt.subplot(1, 2, 2)
                plt.plot(hist['loss'], label='Train Loss')
                plt.plot(hist['val_loss'], label='Val Loss')
                plt.title('Model Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True)
                
                plt.tight_layout()
                plt.show(block=False)
                plt.pause(0.1)  # Allow plot to render without blocking
        
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
        
        # Also save test data for later evaluation
        test_data_path = filepath.replace('.keras', '_testdata.npz')
        try:
            np.savez_compressed(
                test_data_path,
                X_test=self.X_test,
                age_test=self.age_test,
                temperature=self.temperature
            )
            print(f"Test data saved to {test_data_path}")
        except Exception as e:
            print(f"Warning: Could not save test data: {e}")
    
    def load_model(self, filepath=None):
        """Load a trained model and test data from files"""
        if filepath is None:
            filepath = DEFAULT_MODEL_PATH
        
        # Custom objects to handle Lambda layer with preprocess_input
        from keras.applications.efficientnet import preprocess_input as enet_preprocess
        custom_objs = {
            'preprocess_input': enet_preprocess,
        }
        
        try:
            self.model = load_model(filepath, custom_objects=custom_objs)
        except Exception as e:
            print(f"Load with custom_objects failed: {e}")
            print("Trying without custom_objects...")
            self.model = load_model(filepath)
        
        print(f"Model loaded from {filepath}")
        
        # Also load test data if it exists
        test_data_path = filepath.replace('.keras', '_testdata.npz')
        if os.path.exists(test_data_path):
            try:
                test_data = np.load(test_data_path, allow_pickle=True)
                self.X_test = test_data['X_test']
                self.age_test = test_data['age_test']
                if 'temperature' in test_data:
                    self.temperature = float(test_data['temperature'])
                    print(f"Temperature loaded: {self.temperature:.3f}")
                print(f"Test data loaded from {test_data_path} ({len(self.age_test)} samples)")
            except Exception as e:
                print(f"Warning: Could not load test data: {e}")
        else:
            print(f"Warning: Test data file not found at {test_data_path}")
        
        return self.model