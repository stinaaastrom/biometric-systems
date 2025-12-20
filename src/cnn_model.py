# Standard library imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

# Third-party imports
from sklearn.metrics import confusion_matrix



# TensorFlow/Keras imports
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical

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

    def _apply_augmentation(self, images):
        """
        Apply data augmentation to a batch of images using ImageProcesser.
        Images should be in range [0, 1] and will be converted to uint8 for augmentation.
        """
        from image_processing import ImageProcesser
        
        processor = ImageProcesser()
        transform = processor.get_augmentation_transform()
        
        augmented_images = []
        total = len(images)
        
        for idx, img in enumerate(images):
            # Convert to uint8 for albumentations (expects 0-255)
            img_uint8 = (img * 255).astype(np.uint8)
            
            # Apply augmentation
            augmented = transform(image=img_uint8)
            aug_img = augmented['image']
            
            # Convert back to float32 in range [0, 1]
            aug_img = aug_img.astype(np.float32) / 255.0
            augmented_images.append(aug_img)
            
            if (idx + 1) % 500 == 0:
                print(f"Augmented {idx + 1}/{total} images", end='\r')
        
        print(f"Augmented {total}/{total} images ✓")
        return np.array(augmented_images)


    def build_cnn_model(self, use_augmentation=True):

        # Convert ages to class indices
        train_classes = np.array([self.find_age_class(age) for age in self.age_train])
        test_classes  = np.array([self.find_age_class(age) for age in self.age_test])

        y_train = to_categorical(train_classes, num_classes=8)
        y_test  = to_categorical(test_classes, num_classes=8)
        
        # Apply data augmentation to training data
        if use_augmentation:
            print("Applying data augmentation to training set...")
            X_train_augmented = self._apply_augmentation(self.X_train)
        else:
            X_train_augmented = self.X_train

        input_layer = Input(shape=(224, 224, 3))

        # Preprocessing for ResNet50
        x = tf.keras.applications.resnet50.preprocess_input(input_layer)

        # Backbone: ResNet50
        self.base_model = tf.keras.applications.ResNet50(
            include_top=False,
            weights="imagenet",
            input_tensor=x
        )

        self.base_model.trainable = False

        x = self.base_model.output
        x = GlobalAveragePooling2D()(x)

        x = Dense(512, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)

        x = Dense(256, activation="relu")(x)
        x = Dropout(0.4)(x)

        age_output = Dense(8, activation="softmax")(x)

        self.model = Model(inputs=input_layer, outputs=age_output)

        self.model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        self.model.summary()

        # -------- FIRST TRAIN --------
        early_stop_1 = EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True
        )

        reduce_lr_1 = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=5,
            min_lr=1e-6
        )

        self.history = self.model.fit(
            X_train_augmented, y_train,
            validation_data=(self.X_test, y_test),
            epochs=30,
            batch_size=32,
            callbacks=[early_stop_1, reduce_lr_1]
        )

        # -------- FINE TUNING --------
        # Unfreeze from conv4_block1 and forward
        set_trainable = False
        for layer in self.base_model.layers:
            if layer.name.startswith("conv4_block1"):
                set_trainable = True

            if set_trainable:
                # Keep BatchNorm frozen
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = False
                else:
                    layer.trainable = True

        self.model.compile(
            optimizer=Adam(learning_rate=1e-6),
            loss="categorical_crossentropy",
            metrics=["accuracy", self.age_mae]
        )

        early_stop_2 = EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True
        )

        reduce_lr_2 = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=4,
            min_lr=1e-6
        )

        self.history = self.model.fit(
            X_train_augmented, y_train,
            validation_data=(self.X_test, y_test),
            epochs=20,
            batch_size=32,
            callbacks=[early_stop_2, reduce_lr_2]
        )

    def age_mae(y_true, y_pred):
        class_centers = tf.constant([1, 4, 8, 13, 18, 30, 50, 70], dtype=tf.float32)
        y_true_age = tf.reduce_sum(y_true * class_centers, axis=1)
        y_pred_age = tf.reduce_sum(y_pred * class_centers, axis=1)
        return tf.reduce_mean(tf.abs(y_true_age - y_pred_age))


            

    def evaluate_model_performance(self, gen_test=None, etn_test=None):
        """
        Evaluate model performance by comparing age class predictions
        """
        # Make predictions on test set (returns probabilities for each class)
        predictions = self.model.predict(self.X_test)
        predicted_classes = np.argmax(predictions, axis=1)  # Get class with highest probability
        
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
        
        # Calculate accuracy
        correct_predictions = np.sum(predicted_classes == actual_classes)
        total_predictions = len(predictions)
        accuracy = (correct_predictions / total_predictions) * 100
        
        # Print results
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS - AGE CLASS ACCURACY")
        print("="*50)
        print(f"Total predictions: {total_predictions}")
        print(f"Correct age class predictions: {correct_predictions}")
        print(f"Incorrect age class predictions: {total_predictions - correct_predictions}")
        print(f"Accuracy: {accuracy:.2f}%")
        print("\nAge Classes:")
        for idx, (min_age, max_age) in enumerate(AGE_CLASSES):
            if max_age is None:
                print(f"  Class {idx}: {min_age}+")
            else:
                print(f"  Class {idx}: {min_age}-{max_age}")
        
        # Per-class accuracy
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
        print("="*50 + "\n")
        
        # Plot training history
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='Train Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Val Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='Train Loss')
        plt.plot(self.history.history['val_loss'], label='Val Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
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
        
        # Also save test data for later evaluation
        test_data_path = filepath.replace('.keras', '_testdata.npz')
        try:
            np.savez_compressed(
                test_data_path,
                X_test=self.X_test,
                age_test=self.age_test
            )
            print(f"Test data saved to {test_data_path}")
        except Exception as e:
            print(f"Warning: Could not save test data: {e}")
    
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