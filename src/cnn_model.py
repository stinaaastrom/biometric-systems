# Standard library imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Third-party imports
from sklearn.metrics import confusion_matrix

# TensorFlow/Keras imports
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam

AGE_CLASSES = [(0,12), (13,17), (18,25), (26,35), (36,45), (46,60), (61,74), (75, None)]

class CNNModel:
    
    def __init__(self, X_train, age_train, X_test, age_test):
        self.X_train = X_train
        self.age_train = age_train
        self.X_test = X_test
        self.age_test = age_test
        self.model = None
        self.history = None

    def find_age_class(self, predicted_age):
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


    def build_cnn_model(self):
        input_layer = Input(shape=(64, 64, 3))
        x = Conv2D(32, (3, 3), activation='relu')(input_layer)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)

        # Two outputs
        age_output = Dense(1, name='age_output')(x)

        self.model = Model(inputs=input_layer, outputs=age_output)
        self.model.compile(
            loss='mse',
            optimizer=Adam(learning_rate=0.001),
            metrics=['mae']
        )

        self.model.summary()

        self.history = self.model.fit(
            self.X_train,
            self.age_train,
            validation_data=(self.X_test, self.age_test),
            epochs=10,
            batch_size=64
        )

    def evaluate_model_performance(self):
        """
        Evaluate model performance by comparing age class predictions
        """
        # Make predictions on test set
        predictions = self.model.predict(self.X_test)
        predictions = predictions.flatten()  # Convert to 1D array
        
        # Convert predicted ages and actual ages to class indices
        predicted_classes = np.array([self.find_age_class(age) for age in predictions])
        actual_classes = np.array([self.find_age_class(age) for age in self.age_test])
        
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
        plt.plot(self.history.history['mae'], label='Train MAE')
        plt.plot(self.history.history['val_mae'], label='Val MAE')
        plt.title('Age Mean Absolute Error')
        plt.xlabel('Epoch')
        plt.ylabel('MAE (years)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        return accuracy, correct_predictions, total_predictions - correct_predictions
    
    def save_model(self, filepath='../models/age_model.keras'):
        """Save the trained model to a file"""
        if self.model is not None:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
        else:
            print("No model to save. Train the model first.")
    
    def load_model(self, filepath='../models/age_model.keras'):
        """Load a trained model from a file"""
        self.model = load_model(filepath)
        print(f"Model loaded from {filepath}")
        return self.model