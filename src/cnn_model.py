# Standard library imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Third-party imports
from sklearn.metrics import confusion_matrix, classification_report

# TensorFlow/Keras imports
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50

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


    def build_cnn_model(self):
        """Build CNN model using ResNet50 transfer learning."""
        print("\nLoading ResNet50 with transfer learning...")
        
        # Load pre-trained ResNet50 (without top classification layer)
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(64, 64, 3)
        )
        
        # Freeze only the first 100 layers, allow fine-tuning of later layers
        base_model.trainable = True
        for layer in base_model.layers[:100]:
            layer.trainable = False
        
        print(f"Frozen first 100 layers, {len([l for l in base_model.layers if l.trainable])} layers trainable")
        
        # Build model on top of pre-trained base
        input_layer = Input(shape=(64, 64, 3))
        x = base_model(input_layer, training=True)
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(128, activation='relu')(x)
        age_output = Dense(1, name='age_output')(x)
        
        print("ResNet50 transfer learning model created")

        self.model = Model(inputs=input_layer, outputs=age_output)
        self.model.compile(
            loss='mse',
            optimizer=Adam(learning_rate=0.001),
            metrics=['mae']
        )

        self.model.summary()

        print("\nStarting training...")
        self.history = self.model.fit(
            self.X_train,
            self.age_train,
            validation_data=(self.X_test, self.age_test),
            epochs=5,
            batch_size=128,
            verbose=1
        )
        print("Training complete!")

    def evaluate_model_performance(self, gen_test=None, etn_test=None):
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
        
        # Classification report by age period
        print("\n" + "="*50)
        print("Classification Report by Age Period")
        print("="*50)
        
        # Find which classes are actually present in the test set
        unique_classes = np.unique(np.concatenate([actual_classes, predicted_classes]))
        unique_classes = unique_classes[unique_classes != None]  # Remove None values if any
        
        # Create labels only for classes that exist
        labels = sorted([int(c) for c in unique_classes])
        age_class_labels = [f"{AGE_CLASSES[i][0]}-{AGE_CLASSES[i][1]}" if AGE_CLASSES[i][1] else f"{AGE_CLASSES[i][0]}+" 
                           for i in labels]
        
        # Generate classification report only for existing classes
        report = classification_report(
            actual_classes, 
            predicted_classes, 
            labels=labels,
            target_names=age_class_labels,
            zero_division=0
        )
        print(report)
        print("="*50 + "\n")
        
        # Detailed statistics per age period
        print("\n" + "="*50)
        print("Detailed Statistics per Age Period")
        print("="*50)
        
        for class_idx in range(len(AGE_CLASSES)):
            min_age, max_age = AGE_CLASSES[class_idx]
            class_label = f"{min_age}-{max_age}" if max_age else f"{min_age}+"
            
            # True samples (actual class)
            true_mask = actual_classes == class_idx
            true_count = np.sum(true_mask)
            
            # Predicted as this period
            pred_mask = predicted_classes == class_idx
            pred_count = np.sum(pred_mask)
            
            # Correctly classified (true positives)
            correct_count = np.sum((actual_classes == class_idx) & (predicted_classes == class_idx))
            
            # MAE for this specific age class
            if true_count > 0:
                class_predictions = predictions[true_mask]
                class_actuals = self.age_test[true_mask]
                mae = np.mean(np.abs(class_predictions - class_actuals))
            else:
                mae = 0.0
            
            print(f"\nAge Period: {class_label}")
            print(f"  True samples: {true_count}")
            print(f"  Predicted as this period: {pred_count}")
            print(f"  Correctly classified: {correct_count}")
            print(f"  Mean Absolute Error: {mae:.2f} years")
        
        print("\n" + "="*50 + "\n")
        
        # Demographic analysis (gender and ethnicity misclassifications)
        if gen_test is not None and etn_test is not None:
            print("\n" + "="*50)
            print("Demographic Analysis of Misclassifications")
            print("="*50)
            
            # Create mask for misclassified samples
            misclassified_mask = predicted_classes != actual_classes
            
            # Filter out 'unknown' values
            gen_test_array = np.array(gen_test)
            etn_test_array = np.array(etn_test)
            
            known_mask = (gen_test_array != 'unknown') & (etn_test_array != 'unknown')
            
            # Gender analysis
            print("\nGender Misclassification Analysis:")
            unique_genders = np.unique(gen_test_array[known_mask])
            for gender in unique_genders:
                gender_mask = (gen_test_array == gender) & known_mask
                total_gender = np.sum(gender_mask)
                misclassified_gender = np.sum(gender_mask & misclassified_mask)
                if total_gender > 0:
                    error_rate = (misclassified_gender / total_gender) * 100
                    print(f"  {gender}: {misclassified_gender}/{total_gender} misclassified ({error_rate:.2f}%)")
            
            # Ethnicity analysis
            print("\nEthnicity Misclassification Analysis:")
            unique_ethnicities = np.unique(etn_test_array[known_mask])
            for ethnicity in unique_ethnicities:
                ethnicity_mask = (etn_test_array == ethnicity) & known_mask
                total_ethnicity = np.sum(ethnicity_mask)
                misclassified_ethnicity = np.sum(ethnicity_mask & misclassified_mask)
                if total_ethnicity > 0:
                    error_rate = (misclassified_ethnicity / total_ethnicity) * 100
                    print(f"  {ethnicity}: {misclassified_ethnicity}/{total_ethnicity} misclassified ({error_rate:.2f}%)")
            
            print("\n" + "="*50 + "\n")
        
        # Create confusion matrix
        cm = confusion_matrix(actual_classes, predicted_classes)
        
        # Plot training history and confusion matrix
        plt.figure(figsize=(16, 5))
        
        # Plot 1: MAE history
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['mae'], label='Train MAE')
        plt.plot(self.history.history['val_mae'], label='Val MAE')
        plt.title('Age Mean Absolute Error')
        plt.xlabel('Epoch')
        plt.ylabel('MAE (years)')
        plt.legend()
        plt.grid(True)
        
        # Plot 2: Confusion Matrix
        plt.subplot(1, 2, 2)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=[f"{min_age}-{max_age}" if max_age else f"{min_age}+" 
                                 for min_age, max_age in AGE_CLASSES],
                    yticklabels=[f"{min_age}-{max_age}" if max_age else f"{min_age}+" 
                                 for min_age, max_age in AGE_CLASSES])
        plt.title('Confusion Matrix - Age Classes')
        plt.xlabel('Predicted Age Class')
        plt.ylabel('Actual Age Class')
        
        plt.tight_layout()
        plt.show()
        
        return accuracy, correct_predictions, total_predictions - correct_predictions
    
    def save_model(self, filepath=None):
        """Save the trained model to a file"""
        if self.model is None:
            print("No model to save. Train the model first.")
            return
        
        if filepath is None:
            filepath = DEFAULT_MODEL_PATH
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath=None):
        """Load a trained model from a file"""
        if filepath is None:
            filepath = DEFAULT_MODEL_PATH
        
        self.model = load_model(filepath)
        print(f"Model loaded from {filepath}")
        return self.model