import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import cv2


class Evaluation:
    """
    Encapsulated evaluator for age prediction models.

    Usage:
        ev = Evaluation(AGE_CLASSES, find_age_class_fn)
        acc, correct, incorrect = ev.evaluate(model, X_test, age_test, history,
                                              gen_test=gen, etn_test=etn,
                                              top_n=10, output_dir=None)
    """

    def __init__(self, age_classes, find_age_class_fn):
        self.age_classes = age_classes
        self.find_age_class_fn = find_age_class_fn

    def _predict(self, model, test_generator):
        """Get predictions from model using generator."""
        print("Running predictions from generator...")
        predictions = model.predict(test_generator)
        
        # For regression model: predictions shape is (N, 1) - direct age values
        # Flatten to 1D array of ages
        if predictions.ndim == 2 and predictions.shape[1] == 1:
            predicted_ages = predictions.flatten()
        else:
            # Fallback for other shapes
            predicted_ages = predictions.ravel()
        
        return predicted_ages
    
    def _class_to_age(self, class_idx):
        """Convert age class index to representative age value (midpoint of range)."""
        min_age, max_age = self.age_classes[class_idx]
        if max_age is None:
            # For open-ended class (e.g., 75+), use min_age + 10 as estimate
            return min_age + 10
        else:
            # Use midpoint of age range
            return (min_age + max_age) / 2.0

    def _print_summary(self, actual_classes, predicted_classes, predictions_len):
        valid_mask = (actual_classes != None) & (predicted_classes != None)
        correct_predictions = int(np.sum((predicted_classes[valid_mask] == actual_classes[valid_mask])))
        total_predictions = int(predictions_len)
        accuracy = (correct_predictions / total_predictions * 100.0) if total_predictions > 0 else 0.0

        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS - AGE CLASS ACCURACY")
        print("="*50)
        print(f"Total predictions: {total_predictions}")
        print(f"Correct age class predictions: {correct_predictions}")
        print(f"Incorrect age class predictions: {total_predictions - correct_predictions}")
        print(f"Accuracy: {accuracy:.2f}%")
        print("\nAge Classes:")
        for idx, (min_age, max_age) in enumerate(self.age_classes):
            if max_age is None:
                print(f"  Class {idx}: {min_age}+")
            else:
                print(f"  Class {idx}: {min_age}-{max_age}")
        return accuracy, correct_predictions, total_predictions - correct_predictions

    def _print_classification_report(self, actual_classes, predicted_classes):
        print("\n" + "="*50)
        print("Classification Report by Age Period")
        print("="*50)
        unique_classes = np.unique(np.concatenate([actual_classes, predicted_classes]))
        unique_classes = unique_classes[unique_classes != None]
        labels = sorted([int(c) for c in unique_classes])
        age_class_labels = [
            f"{self.age_classes[i][0]}-{self.age_classes[i][1]}" if self.age_classes[i][1] else f"{self.age_classes[i][0]}+"
            for i in labels
        ]
        report = classification_report(
            actual_classes,
            predicted_classes,
            labels=labels,
            target_names=age_class_labels,
            zero_division=0
        )
        print(report)
        print("="*50 + "\n")

    def _print_detailed_stats(self, predictions, age_test, actual_classes, predicted_classes):
        print("\n" + "="*50)
        print("Detailed Statistics per Age Period")
        print("="*50)
        for class_idx in range(len(self.age_classes)):
            min_age, max_age = self.age_classes[class_idx]
            class_label = f"{min_age}-{max_age}" if max_age else f"{min_age}+"
            valid_actual = actual_classes != None
            valid_pred = predicted_classes != None
            true_mask = valid_actual & (actual_classes == class_idx)
            true_count = int(np.sum(true_mask))
            pred_mask = valid_pred & (predicted_classes == class_idx)
            pred_count = int(np.sum(pred_mask))
            correct_count = int(np.sum(true_mask & (predicted_classes == class_idx)))
            if true_count > 0:
                class_predictions = predictions[true_mask]
                class_actuals = age_test[true_mask]
                mae = float(np.mean(np.abs(class_predictions - class_actuals)))
                class_accuracy = (correct_count / true_count * 100.0) if true_count > 0 else 0.0
            else:
                mae = 0.0
                class_accuracy = 0.0
            print(f"\nAge Period: {class_label}")
            print(f"  True samples: {true_count}")
            print(f"  Predicted as this period: {pred_count}")
            print(f"  Correctly classified: {correct_count}")
            print(f"  Accuracy: {class_accuracy:.2f}% ({correct_count}/{true_count})")
            print(f"  Mean Absolute Error: {mae:.2f} years")
        print("\n" + "="*50 + "\n")

    def _print_under18_over25_analysis(self, predictions, age_test, actual_classes):
        UNDER_18_THRESHOLD = 18
        OVER_25_THRESHOLD = 25
        predicted_under_18_mask = predictions < UNDER_18_THRESHOLD
        actual_over_25_mask = age_test > OVER_25_THRESHOLD
        critical_case_mask = predicted_under_18_mask & actual_over_25_mask
        total_critical = int(np.sum(critical_case_mask))
        percent_critical = (total_critical / len(age_test) * 100.0) if len(age_test) > 0 else 0.0

        critical_actual_classes = actual_classes[critical_case_mask]
        critical_actual_classes_valid = critical_actual_classes[critical_actual_classes != None]
        breakdown = {}
        for class_idx in range(len(self.age_classes)):
            count = int(np.sum(critical_actual_classes_valid == class_idx))
            if count > 0:
                min_age, max_age = self.age_classes[class_idx]
                label = f"{min_age}-{max_age}" if max_age else f"{min_age}+"
                breakdown[label] = count

        print("\n" + "="*50)
        print("Pred<18 while True>25 Analysis")
        print("="*50)
        print(f"Cases: {total_critical}/{len(age_test)} ({percent_critical:.2f}%)")
        if total_critical > 0:
            print("Breakdown by true age class:")
            for class_idx in range(len(self.age_classes)):
                min_age, max_age = self.age_classes[class_idx]
                label = f"{min_age}-{max_age}" if max_age else f"{min_age}+"
                if label in breakdown:
                    share = breakdown[label] / total_critical * 100.0
                    print(f"  {label}: {breakdown[label]} ({share:.2f}%)")
        print("\n" + "="*50 + "\n")

    def _demographic_analysis(self, actual_classes, predicted_classes, gen_test, etn_test):
        if gen_test is None or etn_test is None:
            return
        print("\n" + "="*50)
        print("Demographic Analysis of Misclassifications")
        print("="*50)
        valid_class_mask = (actual_classes != None) & (predicted_classes != None)
        misclassified_mask = valid_class_mask & (predicted_classes != actual_classes)
        gen_test_array = np.array(gen_test)
        etn_test_array = np.array(etn_test)
        known_mask = (gen_test_array != 'unknown') & (etn_test_array != 'unknown')
        print("\nGender Misclassification Analysis:")
        unique_genders = np.unique(gen_test_array[known_mask])
        for gender in unique_genders:
            gender_mask = (gen_test_array == gender) & known_mask
            total_gender = int(np.sum(gender_mask))
            misclassified_gender = int(np.sum(gender_mask & misclassified_mask))
            if total_gender > 0:
                error_rate = (misclassified_gender / total_gender) * 100.0
                print(f"  {gender}: {misclassified_gender}/{total_gender} misclassified ({error_rate:.2f}%)")
        print("\nEthnicity Misclassification Analysis:")
        unique_ethnicities = np.unique(etn_test_array[known_mask])
        for ethnicity in unique_ethnicities:
            ethnicity_mask = (etn_test_array == ethnicity) & known_mask
            total_ethnicity = int(np.sum(ethnicity_mask))
            misclassified_ethnicity = int(np.sum(ethnicity_mask & misclassified_mask))
            if total_ethnicity > 0:
                error_rate = (misclassified_ethnicity / total_ethnicity) * 100.0
                print(f"  {ethnicity}: {misclassified_ethnicity}/{total_ethnicity} misclassified ({error_rate:.2f}%)")
        print("\n" + "="*50 + "\n")

    def _plot_metrics(self, history, actual_classes, predicted_classes):
        cm = confusion_matrix(actual_classes, predicted_classes)
        has_history = history is not None and hasattr(history, 'history')
        fig_width = 16 if has_history else 8

        plt.figure(figsize=(fig_width, 5))
        if has_history:
            plt.subplot(1, 2, 1)
            # History keys may differ; fall back if not present
            if 'mae' in history.history and 'val_mae' in history.history:
                plt.plot(history.history['mae'], label='Train MAE')
                plt.plot(history.history['val_mae'], label='Val MAE')
                plt.ylabel('MAE (years)')
            else:
                if 'accuracy' in history.history:
                    plt.plot(history.history['accuracy'], label='Train Acc')
                if 'val_accuracy' in history.history:
                    plt.plot(history.history['val_accuracy'], label='Val Acc')
                plt.ylabel('Accuracy')
            plt.title('Training History')
            plt.xlabel('Epoch')
            plt.legend()
            plt.grid(True)
            plt.subplot(1, 2, 2)
        else:
            plt.subplot(1, 1, 1)

        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[f"{min_age}-{max_age}" if max_age else f"{min_age}+" for min_age, max_age in self.age_classes],
            yticklabels=[f"{min_age}-{max_age}" if max_age else f"{min_age}+" for min_age, max_age in self.age_classes]
        )
        plt.title('Confusion Matrix - Age Classes')
        plt.xlabel('Predicted Age Class')
        plt.ylabel('Actual Age Class')
        plt.tight_layout()
        plt.show()

    def evaluate(self, model, test_generator, history=None, gen_test=None, etn_test=None):
        """
        Evaluate model performance using test generator.
        
        Args:
            model: Trained Keras model
            test_generator: Keras Sequence generator for test data
            history: Training history object (optional)
            gen_test: Gender labels (optional)
            etn_test: Ethnicity labels (optional)
        
        Returns:
            (accuracy_percent, correct_count, incorrect_count)
        """
        print("Using generator-based evaluation (memory efficient)...")
        
        # Get predictions from generator (regression: direct age values)
        predictions = self._predict(model, test_generator)
        
        # Get actual age values from generator
        # Check if labels are one-hot encoded (classification) or scalar (regression)
        if hasattr(test_generator, 'labels'):
            labels = test_generator.labels
            if labels.ndim == 2 and labels.shape[1] > 1:
                # One-hot encoded - convert to class indices then to ages
                actual_classes = np.argmax(labels, axis=1)
                age_test = np.array([self._class_to_age(cls) for cls in actual_classes])
            else:
                # Already scalar ages
                age_test = labels.flatten() if labels.ndim > 1 else labels
                actual_classes = np.array([self.find_age_class_fn(age) for age in age_test])
        else:
            # Fallback: collect from generator batches
            age_test_list = []
            for i in range(len(test_generator)):
                _, y_batch = test_generator[i]
                if y_batch.ndim == 2 and y_batch.shape[1] > 1:
                    # One-hot to ages
                    batch_classes = np.argmax(y_batch, axis=1)
                    batch_ages = np.array([self._class_to_age(cls) for cls in batch_classes])
                else:
                    batch_ages = y_batch.flatten() if y_batch.ndim > 1 else y_batch
                age_test_list.append(batch_ages)
            age_test = np.concatenate(age_test_list)
            actual_classes = np.array([self.find_age_class_fn(age) for age in age_test])
        
        # Drop any samples where age is None/NaN to avoid comparison errors
        age_test = np.array(age_test, dtype=object)
        predictions = np.array(predictions, dtype=float)
        valid_mask = []
        for idx, age_val in enumerate(age_test):
            if age_val is None:
                valid_mask.append(False)
                continue
            if isinstance(age_val, float) and np.isnan(age_val):
                valid_mask.append(False)
                continue
            valid_mask.append(True)
        valid_mask = np.array(valid_mask, dtype=bool)

        if not np.any(valid_mask):
            raise ValueError("No valid age labels found for evaluation")

        age_test = age_test[valid_mask].astype(float)
        predictions = predictions[valid_mask]

        # Calculate predicted and actual classes on the filtered data
        actual_classes = np.array([self.find_age_class_fn(age) for age in age_test])
        predicted_classes = np.array([self.find_age_class_fn(age) for age in predictions])

        # Print all evaluation metrics
        accuracy, correct_predictions, incorrect = self._print_summary(actual_classes, predicted_classes, len(predictions))
        self._print_classification_report(actual_classes, predicted_classes)
        self._print_detailed_stats(predictions, age_test, actual_classes, predicted_classes)
        self._print_under18_over25_analysis(predictions, age_test, actual_classes)
        self._demographic_analysis(actual_classes, predicted_classes, gen_test, etn_test)
        self._plot_metrics(history, actual_classes, predicted_classes)

        return accuracy, correct_predictions, incorrect


# Backward-compatible functional API
def evaluate_model_performance(model, test_generator, history, AGE_CLASSES, find_age_class_fn, gen_test=None, etn_test=None):
    ev = Evaluation(AGE_CLASSES, find_age_class_fn)
    return ev.evaluate(model, test_generator, history, gen_test=gen_test, etn_test=etn_test)

