import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
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
        """Get predictions from model using generator.

        Returns:
            predicted_ages (np.ndarray): estimated ages using class midpoints
            predicted_probs (np.ndarray): raw class probabilities (N, num_classes)
        """
        print("Running predictions from generator...")
        predicted_probs = model.predict(test_generator)

        # Convert class probabilities to age values via class midpoint of argmax
        predicted_classes = np.argmax(predicted_probs, axis=1)
        predicted_ages = np.array([self._class_to_age(cls) for cls in predicted_classes])
        return predicted_ages, predicted_probs
    
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

    def _print_policy_summary(self, predicted_ages, age_test):
        """Print summary where only severe errors count:
        - Minor (<18) predicted as >25
        - Adult (>25) predicted as <18

        Returns (accuracy_percent, correct_count, incorrect_count)
        where accuracy is 1 - severe_error_rate.
        """
        ages_true = np.asarray(age_test)
        ages_pred = np.asarray(predicted_ages)
        total = ages_true.size

        minors_pred_over25 = (ages_true < 18) & (ages_pred > 25)
        adults_pred_under18 = (ages_true > 25) & (ages_pred < 18)
        severe_errors_mask = minors_pred_over25 | adults_pred_under18

        n_severe = int(np.sum(severe_errors_mask))
        n_correct = int(total - n_severe)
        acc = (n_correct / total * 100.0) if total > 0 else 0.0

        print("\n" + "="*50)
        print("POLICY EVALUATION RESULTS - SEVERE ERRORS ONLY")
        print("="*50)
        print(f"Total predictions: {total}")
        print(f"Severe errors: {n_severe}")
        print(f"  - Minor (<18) predicted >25: {int(np.sum(minors_pred_over25))}")
        print(f"  - Adult (>25) predicted <18: {int(np.sum(adults_pred_under18))}")
        print(f"Policy Accuracy: {acc:.2f}% (non-severe)")
        return acc, n_correct, n_severe

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

    def _decision_policy(self, pred_age):
        """Apply decision policy based on predicted age.

        Returns one of: 'approved', 'denied', 'human_review'.
        """
        if pred_age > 25:
            return 'approved'
        if pred_age < 18:
            return 'denied'
        return 'human_review'

    def _print_decision_summary(self, predicted_ages):
        decisions = np.array([self._decision_policy(a) for a in predicted_ages])
        total = decisions.size
        approved = int(np.sum(decisions == 'approved'))
        denied = int(np.sum(decisions == 'denied'))
        human = int(np.sum(decisions == 'human_review'))

        print("\n" + "="*50)
        print("Decision Policy Summary")
        print("="*50)
        print(f"Approved (>25): {approved} ({approved/total*100:.2f}%)")
        print(f"Denied (<18): {denied} ({denied/total*100:.2f}%)")
        print(f"Human review (18-25): {human} ({human/total*100:.2f}%)")
        print("\n" + "="*50 + "\n")

    def _compute_and_print_binary_metrics(self, predicted_ages, predicted_probs, age_test, output_dir=None):
        """Compute FPR/FNR and ROC/AUC for adult (>=18) detection.

        Also reports a special FPR for minors incorrectly classified as over 15.
        """
        ages = np.asarray(age_test)
        preds_age = np.asarray(predicted_ages)

        # Ground truth: adult if >=18
        y_true_adult = ages >= 18

        # Score: probability of being adult (sum of class probs with min_age>=18) if available,
        # fallback to predicted ages normalized.
        if predicted_probs is not None and predicted_probs.ndim == 2:
            # Determine indices of classes where min_age >= 18
            adult_class_idxs = [i for i, (min_age, _max) in enumerate(self.age_classes) if min_age >= 18]
            adult_scores = predicted_probs[:, adult_class_idxs].sum(axis=1)
        else:
            # Normalize ages to [0,1] using a cap at, say, 80 years
            adult_scores = np.clip(preds_age / 80.0, 0.0, 1.0)

        # Binary prediction using 18 threshold on predicted age
        y_pred_adult = preds_age >= 18

        # FPR (minors predicted as adult at 18 threshold)
        minors_mask = ~y_true_adult
        adults_mask = y_true_adult
        fpr_18 = float(np.sum(minors_mask & y_pred_adult)) / float(np.sum(minors_mask)) if np.sum(minors_mask) > 0 else 0.0
        fnr_18 = float(np.sum(adults_mask & (~y_pred_adult))) / float(np.sum(adults_mask)) if np.sum(adults_mask) > 0 else 0.0

        # Special FPR requested: minors (<18) predicted as >15
        fpr_minors_over15 = float(np.sum((ages < 18) & (preds_age > 15))) / float(np.sum(ages < 18)) if np.sum(ages < 18) > 0 else 0.0

        # ROC and AUC for adult detection
        try:
            fpr_curve, tpr_curve, _ = roc_curve(y_true_adult.astype(int), adult_scores)
            auc_val = auc(fpr_curve, tpr_curve)
        except Exception:
            fpr_curve, tpr_curve, auc_val = None, None, None

        print("\n" + "="*50)
        print("Binary Metrics (Adult >=18)")
        print("="*50)
        print(f"FPR @18 (minors predicted adult): {fpr_18:.4f}")
        print(f"FNR @18 (adults predicted minor): {fnr_18:.4f}")
        print(f"Special FPR (minors predicted >15): {fpr_minors_over15:.4f}")
        if auc_val is not None:
            print(f"ROC AUC (adult score): {auc_val:.4f}")
        print("\n" + "="*50 + "\n")

        # Optionally save ROC curve plot
        if auc_val is not None:
            self._plot_roc_curve(fpr_curve, tpr_curve, auc_val, output_dir=output_dir)

    def _plot_roc_curve(self, fpr, tpr, auc_val, output_dir=None):
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_val:.3f})')
        plt.plot([0, 0, 1], [0, 1, 1], color='navy', lw=1, linestyle='--', label='Ideal')
        plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle=':', label='Chance')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Adult (>=18) Detection')
        plt.legend(loc="lower right")
        plt.tight_layout()

        # Save to reports by default under project root
        save_dir = output_dir or os.path.join(os.path.dirname(__file__), 'reports')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'roc_curve_adult_detection.png')
        plt.savefig(save_path, dpi=150)
        print(f"ROC curve saved to: {save_path}")
        plt.show()

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

    def evaluate(self, model, test_generator, history=None, gen_test=None, etn_test=None, output_dir=None):
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
        
        # Get predictions from generator
        predicted_ages, predicted_probs = self._predict(model, test_generator)
        
        # Get age class indices from generator (one-hot encoded -> class indices)
        actual_classes = np.argmax(test_generator.labels, axis=1)
        
        # Convert class indices to ages for MAE calculation
        age_test = np.array([self._class_to_age(cls) for cls in actual_classes])
        
        # Calculate predicted classes from predicted ages
        predicted_classes = np.array([self.find_age_class_fn(age) for age in predictions])

        # Print policy-based summary (only count severe errors requested by user)
        accuracy, correct_predictions, incorrect = self._print_policy_summary(predicted_ages, age_test)
        self._print_classification_report(actual_classes, predicted_classes)
        self._print_detailed_stats(predicted_ages, age_test, actual_classes, predicted_classes)
        self._print_under18_over25_analysis(predicted_ages, age_test, actual_classes)
        # Decision policy summary and binary metrics with ROC/AUC
        self._print_decision_summary(predicted_ages)
        self._compute_and_print_binary_metrics(predicted_ages, predicted_probs, age_test, output_dir=output_dir)
        self._demographic_analysis(actual_classes, predicted_classes, gen_test, etn_test)
        self._plot_metrics(history, actual_classes, predicted_classes)

        return accuracy, correct_predictions, incorrect


# Backward-compatible functional API
def evaluate_model_performance(model, test_generator, history, AGE_CLASSES, find_age_class_fn, gen_test=None, etn_test=None):
    ev = Evaluation(AGE_CLASSES, find_age_class_fn)
    return ev.evaluate(model, test_generator, history, gen_test=gen_test, etn_test=etn_test)

