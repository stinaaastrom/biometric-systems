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

    def _predict(self, model, X_test):
        return model.predict(X_test).flatten()

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
            else:
                mae = 0.0
            print(f"\nAge Period: {class_label}")
            print(f"  True samples: {true_count}")
            print(f"  Predicted as this period: {pred_count}")
            print(f"  Correctly classified: {correct_count}")
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

    def _save_worst_images(self, X_test, predictions, age_test, *, top_n=10, output_dir=None):
        total_predictions = len(predictions)
        if total_predictions == 0:
            return 0, None
        if output_dir is None:
            root = os.path.dirname(os.path.dirname(__file__))
            output_dir = os.path.join(root, 'reports', 'evaluation_mistakes')
        os.makedirs(output_dir, exist_ok=True)

        abs_errors = np.abs(predictions - age_test)
        sorted_idx = np.argsort(-abs_errors)
        n = int(min(top_n, len(sorted_idx)))
        selected = sorted_idx[:n]

        for rank, idx in enumerate(selected, start=1):
            img = X_test[idx]
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img_to_save = (img * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    img_to_save = img.clip(0, 255).astype(np.uint8)
            else:
                img_to_save = img

            scale_factor = 4
            h, w = img_to_save.shape[:2]
            img_large = cv2.resize(img_to_save, (w * scale_factor, h * scale_factor), interpolation=cv2.INTER_LANCZOS4)

            pred_age = float(predictions[idx])
            true_age = float(age_test[idx])
            err = float(abs_errors[idx])

            overlay = img_large.copy()
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            line_height = 28
            texts = [
                (f"Pred: {pred_age:.1f}", (0, 0, 255)),
                (f"True: {true_age:.1f}", (0, 255, 0)),
                (f"Err: {err:.1f}", (255, 128, 0))
            ]
            y_offset = 10
            for i, (text, color) in enumerate(texts):
                y = y_offset + i * line_height
                (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                cv2.rectangle(overlay, (5, y - text_h - 3), (5 + text_w + 6, y + baseline + 3), (0, 0, 0), -1)
                cv2.putText(overlay, text, (8, y), font, font_scale, color, thickness, cv2.LINE_AA)

            filename = f"rank_{rank:02d}_idx_{idx}_pred_{pred_age:.1f}_true_{true_age:.1f}_err_{err:.1f}.png"
            cv2.imwrite(os.path.join(output_dir, filename), overlay)

        print(f"Saved {n} worst predictions to: {output_dir}")
        return n, output_dir

    def evaluate(self, model, X_test, age_test, history, gen_test=None, etn_test=None, *, top_n=10, output_dir=None):
        predictions = self._predict(model, X_test)
        predicted_classes = np.array([self.find_age_class_fn(age) for age in predictions])
        actual_classes = np.array([self.find_age_class_fn(age) for age in age_test])

        accuracy, correct_predictions, incorrect = self._print_summary(actual_classes, predicted_classes, len(predictions))
        self._print_classification_report(actual_classes, predicted_classes)
        self._print_detailed_stats(predictions, age_test, actual_classes, predicted_classes)
        self._print_under18_over25_analysis(predictions, age_test, actual_classes)
        self._demographic_analysis(actual_classes, predicted_classes, gen_test, etn_test)
        self._plot_metrics(history, actual_classes, predicted_classes)
        self._save_worst_images(X_test, predictions, age_test, top_n=top_n, output_dir=output_dir)

        return accuracy, correct_predictions, incorrect


# Backward-compatible functional API
def evaluate_model_performance(model, X_test, age_test, history, AGE_CLASSES, find_age_class_fn, gen_test=None, etn_test=None, *, top_n=10, output_dir=None):
    ev = Evaluation(AGE_CLASSES, find_age_class_fn)
    return ev.evaluate(model, X_test, age_test, history, gen_test=gen_test, etn_test=etn_test, top_n=top_n, output_dir=output_dir)
