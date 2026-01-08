import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import cv2
from data_generator import AGE_NORMALIZATION_FACTOR


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
    
        
        # Denormalize predictions (model was trained with normalized labels)
        predicted_ages = predicted_ages * AGE_NORMALIZATION_FACTOR
        
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
        # At this point, actual_classes and predicted_classes should already be filtered (no None values)
        correct_predictions = int(np.sum(predicted_classes == actual_classes))
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
        # At this point, data should already be filtered (no None values)
        unique_classes = np.unique(np.concatenate([actual_classes, predicted_classes]))
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
            # At this point, data should already be filtered (no None values)
            true_mask = (actual_classes == class_idx)
            true_count = int(np.sum(true_mask))
            pred_mask = (predicted_classes == class_idx)
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

    def _decision_policy(self, pred_age):
        """Decision policy based on predicted age.

        Returns: 'approved' (>25), 'denied' (<18), 'human_review' (18-25).
        """
        if pred_age > 25:
            return 'approved'
        if pred_age < 18:
            return 'denied'
        return 'human_review'

    def _print_decision_summary(self, predicted_ages):
        decisions = np.array([self._decision_policy(a) for a in predicted_ages])
        total = int(decisions.size)
        if total == 0:
            return
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

    def _compute_and_print_binary_metrics(self, predicted_ages, age_test, output_dir=None):
        """Compute FPR/FNR and ROC/AUC for adult (>=18) using regression ages.

        Also prints special FPR: minors (<18) predicted as >15.
        """
        ages_true = np.asarray(age_test)
        ages_pred = np.asarray(predicted_ages)

        y_true_adult = ages_true >= 18
        y_pred_adult = ages_pred >= 18

        minors_mask = ~y_true_adult
        adults_mask = y_true_adult
        fpr_18 = float(np.sum(minors_mask & y_pred_adult)) / float(np.sum(minors_mask)) if np.sum(minors_mask) > 0 else 0.0
        fnr_18 = float(np.sum(adults_mask & (~y_pred_adult))) / float(np.sum(adults_mask)) if np.sum(adults_mask) > 0 else 0.0

        fpr_minors_over15 = float(np.sum((ages_true < 18) & (ages_pred > 15))) / float(np.sum(ages_true < 18)) if np.sum(ages_true < 18) > 0 else 0.0

        # Use normalized predicted age as score proxy for ROC
        adult_scores = np.clip(ages_pred / 80.0, 0.0, 1.0)
        try:
            fpr_curve, tpr_curve, thresholds = roc_curve(y_true_adult.astype(int), adult_scores)
            auc_val = auc(fpr_curve, tpr_curve)
            frr_curve = 1.0 - tpr_curve  # FNMR
            gar_curve = 1.0 - frr_curve  # GAR = 1 - FRR

            # ZeroFRR: FAR when FRR==0 (or closest)
            zero_frr_mask = frr_curve <= 1e-6
            if np.any(zero_frr_mask):
                zero_frr_far = float(np.min(fpr_curve[zero_frr_mask]))
            else:
                zero_frr_far = float(fpr_curve[int(np.argmin(frr_curve))])

            # ZeroFAR: FRR when FAR==0 (or closest)
            zero_far_mask = fpr_curve <= 1e-6
            if np.any(zero_far_mask):
                zero_far_frr = float(np.min(frr_curve[zero_far_mask]))
            else:
                zero_far_frr = float(frr_curve[int(np.argmin(fpr_curve))])
        except Exception:
            fpr_curve, tpr_curve, thresholds, auc_val = None, None, None, None
            frr_curve, gar_curve = None, None
            zero_frr_far, zero_far_frr = None, None

        print("\n" + "="*50)
        print("Binary Metrics (Adult >=18)")
        print("="*50)
        print(f"FPR @18 (minors predicted adult): {fpr_18:.4f}")
        print(f"FNR @18 (adults predicted minor): {fnr_18:.4f}")
        print(f"Special FPR (minors predicted >15): {fpr_minors_over15:.4f}")
        if auc_val is not None:
            print(f"ROC AUC (adult score): {auc_val:.4f}")
            print(f"GAR (TPR) at default threshold: {float(tpr_curve[0]):.4f}")
            print(f"ZeroFRR (FAR when FRR=0): {zero_frr_far:.6f}")
            print(f"ZeroFAR (FRR when FAR=0): {zero_far_frr:.6f}")
        print("\n" + "="*50 + "\n")

        if auc_val is not None:
            self._plot_roc_curve(fpr_curve, tpr_curve, auc_val, output_dir=output_dir)
            eer, eer_threshold = self._compute_eer(fpr_curve, tpr_curve, thresholds)
            self._plot_far_frr(thresholds, fpr_curve, 1 - tpr_curve, eer, eer_threshold, output_dir=output_dir)
            print(f"EER: {eer:.4f} at threshold {eer_threshold:.4f}")
            self._plot_det_curve(fpr_curve, frr_curve, output_dir=output_dir)

    def _compute_eer(self, fpr, tpr, thresholds):
        frr = 1.0 - tpr
        diff = np.abs(fpr - frr)
        idx = int(np.argmin(diff))
        return float((fpr[idx] + frr[idx]) / 2.0), float(thresholds[idx])

    def _plot_far_frr(self, thresholds, far, frr, eer, eer_threshold, output_dir=None):
        plt.figure(figsize=(6, 5))
        plt.plot(thresholds, far, label='FAR (FPR)', color='red')
        plt.plot(thresholds, frr, label='FRR', color='blue')
        plt.axhline(eer, color='gray', linestyle='--', linewidth=1)
        plt.axvline(eer_threshold, color='gray', linestyle='--', linewidth=1)
        plt.scatter([eer_threshold], [eer], color='black', zorder=5, label=f'EER={eer:.3f}')
        plt.xlabel('Threshold (normalized score)')
        plt.ylabel('Rate')
        plt.title('FAR/FRR vs Threshold')
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()

        save_dir = output_dir or os.path.join(os.path.dirname(__file__), 'reports')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'far_frr_curve.png')
        plt.savefig(save_path, dpi=150)
        print(f"FAR/FRR curve saved to: {save_path}")
        plt.show()

    def _plot_det_curve(self, fpr_curve, frr_curve, output_dir=None):
        # Avoid zeros for log scale by clipping to a small epsilon
        eps = 1e-6
        far = np.clip(fpr_curve, eps, 1.0)
        frr = np.clip(frr_curve, eps, 1.0)

        plt.figure(figsize=(6, 5))
        plt.loglog(far, frr, marker='o', linewidth=1, markersize=3, color='purple')
        plt.xlabel('FAR (False Accept Rate)')
        plt.ylabel('FRR (False Reject Rate)')
        plt.title('DET Curve (log-log)')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()

        save_dir = output_dir or os.path.join(os.path.dirname(__file__), 'reports')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'det_curve.png')
        plt.savefig(save_path, dpi=150)
        print(f"DET curve saved to: {save_path}")
        plt.show()

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

        save_dir = output_dir or os.path.join(os.path.dirname(__file__), 'reports')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'roc_curve_adult_detection.png')
        plt.savefig(save_path, dpi=150)
        print(f"ROC curve saved to: {save_path}")
        plt.show()

    def _print_policy_summary(self, predicted_ages, age_test):
        """Policy summary where only severe errors count:
        - Minor (<18) predicted as >25
        - Adult (>25) predicted as <18

        Returns (accuracy_percent, correct_count, incorrect_count)
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

    def _misclassification_top_breakdown(self, actual_classes, predicted_classes, etn_test):
        """Report which ethnicity and age class are most common among misclassified samples.

        Requires that `etn_test` is aligned 1:1 with `actual_classes`/`predicted_classes` lengths.
        """
        if etn_test is None:
            return
        etn_arr = np.array(etn_test)
        if etn_arr.shape[0] != actual_classes.shape[0]:
            print("[WARN] Skipping misclassification top breakdown: etn_test length mismatch.")
            return

        mis_mask = (predicted_classes != actual_classes)
        if not np.any(mis_mask):
            print("No misclassifications to analyze for ethnicity/age group.")
            return

        # Filter known ethnicities
        known_mask = etn_arr != 'unknown'
        use_mask = mis_mask & known_mask
        if not np.any(use_mask):
            print("No misclassified samples with known ethnicity.")
            return

        # Top ethnicity among misclassified
        mis_etn = etn_arr[use_mask]
        etn_vals, etn_counts = np.unique(mis_etn, return_counts=True)
        top_etn_idx = int(np.argmax(etn_counts))
        top_etn = str(etn_vals[top_etn_idx])
        top_etn_count = int(etn_counts[top_etn_idx])

        # Top age class among misclassified (by true class)
        mis_true_classes = actual_classes[mis_mask]
        age_vals, age_counts = np.unique(mis_true_classes, return_counts=True)
        top_age_idx = int(age_vals[int(np.argmax(age_counts))])
        top_age_count = int(np.max(age_counts))
        min_age, max_age = self.age_classes[top_age_idx]
        top_age_label = f"{min_age}-{max_age}" if max_age else f"{min_age}+"

        # Optional: top ethnicity x age class pair
        pair_counts = {}
        for etn, cls in zip(etn_arr[mis_mask], mis_true_classes):
            if etn == 'unknown':
                continue
            pair_counts[(etn, int(cls))] = pair_counts.get((etn, int(cls)), 0) + 1
        if pair_counts:
            (pair_etn, pair_cls), pair_cnt = max(pair_counts.items(), key=lambda kv: kv[1])
            p_min, p_max = self.age_classes[pair_cls]
            pair_age_label = f"{p_min}-{p_max}" if p_max else f"{p_min}+"
        else:
            pair_etn, pair_cls, pair_cnt, pair_age_label = None, None, 0, "-"

        print("\n" + "="*50)
        print("Misclassification Top Breakdown (Ethnicity/Age)")
        print("="*50)
        print(f"Most common ethnicity among misclassified: {top_etn} ({top_etn_count})")
        print(f"Most common true age class among misclassified: {top_age_label} (class {top_age_idx}, {top_age_count})")
        if pair_etn is not None:
            print(f"Top ethnicity x age pair: {pair_etn} & {pair_age_label} ({pair_cnt})")
        print("="*50 + "\n")

    def _print_ethnicity_ranking(self, etn_arr, actual_classes, predicted_classes):
        """Print sorted ethnicity counts for overall, correctly classified, and misclassified samples."""
        etn_arr = np.array(etn_arr)
        # Overall
        vals, counts = np.unique(etn_arr, return_counts=True)
        overall = sorted(zip(vals, counts), key=lambda x: x[1], reverse=True)

        # Correct and misclassified
        correct_mask = (predicted_classes == actual_classes)
        mis_mask = ~correct_mask
        vals_c, counts_c = np.unique(etn_arr[correct_mask], return_counts=True) if np.any(correct_mask) else ([], [])
        vals_m, counts_m = np.unique(etn_arr[mis_mask], return_counts=True) if np.any(mis_mask) else ([], [])
        ranked_c = sorted(zip(vals_c, counts_c), key=lambda x: x[1], reverse=True)
        ranked_m = sorted(zip(vals_m, counts_m), key=lambda x: x[1], reverse=True)

        print("\n" + "="*50)
        print("Ethnicity Ranking (sorted by count)")
        print("="*50)
        print("Overall:")
        for etn, cnt in overall:
            print(f"  {etn}: {int(cnt)}")
        if ranked_c:
            print("Correctly classified:")
            for etn, cnt in ranked_c:
                print(f"  {etn}: {int(cnt)}")
        if ranked_m:
            print("Misclassified:")
            for etn, cnt in ranked_m:
                print(f"  {etn}: {int(cnt)}")
        print("="*50 + "\n")

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

        # Save raw confusion matrix
        save_dir = os.path.join(os.path.dirname(__file__), 'reports')
        os.makedirs(save_dir, exist_ok=True)
        cm_raw_path = os.path.join(save_dir, 'confusion_matrix.png')
        plt.savefig(cm_raw_path, dpi=150)
        print(f"Confusion matrix saved to: {cm_raw_path}")
        plt.show()

        # Also plot and save a normalized confusion matrix (per-row)
        with np.errstate(invalid='ignore', divide='ignore'):
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_norm = np.divide(cm, row_sums, where=row_sums != 0)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm_norm, annot=True, fmt='.2f', cmap='Blues', vmin=0, vmax=1,
            xticklabels=[f"{min_age}-{max_age}" if max_age else f"{min_age}+" for min_age, max_age in self.age_classes],
            yticklabels=[f"{min_age}-{max_age}" if max_age else f"{min_age}+" for min_age, max_age in self.age_classes]
        )
        plt.title('Normalized Confusion Matrix (per true class)')
        plt.xlabel('Predicted Age Class')
        plt.ylabel('Actual Age Class')
        plt.tight_layout()
        cm_norm_path = os.path.join(save_dir, 'confusion_matrix_normalized.png')
        plt.savefig(cm_norm_path, dpi=150)
        print(f"Normalized confusion matrix saved to: {cm_norm_path}")
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
        
        # Get predictions from generator (regression: direct age values)
        predictions = self._predict(model, test_generator)
        
        # Get actual age values from generator
        # Note: test_generator.labels always contains ORIGINAL (non-normalized) age values
        # Predictions are normalized (0-1 range) and get denormalized in _predict()
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

        # Apply validity mask to labels, predictions, and align demographics if provided
        age_test = age_test[valid_mask].astype(float)
        predictions = predictions[valid_mask]
        gen_arr = None
        etn_arr = None
        if gen_test is not None:
            gen_arr = np.array(gen_test)
            if gen_arr.shape[0] == valid_mask.shape[0]:
                gen_arr = gen_arr[valid_mask]
            else:
                print("[WARN] gen_test length mismatch; demographic filtering will be skipped.")
                gen_arr = None
        if etn_test is not None:
            etn_arr = np.array(etn_test)
            if etn_arr.shape[0] == valid_mask.shape[0]:
                etn_arr = etn_arr[valid_mask]
            else:
                print("[WARN] etn_test length mismatch; demographic filtering will be skipped.")
                etn_arr = None

        # Clamp predictions to a minimum valid age to avoid None classes from negatives
        predictions = np.clip(predictions, 0, 110)

        mae = np.mean(np.abs(predictions - age_test))
        median_mae = np.median(np.abs(predictions - age_test))

        print(f"Overall MAE: {mae:.2f} years")
        print(f"Median MAE:  {median_mae:.2f} years")


        # Calculate predicted and actual classes on the filtered data
        actual_classes = np.array([self.find_age_class_fn(age) for age in age_test])
        predicted_classes = np.array([self.find_age_class_fn(age) for age in predictions])

        # Drop samples that still lack valid class mapping to avoid downstream errors
        class_valid_mask = np.array([(a is not None and p is not None) 
                                      for a, p in zip(actual_classes, predicted_classes)])
        
        invalid_count = np.sum(~class_valid_mask)
        if invalid_count > 0:
            print(f"\n[WARNING] Filtering out {invalid_count} predictions with invalid age classes (out of range)")
            # Show some examples of invalid predictions
            invalid_idx = np.where(~class_valid_mask)[0][:5]
            for idx in invalid_idx:
                print(f"  Example: predicted={predictions[idx]:.1f}, actual={age_test[idx]:.1f}")
        
        if not np.any(class_valid_mask):
            raise ValueError("No valid age classes found after filtering")

        age_test = age_test[class_valid_mask]
        predictions = predictions[class_valid_mask]
        actual_classes = actual_classes[class_valid_mask].astype(int)
        predicted_classes = predicted_classes[class_valid_mask].astype(int)
        if gen_arr is not None and gen_arr.shape[0] == class_valid_mask.shape[0]:
            gen_arr = gen_arr[class_valid_mask]
        if etn_arr is not None and etn_arr.shape[0] == class_valid_mask.shape[0]:
            etn_arr = etn_arr[class_valid_mask]

        # Optional: filter out unknown demographics entirely if provided
        if (gen_arr is not None) and (etn_arr is not None):
            demo_known_mask = (gen_arr != 'unknown') & (etn_arr != 'unknown')
            removed_demo = int(np.sum(~demo_known_mask))
            if removed_demo > 0:
                print(f"[INFO] Filtering out {removed_demo} samples with unknown gender/ethnicity")
            age_test = age_test[demo_known_mask]
            predictions = predictions[demo_known_mask]
            actual_classes = actual_classes[demo_known_mask]
            predicted_classes = predicted_classes[demo_known_mask]
            gen_arr = gen_arr[demo_known_mask]
            etn_arr = etn_arr[demo_known_mask]

        # Overall age-class accuracy summary
        self._print_summary(actual_classes, predicted_classes, len(predictions))

        # Policy-based summary (severe errors only)
        accuracy, correct_predictions, incorrect = self._print_policy_summary(predictions, age_test)
        # Decision summary and binary metrics
        self._print_decision_summary(predictions)
        self._compute_and_print_binary_metrics(predictions, age_test, output_dir=output_dir)
        self._print_classification_report(actual_classes, predicted_classes)
        self._print_detailed_stats(predictions, age_test, actual_classes, predicted_classes)
        self._print_under18_over25_analysis(predictions, age_test, actual_classes)
        # Use aligned, filtered demographics when available
        self._demographic_analysis(actual_classes, predicted_classes, gen_arr, etn_arr)
        # Ethnicity ranking (overall, correct, misclassified)
        if etn_arr is not None:
            self._print_ethnicity_ranking(etn_arr, actual_classes, predicted_classes)
        # New: top ethnicity and age class among misclassified
        try:
            self._misclassification_top_breakdown(actual_classes, predicted_classes, etn_arr)
        except Exception as e:
            print(f"[WARN] Could not compute misclassification top breakdown: {e}")
        self._plot_metrics(history, actual_classes, predicted_classes)

        return accuracy, correct_predictions, incorrect


# Backward-compatible functional API
def evaluate_model_performance(model, test_generator, history, AGE_CLASSES, find_age_class_fn, gen_test=None, etn_test=None):
    ev = Evaluation(AGE_CLASSES, find_age_class_fn)
    return ev.evaluate(model, test_generator, history, gen_test=gen_test, etn_test=etn_test)

