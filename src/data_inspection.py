"""
Data Inspection Tool: Visual inspection of weak age classes
Shows 50 random images per weak class with actual age, prediction, and confidence
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from dataset import DatasetDownloader, IMAGE_SIZE
from cnn_model import CNNModel, AGE_CLASSES

def inspect_weak_classes(num_samples=50):
    """
    Load model and test data, show weak classes with predictions
    """
    print("\n" + "="*60)
    print("DATA INSPECTION: Weak Age Classes")
    print("="*60)
    
    # Load dataset
    print("\nLoading dataset...")
    downloader = DatasetDownloader()
    X_train, X_test, age_train, age_test, gen_train, gen_test, etn_train, etn_test = downloader.generate_dataset()
    
    # Load model
    print("Loading trained model...")
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'age_model.keras')
    cnn = CNNModel(X_train, age_train, X_test, age_test)
    cnn.load_model(model_path)
    
    # Get predictions
    print("Getting predictions...")
    pred_probs, pred_reg = cnn.model.predict(X_test, verbose=0)
    pred_classes = np.argmax(pred_probs, axis=1)
    pred_confidence = np.max(pred_probs, axis=1)
    
    # Convert test ages to classes
    test_classes = np.array([cnn.find_age_class(age) for age in age_test])
    
    # Weak classes
    weak_classes = [2, 3, 5]  # 18-25, 26-35, 46-60
    weak_names = ["18-25", "26-35", "46-60"]
    
    for weak_idx, weak_name in zip(weak_classes, weak_names):
        print(f"\n{'-'*60}")
        print(f"CLASS {weak_idx}: {weak_name} ({num_samples} random samples)")
        print(f"{'-'*60}")
        
        # Find indices in this class
        mask = test_classes == weak_idx
        indices = np.where(mask)[0]
        
        if len(indices) == 0:
            print(f"No samples in class {weak_name}")
            continue
        
        # Sample randomly
        sampled_idx = np.random.choice(indices, min(num_samples, len(indices)), replace=False)
        
        # Create grid
        cols = 10
        rows = (len(sampled_idx) + cols - 1) // cols
        fig = plt.figure(figsize=(20, 4*rows))
        
        for plot_idx, sample_idx in enumerate(sampled_idx, 1):
            ax = fig.add_subplot(rows, cols, plot_idx)
            
            # Get image and data
            img = X_test[sample_idx]
            actual_age = age_test[sample_idx]
            actual_class = test_classes[sample_idx]
            pred_class = pred_classes[sample_idx]
            confidence = pred_confidence[sample_idx]
            reg_age = float(pred_reg[sample_idx])
            
            # Display
            ax.imshow(img)
            
            # Color: green if correct, red if wrong
            color = 'green' if actual_class == pred_class else 'red'
            
            min_age, max_age = AGE_CLASSES[actual_class]
            actual_label = f"{min_age}+" if max_age is None else f"{min_age}-{max_age}"
            
            pred_min, pred_max = AGE_CLASSES[pred_class]
            pred_label = f"{pred_min}+" if pred_max is None else f"{pred_min}-{pred_max}"
            
            title = f"Age: {actual_age:.0f}\n{actual_label} → {pred_label}\nConf: {confidence:.2f}\nReg: {reg_age:.1f}"
            ax.set_title(title, fontsize=8, color=color, weight='bold')
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'inspection_{weak_name}.png', dpi=100, bbox_inches='tight')
        print(f"✓ Saved: inspection_{weak_name}.png")
        plt.close()
        
        # Print statistics
        print(f"\nStatistics for {weak_name}:")
        correct = np.sum(pred_classes[sampled_idx] == actual_class)
        print(f"  Accuracy: {correct}/{len(sampled_idx)} ({100*correct/len(sampled_idx):.1f}%)")
        print(f"  Avg confidence: {pred_confidence[sampled_idx].mean():.3f}")
        print(f"  Avg regression error: {np.abs(pred_reg[sampled_idx].flatten() - age_test[sampled_idx]).mean():.1f} years")
        
        # Check for obvious issues
        print(f"\n  Issues found:")
        for sample_idx in sampled_idx:
            actual_age = age_test[sample_idx]
            actual_class = test_classes[sample_idx]
            pred_class = pred_classes[sample_idx]
            confidence = pred_confidence[sample_idx]
            
            # Off by 2+ classes
            if abs(actual_class - pred_class) >= 2:
                pred_min, pred_max = AGE_CLASSES[pred_class]
                pred_label = f"{pred_min}+" if pred_max is None else f"{pred_min}-{pred_max}"
                print(f"    - Age {actual_age:.0f}: predicted {pred_label} (conf {confidence:.2f}) - BIG ERROR")
            
            # Low confidence on correct class
            if actual_class == pred_class and confidence < 0.5:
                print(f"    - Age {actual_age:.0f}: correct but LOW CONFIDENCE ({confidence:.2f})")

if __name__ == "__main__":
    inspect_weak_classes(num_samples=50)
    print("\n✓ Inspection complete! Check inspection_*.png files in workspace root.")
