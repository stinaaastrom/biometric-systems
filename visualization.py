import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import cv2

from constants import AGE_CLASSES

class DatasetVisualizer:
    """Visualize dataset distributions for age, gender, and ethnicity"""
    
    def __init__(self, ages, genders, ethnicities):
        """
        Initialize visualizer with dataset arrays
        
        Args:
            ages: numpy array or list of ages
            genders: list of gender strings (optional)
            ethnicities: list of ethnicity strings (optional)
        """
        self.ages = ages
        self.genders = [g for g in genders if genders is not None and g.lower() != 'unknown'] if genders is not None else []
        self.ethnicities = [e for e in ethnicities if ethnicities is not None and e.lower() != 'unknown'] if ethnicities is not None else []
        
    def _get_age_class_label(self, age):
        """Map age to age class label"""
        for idx, (min_age, max_age) in enumerate(AGE_CLASSES):
            if max_age is None:
                if age >= min_age:
                    return f"{min_age}+"
            elif min_age <= age <= max_age:
                return f"{min_age}-{max_age}"
        return None
    
    def plot_age_distribution(self):
        """Plot histogram of age distribution across age classes"""
        # Map ages to class labels
        age_class_labels = [self._get_age_class_label(age) for age in self.ages]
        age_class_labels = [label for label in age_class_labels if label is not None]
        
        # Count occurrences
        age_counts = Counter(age_class_labels)
        
        # Sort by age class order
        class_labels = []
        counts = []
        for min_age, max_age in AGE_CLASSES:
            label = f"{min_age}+" if max_age is None else f"{min_age}-{max_age}"
            if label in age_counts:
                class_labels.append(label)
                counts.append(age_counts[label])
        
        # Plot
        plt.figure(figsize=(10, 6))
        bars = plt.bar(class_labels, counts, color='steelblue', edgecolor='black')
        plt.xlabel('Age Class', fontsize=12)
        plt.ylabel('Number of Images', fontsize=12)
        plt.title('Age Distribution Across Age Classes', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.show()
        
        return age_counts
    
    def plot_gender_distribution(self):
        """Plot pie chart of gender distribution"""
        gender_counts = Counter(self.genders)
        
        plt.figure(figsize=(8, 8))
        colors = ['#ff9999', '#66b3ff']
        plt.pie(gender_counts.values(), 
                labels=gender_counts.keys(), 
                autopct='%1.1f%%',
                colors=colors,
                startangle=90,
                textprops={'fontsize': 12})
        plt.title('Gender Distribution', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        return gender_counts
    
    def plot_ethnicity_distribution(self):
        """Plot bar chart of ethnicity distribution"""
        ethnicity_counts = Counter(self.ethnicities)
        
        # Sort by count
        sorted_ethnicities = sorted(ethnicity_counts.items(), key=lambda x: x[1], reverse=True)
        labels = [item[0] for item in sorted_ethnicities]
        counts = [item[1] for item in sorted_ethnicities]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(labels, counts, color='coral', edgecolor='black')
        plt.xlabel('Ethnicity', fontsize=12)
        plt.ylabel('Number of Images', fontsize=12)
        plt.title('Ethnicity Distribution', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.show()
        
        return ethnicity_counts
    
    def plot_all_distributions(self):
        """Plot all distributions in a single figure"""
        fig = plt.figure(figsize=(15, 5))
        
        # Age distribution
        plt.subplot(1, 3, 1)
        age_class_labels = [self._get_age_class_label(age) for age in self.ages]
        age_class_labels = [label for label in age_class_labels if label is not None]
        age_counts = Counter(age_class_labels)
        
        class_labels = []
        counts = []
        for min_age, max_age in AGE_CLASSES:
            label = f"{min_age}+" if max_age is None else f"{min_age}-{max_age}"
            if label in age_counts:
                class_labels.append(label)
                counts.append(age_counts[label])
        
        plt.bar(class_labels, counts, color='steelblue', edgecolor='black')
        plt.xlabel('Age Class')
        plt.ylabel('Count')
        plt.title('Age Distribution')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Gender distribution
        plt.subplot(1, 3, 2)
        gender_counts = Counter(self.genders)
        colors = ['#ff9999', '#66b3ff']
        plt.pie(gender_counts.values(), 
                labels=gender_counts.keys(), 
                autopct='%1.1f%%',
                colors=colors,
                startangle=90)
        plt.title('Gender Distribution')
        
        # Ethnicity distribution
        plt.subplot(1, 3, 3)
        ethnicity_counts = Counter(self.ethnicities)
        sorted_ethnicities = sorted(ethnicity_counts.items(), key=lambda x: x[1], reverse=True)
        labels = [item[0] for item in sorted_ethnicities]
        counts = [item[1] for item in sorted_ethnicities]
        
        plt.bar(labels, counts, color='coral', edgecolor='black')
        plt.xlabel('Ethnicity')
        plt.ylabel('Count')
        plt.title('Ethnicity Distribution')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def print_statistics(self):
        """Print dataset statistics"""
        print("\n" + "="*60)
        print("DATASET STATISTICS")
        print("="*60)
        
        print(f"\nTotal images: {len(self.ages)}")
        print(f"Age range: {min(self.ages):.0f} - {max(self.ages):.0f} years")
        print(f"Mean age: {np.mean(self.ages):.1f} years")
        print(f"Median age: {np.median(self.ages):.1f} years")
        
        print("\nAge class distribution:")
        age_class_labels = [self._get_age_class_label(age) for age in self.ages]
        age_class_labels = [label for label in age_class_labels if label is not None]
        age_counts = Counter(age_class_labels)
        for min_age, max_age in AGE_CLASSES:
            label = f"{min_age}+" if max_age is None else f"{min_age}-{max_age}"
            count = age_counts.get(label, 0)
            percentage = (count / len(age_class_labels)) * 100 if age_class_labels else 0
            print(f"  {label:10} : {count:6} ({percentage:5.1f}%)")
        
        print("\nGender distribution:")
        gender_counts = Counter(self.genders)
        for gender, count in gender_counts.items():
            percentage = (count / len(self.genders)) * 100
            print(f"  {gender:10} : {count:6} ({percentage:5.1f}%)")
        
        print("\nEthnicity distribution:")
        ethnicity_counts = Counter(self.ethnicities)
        sorted_ethnicities = sorted(ethnicity_counts.items(), key=lambda x: x[1], reverse=True)
        for ethnicity, count in sorted_ethnicities:
            percentage = (count / len(self.ethnicities)) * 100
            print(f"  {ethnicity:10} : {count:6} ({percentage:5.1f}%)")
        
        print("="*60 + "\n")

    def show_initial_visuals(self):
        self.plot_all_distributions()
        self.print_statistics()
    
    @staticmethod
    def display_prediction(img, predicted_age, actual_age=None):
        """
        Display an image with predicted age and optional actual age
        """
        
        # Convert to age class
        age_class_idx = None
        for idx, (min_age, max_age) in enumerate(AGE_CLASSES):
            if max_age is None:
                if predicted_age >= min_age:
                    age_class_idx = idx
                    break
            elif min_age <= predicted_age <= max_age:
                age_class_idx = idx
                break
        
        # Get age class label
        if age_class_idx is not None:
            min_age, max_age = AGE_CLASSES[age_class_idx]
            age_class_label = f"{min_age}+" if max_age is None else f"{min_age}-{max_age}"
        else:
            age_class_label = "Unknown"
        
        # Handle normalized vs unnormalized images
        if img.max() <= 1.0:
            display_img = (img * 255).astype(np.uint8)
        else:
            display_img = img.astype(np.uint8)
        
        # Convert BGR to RGB for matplotlib
        if len(display_img.shape) == 3 and display_img.shape[2] == 3:
            display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        
        # Create figure
        plt.figure(figsize=(8, 8))
        plt.imshow(display_img)
        plt.axis('off')
        
        # Create title
        if actual_age is not None:
            # Get actual age class
            actual_class_idx = None
            for idx, (min_age, max_age) in enumerate(AGE_CLASSES):
                if max_age is None:
                    if actual_age >= min_age:
                        actual_class_idx = idx
                        break
                elif min_age <= actual_age <= max_age:
                    actual_class_idx = idx
                    break
            
            if actual_class_idx is not None:
                min_age, max_age = AGE_CLASSES[actual_class_idx]
                actual_class_label = f"{min_age}+" if max_age is None else f"{min_age}-{max_age}"
            else:
                actual_class_label = "Unknown"
            
            title = f"Predicted: {predicted_age:.1f} years (Class: {age_class_label})\n"
            title += f"Actual: {actual_age:.1f} years (Class: {actual_class_label})"
            
            # Color code: green if correct class, red if wrong
            color = 'green' if age_class_idx == actual_class_idx else 'red'
        else:
            title = f"Predicted Age: {predicted_age:.1f} years\nAge Class: {age_class_label}"
            color = 'black'
        
        plt.title(title, fontsize=14, fontweight='bold', color=color, pad=20)
        plt.tight_layout()
        plt.show()