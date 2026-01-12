"""
Fine-tune existing age model on IMDB-WIKI dataset for better generalization to selfies/webcam
"""

import os
import numpy as np
import cv2
from scipy.io import loadmat
from datetime import datetime
from pathlib import Path
import tensorflow as tf

from src.cnn_model import CNNModel

# Image size for model input
IMAGE_SIZE = (224, 224)

# IMDB-WIKI dataset path
WIKI_PATH = r"C:\Users\Robin Rienks Hestad\Downloads\wiki_crop\wiki_crop"
WIKI_MAT = os.path.join(WIKI_PATH, "wiki.mat")

def load_imdb_wiki_data(wiki_path, max_samples=None, min_age=0, max_age=100):
    """
    Load IMDB-WIKI dataset with age labels
    
    Returns:
        images: List of preprocessed image arrays
        ages: List of age labels
    """
    print(f"Loading IMDB-WIKI metadata from {WIKI_MAT}...")
    
    # Load metadata
    mat = loadmat(WIKI_MAT)
    
    # Extract relevant fields
    dob = mat['wiki'][0][0][0][0]  # Date of birth (Matlab ordinal)
    photo_taken = mat['wiki'][0][0][1][0]  # Year photo was taken
    full_path = mat['wiki'][0][0][2][0]  # Full path to image
    gender = mat['wiki'][0][0][3][0]  # Gender (optional)
    face_score = mat['wiki'][0][0][6][0]  # Face detection score
    second_face_score = mat['wiki'][0][0][7][0]  # Second face score (higher = multiple faces)
    
    images = []
    ages = []
    
    print(f"Processing {len(photo_taken)} total entries...")
    
    for i in range(len(photo_taken)):
        # Calculate age
        try:
            # Convert Matlab ordinal to datetime
            dob_year = datetime.fromordinal(max(int(dob[i]) - 366, 1)).year
            age = photo_taken[i] - dob_year
        except:
            continue
        
        # Filter by age range
        if age < min_age or age > max_age:
            continue
        
        # Filter by face score (remove poor detections)
        if face_score[i] < 1.0:  # Low quality face
            continue
        
        # Filter out images with multiple faces
        if second_face_score[i] > 0.0:
            continue
        
        # Get image path
        img_path = os.path.join(wiki_path, full_path[i][0])
        
        if not os.path.exists(img_path):
            continue
        
        # Load and preprocess image
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Resize to model input size
        img = cv2.resize(img, IMAGE_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        
        images.append(img)
        ages.append(age)
        
        if max_samples and len(images) >= max_samples:
            break
        
        if len(images) % 1000 == 0:
            print(f"Loaded {len(images)} valid images...")
    
    print(f"\nLoaded {len(images)} images total")
    print(f"Age range: {min(ages)}-{max(ages)}")
    print(f"Mean age: {np.mean(ages):.1f}")
    
    return np.array(images), np.array(ages)


def prepare_labels(ages):
    """
    Convert ages to both classification and regression labels
    Matches the 8-class structure from original model
    """
    # Define same age classes as original model
    age_classes = [
        (0, 2), (4, 6), (8, 13), (15, 20),
        (25, 32), (38, 43), (48, 53), (60, 100)
    ]
    
    class_labels = []
    for age in ages:
        # Find which class this age belongs to
        for idx, (min_age, max_age) in enumerate(age_classes):
            if min_age <= age <= max_age:
                class_labels.append(idx)
                break
        else:
            # Age doesn't fit in any class - assign to nearest
            if age < age_classes[0][0]:
                class_labels.append(0)
            else:
                class_labels.append(len(age_classes) - 1)
    
    # Convert to one-hot
    class_labels_onehot = tf.keras.utils.to_categorical(class_labels, num_classes=8)
    
    # Regression labels (normalized)
    reg_labels = ages.astype(np.float32) / 100.0
    
    return class_labels_onehot, reg_labels


def finetune_model(model_path, wiki_path, epochs=10, batch_size=32, learning_rate=1e-5):
    """
    Fine-tune existing model on IMDB-WIKI dataset
    """
    print("="*60)
    print("FINE-TUNING AGE MODEL ON IMDB-WIKI DATASET")
    print("="*60)
    
    # Load existing model
    print("\n1. Loading pre-trained model...")
    # Create dummy CNNModel instance (we only need the model loading functionality)
    dummy_data = np.zeros((1, 224, 224, 3))
    dummy_ages = np.array([0])
    cnn_model = CNNModel(dummy_data, dummy_ages, dummy_data, dummy_ages)
    cnn_model.load_model(model_path)
    
    # Load IMDB-WIKI data
    print("\n2. Loading IMDB-WIKI dataset...")
    images, ages = load_imdb_wiki_data(
        wiki_path, 
        max_samples=20000,  # Use 20k images for fine-tuning
        min_age=0,
        max_age=100
    )
    
    # Prepare labels
    print("\n3. Preparing labels...")
    class_labels, reg_labels = prepare_labels(ages)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_class_train, y_class_val, y_reg_train, y_reg_val = train_test_split(
        images, class_labels, reg_labels, test_size=0.15, random_state=42
    )
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Freeze early layers for transfer learning
    print("\n4. Freezing early layers...")
    for layer in cnn_model.model.layers:
        if 'efficientnet' in layer.name.lower():
            # Freeze base EfficientNet layers
            layer.trainable = False
        else:
            # Keep classification/regression heads trainable
            layer.trainable = True
    
    print(f"Trainable layers: {sum([1 for l in cnn_model.model.layers if l.trainable])}")
    print(f"Frozen layers: {sum([1 for l in cnn_model.model.layers if not l.trainable])}")
    
    # Compile with lower learning rate for fine-tuning
    print("\n5. Compiling model...")
    cnn_model.model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            'age_output': 'categorical_crossentropy',
            'age_reg': 'mse'
        },
        loss_weights={'age_output': 1.0, 'age_reg': 0.3},
        metrics={
            'age_output': 'accuracy',
            'age_reg': 'mae'
        }
    )
    
    # Training callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-7
        )
    ]
    
    # Fine-tune
    print(f"\n6. Fine-tuning for {epochs} epochs...")
    print("="*60)
    
    history = cnn_model.model.fit(
        X_train,
        {'age_output': y_class_train, 'age_reg': y_reg_train},
        validation_data=(X_val, {'age_output': y_class_val, 'age_reg': y_reg_val}),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )
    
    # Save fine-tuned model
    print("\n7. Saving fine-tuned model...")
    finetuned_path = model_path.replace('.keras', '_finetuned.keras')
    finetuned_npz = model_path.replace('.keras', '_finetuned.npz')
    
    cnn_model.model.save(finetuned_path)
    
    # Save metadata with temperature
    np.savez(
        finetuned_npz,
        temperature=cnn_model.temperature,
        training_history={
            'loss': history.history.get('loss', []),
            'val_loss': history.history.get('val_loss', []),
            'age_output_accuracy': history.history.get('age_output_accuracy', []),
            'val_age_output_accuracy': history.history.get('val_age_output_accuracy', [])
        }
    )
    
    print(f"\nFine-tuned model saved to:")
    print(f"  {finetuned_path}")
    print(f"  {finetuned_npz}")
    
    # Summary
    print("\n" + "="*60)
    print("FINE-TUNING COMPLETE")
    print("="*60)
    print(f"Final validation accuracy: {history.history['val_age_output_accuracy'][-1]*100:.2f}%")
    print(f"Final validation MAE: {history.history['val_age_reg_mae'][-1]*100:.2f} years")
    print("\nNext steps:")
    print("1. Test fine-tuned model on webcam: python test_webcam.py")
    print("2. Update main.py to use fine-tuned model")
    print("3. Verify predictions on real-world selfies")
    
    return cnn_model


if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "models/age_model.keras"
    
    # Run fine-tuning
    finetune_model(
        model_path=MODEL_PATH,
        wiki_path=WIKI_PATH,
        epochs=10,
        batch_size=32,
        learning_rate=1e-5
    )
