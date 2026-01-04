import numpy as np
import cv2
import keras
from image_processing import ImageProcesser

# Age normalization constant for regression training
AGE_NORMALIZATION_FACTOR = 100.0

class ImageDiskGenerator(keras.utils.Sequence):
    def __init__(self, image_paths, labels, batch_size=32, augment=False, shuffle=True):
        self.image_paths = np.array(image_paths)
        self.labels = np.array(labels, dtype=np.float32)
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.image_processor = ImageProcesser()
        self.aug_transform = self.image_processor.get_augmentation_transform() if augment else None
        self.on_epoch_end()
        
        # Debug: Print label statistics on initialization
        print(f"\n[ImageDiskGenerator] Initialized with {len(self.labels)} samples")
        print(f"  Original label range: {np.min(self.labels):.2f} - {np.max(self.labels):.2f}")
        print(f"  Labels will be normalized by dividing by {AGE_NORMALIZATION_FACTOR} during training")
        normalized = self.labels / AGE_NORMALIZATION_FACTOR
        print(f"  Normalized label range: {np.min(normalized):.4f} - {np.max(normalized):.4f}")

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_paths = self.image_paths[batch_idx]
        batch_labels = self.labels[batch_idx].copy()
        
        images = []
        for img_path in batch_paths:
            img = cv2.imread(img_path)
            if img is not None:
                if self.aug_transform is not None:
                    try:
                        img = self.aug_transform(image=img)['image']
                    except Exception:
                        pass
                images.append(img)
        
        images = np.array(images, dtype=np.float32)
        
        # Always normalize labels (ages divided by 100 for better training convergence)
        batch_labels = batch_labels / AGE_NORMALIZATION_FACTOR
        
        return images, batch_labels

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)
