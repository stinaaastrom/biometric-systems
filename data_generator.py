import numpy as np
import cv2
import keras
from image_processing import ImageProcesser

class ImageDiskGenerator(keras.utils.Sequence):
    def __init__(self, image_paths, labels, batch_size=32, augment=False, shuffle=True):
        self.image_paths = np.array(image_paths)
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.image_processor = ImageProcesser()
        self.aug_transform = self.image_processor.get_augmentation_transform() if augment else None
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_paths = self.image_paths[batch_idx]
        batch_labels = self.labels[batch_idx]
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
        return images, batch_labels

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)
