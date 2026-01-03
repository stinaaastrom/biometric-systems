import os
import hashlib
import cv2
import numpy as np
import json
import random
import shutil
from collections import defaultdict
from sklearn.model_selection import train_test_split

from image_processing import ImageProcesser
from constants import AGE_CLASSES, IMAGE_SIZE
from data_generator import ImageDiskGenerator


# Define data folder paths (defaults)
DATA_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
UTKFACE_PATH = os.path.join(DATA_FOLDER, 'utkface')
FACIAL_AGE_PATH = os.path.join(DATA_FOLDER, 'facial_age')
PROCESSED_FACES_PATH = os.path.join(DATA_FOLDER, 'processed_faces')



class DatasetProcessor:
    """Preprocessing, face crop, metadata scanning and generator building."""

    def preprocess_and_save(self, output_dir='data/preprocessed_faces'):
        """
        Preprocess (enhancements, normalization, resize) and save all images to output_dir.
        """
        os.makedirs(output_dir, exist_ok=True)
        self.ensure_preprocessed_faces()
        self.create_image_age_list(use_processed=True)
        meta = []
        for img_path, age, gender, etnicity in zip(self.image_paths, self.ages, self.genders, self.etnicity):
            fname = os.path.basename(img_path)
            dest_path = os.path.join(output_dir, fname)
            if not os.path.exists(dest_path):
                img = cv2.imread(img_path)
                if img is not None:
                    img = self.image_processor.image_enhancements(img, target_size=None)
                    img = cv2.resize(img, IMAGE_SIZE)
                    img = self.image_processor.normalize_image(img)
                    img = (img * 255).astype(np.uint8)
                    cv2.imwrite(dest_path, img)
            meta.append({'filename': fname, 'age': float(age), 'gender': str(gender), 'etnicity': str(etnicity)})
        # Save metadata
        with open(os.path.join(output_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"✓ Preprocessed {len(meta)} images to {output_dir}")

    def __init__(self, dataset_utkface: str = UTKFACE_PATH, dataset_facial_age: str = FACIAL_AGE_PATH,
                 processed_root: str = PROCESSED_FACES_PATH):
        self.dataset_utkface = dataset_utkface
        self.dataset_facial_age = dataset_facial_age
        self.processed_root = processed_root
        self.processed_utkface = os.path.join(processed_root, 'utkface')
        self.processed_facial_age = os.path.join(processed_root, 'facial_age')
        self.image_processor = ImageProcesser()
        self.genders = []
        self.etnicity = []
        self.ages = []

    # ---------- Preprocessing (one-time face detection) ----------
    def _has_processed_faces(self, root_dir):
        for _, _, files in os.walk(root_dir):
            if any(f.lower().endswith((".jpg", ".jpeg", ".png")) for f in files):
                return True
        return False

    def ensure_preprocessed_faces(self, overwrite=False):
        os.makedirs(self.processed_root, exist_ok=True)
        os.makedirs(self.processed_utkface, exist_ok=True)
        os.makedirs(self.processed_facial_age, exist_ok=True)

        needs_utk = overwrite or not self._has_processed_faces(self.processed_utkface)
        needs_facial = overwrite or not self._has_processed_faces(self.processed_facial_age)

        if not needs_utk and not needs_facial:
            print("✓ Processed faces already exist. Skipping preprocessing.")
            return self.processed_root

        if needs_utk:
            self._process_utkface()
        if needs_facial:
            self._process_facial_age()

        print("✓ Preprocessing complete. Cropped faces stored in:", self.processed_root)
        return self.processed_root

    def _process_utkface(self):
        print(f"\nPreprocessing UTKFace from {self.dataset_utkface}")
        processed_count, skipped_count = 0, 0
        for root, _, files in os.walk(self.dataset_utkface):
            for img_name in files:
                if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                try:
                    parts = img_name.split('_')
                    int(parts[0]); int(parts[1]); int(parts[2])
                except Exception:
                    skipped_count += 1
                    continue

                img_path = os.path.join(root, img_name)
                frame = cv2.imread(img_path)
                if frame is None:
                    skipped_count += 1
                    continue

                #face_img = self.image_processor.detect_crop_faces(frame)
                face_img = frame
                if face_img is None:
                    skipped_count += 1
                    continue

                face_img = self.image_processor.image_enhancements(face_img)
                face_img = cv2.resize(face_img, IMAGE_SIZE)

                # Ensure unique filename to avoid overwrites when same img_name exists in different folders
                name_no_ext, ext = os.path.splitext(img_name)
                h = hashlib.md5(os.path.join(root, img_name).encode('utf-8')).hexdigest()[:8]
                unique_name = f"{name_no_ext}.{h}{ext}"
                save_path = os.path.join(self.processed_utkface, unique_name)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(save_path, face_img)
                processed_count += 1

                if processed_count % 200 == 0:
                    print(f"Processed {processed_count} UTKFace images", end='\r')

        print(f"Processed UTKFace: {processed_count} saved, {skipped_count} skipped")

    def _process_facial_age(self):
        print(f"\nPreprocessing Facial Age from {self.dataset_facial_age}")
        processed_count, skipped_count = 0, 0
        for root, _, files in os.walk(self.dataset_facial_age):
            age_folder = os.path.basename(root)
            try:
                age_value = int(age_folder)
            except ValueError:
                continue

            for img_name in files:
                if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                img_path = os.path.join(root, img_name)
                frame = cv2.imread(img_path)
                if frame is None:
                    skipped_count += 1
                    continue

                #face_img = self.image_processor.detect_crop_faces(frame)
                face_img = frame
                if face_img is None:
                    skipped_count += 1
                    continue

                face_img = self.image_processor.image_enhancements(face_img)
                face_img = cv2.resize(face_img, IMAGE_SIZE)

                dest_dir = os.path.join(self.processed_facial_age, str(age_value))
                os.makedirs(dest_dir, exist_ok=True)
                # Ensure unique filename inside age folder
                name_no_ext, ext = os.path.splitext(img_name)
                h = hashlib.md5(os.path.join(root, img_name).encode('utf-8')).hexdigest()[:8]
                unique_name = f"{name_no_ext}.{h}{ext}"
                save_path = os.path.join(dest_dir, unique_name)
                cv2.imwrite(save_path, face_img)
                processed_count += 1

                if processed_count % 200 == 0:
                    print(f"Processed {processed_count} Facial Age images", end='\r')

        print(f"Processed Facial Age: {processed_count} saved, {skipped_count} skipped")

    def create_image_age_list(self, use_processed=True):
        """
        Build metadata list (paths only) from preprocessed faces.
        If use_processed is False, it will scan raw datasets (not recommended for training now).
        """

        # Collect file paths and metadata (NOT images)
        self.image_paths = []
        self.ages = []
        self.genders = []
        self.etnicity = []

        utk_root = self.processed_utkface if use_processed else self.dataset_utkface
        facial_root = self.processed_facial_age if use_processed else self.dataset_facial_age

        if use_processed and not (self._has_processed_faces(utk_root) or self._has_processed_faces(facial_root)):
            raise RuntimeError("Processed faces not found. Run ensure_preprocessed_faces() first.")

        # Load UTKFace dataset
        print(f"\nScanning UTKFace dataset from: {utk_root}")
        utkface_files = []
        for root, _, files in os.walk(utk_root):
            for img_name in files:
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    utkface_files.append((root, img_name))

        random.shuffle(utkface_files)

        for root, img_name in utkface_files:
            try:
                age, gender, etnicity = img_name.split('_')[:3]
                age, gender, etnicity = int(age), int(gender), int(etnicity)

                gender_str = "Male" if gender == 0 else "Female"
                etnicity_map = {0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Others"}
                etnicity_str = etnicity_map.get(etnicity, "unknown")

                img_path = os.path.join(root, img_name)

                self.image_paths.append(img_path)
                self.ages.append(age)
                self.genders.append(gender_str)
                self.etnicity.append(etnicity_str)

                if len(self.image_paths) % 200 == 0:
                    print(f"Scanned {len(self.image_paths)} UTKFace paths", end='\r')
            except Exception:
                continue

        print(f"\nFound {len(self.image_paths)} UTKFace paths")

        # Load Facial Age dataset
        print(f"\nScanning Facial Age dataset from: {facial_root}")
        facial_age_files = []
        for root, _, files in os.walk(facial_root):
            age_folder = os.path.basename(root)
            try:
                age = int(age_folder)
                for img_name in files:
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        facial_age_files.append((root, img_name, age))
            except ValueError:
                continue

        random.shuffle(facial_age_files)

        for root, img_name, age in facial_age_files:
            try:
                img_path = os.path.join(root, img_name)

                self.image_paths.append(img_path)
                self.ages.append(age)
                self.genders.append('unknown')
                self.etnicity.append('unknown')

                if len(self.image_paths) % 200 == 0:
                    print(f"Scanned {len(self.image_paths)} total paths", end='\r')
            except Exception:
                continue

        print(f"\nTotal scanned: {len(self.image_paths)} image paths (faces already cropped)")

        # Convert to numpy arrays for indexing
        self.image_paths = np.array(self.image_paths)
        self.ages = np.array(self.ages, dtype=np.float32)
        self.genders = np.array(self.genders)
        self.etnicity = np.array(self.etnicity)

    def _balance_indices(self, indices):
        age_buckets = defaultdict(list)
        for idx in indices:
            age = self.ages[idx]
            for class_idx, (min_age, max_age) in enumerate(AGE_CLASSES):
                if max_age is None:
                    if age >= min_age:
                        age_buckets[class_idx].append(idx)
                        break
                elif min_age <= age <= max_age:
                    age_buckets[class_idx].append(idx)
                    break
        print("\nAge class distribution (before balancing):")
        for class_idx, (min_age, max_age) in enumerate(AGE_CLASSES):
            label = f"{min_age}+" if max_age is None else f"{min_age}-{max_age}"
            count = len(age_buckets[class_idx])
            print(f"  {label}: {count} images")
        class_counts = [len(age_buckets[c]) for c in range(len(AGE_CLASSES))]
        min_count = min(class_counts) if class_counts else 0
        balanced_indices = []
        print(f"\nBalancing to {min_count} images per class...")
        for class_idx in range(len(AGE_CLASSES)):
            bucket_indices = age_buckets[class_idx]
            if len(bucket_indices) > min_count:
                sampled = random.sample(bucket_indices, min_count)
            else:
                sampled = bucket_indices
            balanced_indices.extend(sampled)
            min_age, max_age = AGE_CLASSES[class_idx]
            label = f"{min_age}+" if max_age is None else f"{min_age}-{max_age}"
            print(f"  {label}: {len(sampled)} images")
        return np.array(balanced_indices)

    @staticmethod
    def _age_to_class(age):
        for idx, (min_age, max_age) in enumerate(AGE_CLASSES):
            if max_age is None:
                if age >= min_age:
                    return idx
            elif min_age <= age <= max_age:
                return idx
        return None

    def generate_dataset(self, batch_size=32, test_size=0.2, balance_train=False, overwrite_preprocess=False):
        """
        Prepare balanced train/test generators using pre-cropped faces on disk.
        """
        # Ensure preprocessing is done
        self.ensure_preprocessed_faces(overwrite=overwrite_preprocess)
        
        # Read preprocessed metadata
        pre_dir = os.path.join('data', 'preprocessed_faces')
        metadata_path = os.path.join(pre_dir, 'metadata.json')
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        
        all_paths = np.array([os.path.join(pre_dir, m['filename']) for m in meta])
        all_ages = np.array([m['age'] for m in meta], dtype=np.float32)
        all_genders = np.array([m['gender'] for m in meta])
        all_etnicity = np.array([m['etnicity'] for m in meta])
        
        # Convert ages to class indices
        age_classes = np.array([self._age_to_class(age) for age in all_ages])
        
        # Split into train/test with stratification
        indices = np.arange(len(all_paths))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=42,
            stratify=age_classes
        )
        
        # Balance training data if requested
        if balance_train:
            print("\nBalancing training data...")
            # Store metadata temporarily for balancing
            #self.ages = all_ages
            #train_idx = self._balance_indices(train_idx)
        
        # Use scalar ages as regression targets
        labels = all_ages

        # Extract train/test data
        train_paths = all_paths[train_idx]
        train_labels = labels[train_idx]
        train_ages = all_ages[train_idx]
        train_genders = all_genders[train_idx]
        train_etnicity = all_etnicity[train_idx]
        
        test_paths = all_paths[test_idx]
        test_labels = labels[test_idx]
        test_ages = all_ages[test_idx]
        test_genders = all_genders[test_idx]
        test_etnicity = all_etnicity[test_idx]
        
        # Create generators
        train_gen = ImageDiskGenerator(train_paths, train_labels, batch_size=batch_size, augment=True, shuffle=True)
        test_gen = ImageDiskGenerator(test_paths, test_labels, batch_size=batch_size, augment=False, shuffle=False)
        
        return (
            train_gen,
            test_gen,
            train_paths,
            train_ages,
            train_genders,
            train_etnicity,
            test_paths,
            test_ages,
            test_genders,
            test_etnicity,
        )
