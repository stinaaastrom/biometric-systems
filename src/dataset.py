import os
import kagglehub
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from image_processing import ImageProcesser
import albumentations as A

IMAGE_SIZE = (64, 64)

# Data augmentation pipeline
augmentation = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=10, p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
])

# Define data folder paths
DATA_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
UTKFACE_PATH = os.path.join(DATA_FOLDER, 'utkface')
FACIAL_AGE_PATH = os.path.join(DATA_FOLDER, 'facial_age')
CACHE_FILE = os.path.join(DATA_FOLDER, 'processed_data_cache.npz')

class DatasetDownloader:
    """Class for downloading biometric datasets from Kaggle"""
    
    def __init__(self):
        # Ensure data folder exists
        os.makedirs(DATA_FOLDER, exist_ok=True)
        
        # Use data folder paths
        self.dataset_utkface = UTKFACE_PATH
        self.dataset_facial_age = FACIAL_AGE_PATH
        
        # Check if datasets exist in data folder, if not download them
        if not os.path.exists(self.dataset_utkface):
            os.makedirs(self.dataset_utkface, exist_ok=True)
            print("Downloading UTKFace dataset...")
            self.download_utkface_dataset()
        elif len(os.listdir(self.dataset_utkface)) == 0:
            print("Downloading UTKFace dataset...")
            self.download_utkface_dataset()
        else:
            print(f"Found UTKFace dataset at: {self.dataset_utkface}")
        
        if not os.path.exists(self.dataset_facial_age):
            os.makedirs(self.dataset_facial_age, exist_ok=True)
            print("Downloading Facial Age dataset...")
            self.download_facial_age_dataset()
        elif len(os.listdir(self.dataset_facial_age)) == 0:
            print("Downloading Facial Age dataset...")
            self.download_facial_age_dataset()
        else:
            print(f"Found Facial Age dataset at: {self.dataset_facial_age}")
        
        self.images = []
        self.genders = []
        self.etnicity = []
        self.ages = []
        self.image_processor = ImageProcesser()

    def download_facial_age_dataset(self):
        """Download Facial Age dataset from kaggle"""
        
        # Download latest version from Kaggle
        path = kagglehub.dataset_download("frabbisw/facial-age")
        print(f"Facial Age dataset downloaded to: {path}")
        
        # Copy to data folder
        os.makedirs(self.dataset_facial_age, exist_ok=True)
        if os.path.exists(path) and path != self.dataset_facial_age:
            import shutil
            for item in os.listdir(path):
                src = os.path.join(path, item)
                dst = os.path.join(self.dataset_facial_age, item)
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)
        
        return self.dataset_facial_age
    
    def download_utkface_dataset(self):
        """Download UTKFace dataset from kaggle"""
        
        # Download latest version from Kaggle
        path = kagglehub.dataset_download("jangedoo/utkface-new")
        print(f"UTKFace dataset downloaded to: {path}")
        
        # Copy to data folder
        os.makedirs(self.dataset_utkface, exist_ok=True)
        if os.path.exists(path) and path != self.dataset_utkface:
            import shutil
            for item in os.listdir(path):
                src = os.path.join(path, item)
                dst = os.path.join(self.dataset_utkface, item)
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)
        
        return self.dataset_utkface
    
    def download_all(self):
        """Download all datasets"""
        self.download_utkface_dataset()
        self.download_facial_age_dataset()
        return self.dataset_utkface, self.dataset_facial_age
    
    def create_image_age_list(self, max_images_per_class=800):
        """
        Load balanced dataset with equal samples per age class
        """
        
        # Try to load from cache
        if os.path.exists(CACHE_FILE):
            print(f"\nLoading preprocessed data from cache...")
            try:
                cache = np.load(CACHE_FILE, allow_pickle=True)
                self.images = cache['images']
                self.ages = cache['ages']
                self.genders = cache['genders'].tolist()
                self.etnicity = cache['etnicity'].tolist()
                print(f"Loaded {len(self.images)} cached images")
                return
            except Exception as e:
                print(f"Cache load failed: {e}, processing from scratch...")

        # Collect all images from both datasets
        all_images = []
        all_ages = []
        all_genders = []
        all_etnicity = []
        
        # Load UTKFace dataset
        print(f"\nLoading UTKFace dataset from: {self.dataset_utkface}")
        utkface_files = []
        for root, dirs, files in os.walk(self.dataset_utkface):
            for img_name in files:
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    utkface_files.append((root, img_name))
        
        import random
        random.shuffle(utkface_files)
        
        for root, img_name in utkface_files:
            try:
                age, gender, etnicity = img_name.split('_')[:3]
                age, gender, etnicity = int(age), int(gender), int(etnicity)
                
                gender_str = "Male" if gender == 0 else "Female"
                etnicity_map = {0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Others"}
                etnicity_str = etnicity_map.get(etnicity, "unknown")
                
                img_path = os.path.join(root, img_name)
                frame = cv2.imread(img_path)
                
                if frame is None:
                    continue
                
                face_img = self.image_processor.detect_crop_faces(frame)
                if face_img is None:
                    continue
                    
                face_img = self.image_processor.image_enhancements(face_img)
                face_img = cv2.resize(face_img, IMAGE_SIZE)

                if face_img is not None:
                    all_images.append(face_img)
                    all_ages.append(age)
                    all_genders.append(gender_str)
                    all_etnicity.append(etnicity_str)
                    
                if len(all_images) % 100 == 0:
                    print(f"Loaded {len(all_images)} UTKFace images", end='\r')
            except Exception as e:
                continue

        print(f"\nLoaded {len(all_images)} images from UTKFace dataset")
        
        # Load Facial Age dataset
        print(f"\nLoading Facial Age dataset from: {self.dataset_facial_age}")
        facial_age_files = []
        for root, dirs, files in os.walk(self.dataset_facial_age):
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
                frame = cv2.imread(img_path)
                if frame is None:
                    continue
                face_img = self.image_processor.detect_crop_faces(frame)
                if face_img is None:
                    continue
                face_img = self.image_processor.image_enhancements(face_img)
                face_img = cv2.resize(face_img, IMAGE_SIZE)

                if face_img is not None:
                    all_images.append(face_img)
                    all_ages.append(age)
                    all_genders.append('unknown')
                    all_etnicity.append('unknown')
                    
                if len(all_images) % 100 == 0:
                    print(f"Loaded {len(all_images)} total images", end='\r')
            except:
                continue
        
        print(f"\nTotal collected: {len(all_images)} images")
        
        # Now balance by age class
        from collections import defaultdict
        age_buckets = defaultdict(list)
        
        # Define age classes
        age_classes = [(0,12), (13,17), (18,25), (26,35), (36,45), (46,60), (61,74), (75, 120)]
        
        # Assign each image to an age class
        for idx, age in enumerate(all_ages):
            for class_idx, (min_age, max_age) in enumerate(age_classes):
                if min_age <= age <= max_age:
                    age_buckets[class_idx].append(idx)
                    break
        
        # Print distribution before balancing
        print("\nAge class distribution (before balancing):")
        for class_idx, (min_age, max_age) in enumerate(age_classes):
            label = f"{min_age}+" if max_age is None else f"{min_age}-{max_age}"
            count = len(age_buckets[class_idx])
            print(f"  {label}: {count} images")
        
        # Balance by sampling equally from each class
        balanced_indices = []
        print(f"\nBalancing to {max_images_per_class} images per class...")
        for class_idx in range(len(age_classes)):
            indices = age_buckets[class_idx]
            if len(indices) > max_images_per_class:
                sampled = random.sample(indices, max_images_per_class)
            else:
                sampled = indices
            balanced_indices.extend(sampled)
            min_age, max_age = age_classes[class_idx]
            label = f"{min_age}+" if max_age is None else f"{min_age}-{max_age}"
            print(f"  {label}: {len(sampled)} images")
        
        # Create balanced dataset
        self.images = []
        self.ages = []
        self.genders = []
        self.etnicity = []
        
        for idx in balanced_indices:
            self.images.append(all_images[idx])
            self.ages.append(all_ages[idx])
            self.genders.append(all_genders[idx])
            self.etnicity.append(all_etnicity[idx])
        
        # Convert to numpy arrays and normalize
        self.images = np.array(self.images) / 255.0
        self.ages = np.array(self.ages, dtype=np.float32)
        
        print(f"\nFinal balanced dataset: {len(self.images)} images")
        
        # Save to cache
        print(f"\nSaving preprocessed data to cache...")
        try:
            np.savez_compressed(
                CACHE_FILE,
                images=self.images,
                ages=self.ages,
                genders=np.array(self.genders),
                etnicity=np.array(self.etnicity)
            )
            print(f"Cache saved to {CACHE_FILE}")
        except Exception as e:
            print(f"Cache save failed: {e}")

    
    def generate_dataset(self):
        """Split age_image dictionary into train and test sets
        """
        print("Loading and processing images...")
        self.create_image_age_list()
        print(f"Loaded {len(self.images)} images")
        
        # Split the data - returns in pairs (train, test) for each input array
        X_train, X_test, age_train, age_test, gen_train, gen_test, etn_train, etn_test = train_test_split(
            self.images, self.ages, self.genders, self.etnicity, 
            test_size=0.2, random_state=42
        )
        
        print(f"Split complete: {len(X_train)} train, {len(X_test)} test")
        print(f"Age train dtype: {type(age_train[0])}, example: {age_train[0]}")
        
        return X_train, X_test, age_train, age_test, gen_train, gen_test, etn_train, etn_test

