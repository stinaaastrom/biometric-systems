import os
import kagglehub
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from image_processing import ImageProcesser
IMAGE_SIZE = (64, 64)

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
    
    def create_image_age_list(self, max_images_per_dataset=500):
        
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

        # Start with UTK face dataset
        utkface_count = 0
        print(f"\nLoading UTKFace dataset from: {self.dataset_utkface}")
        
        for root, dirs, files in os.walk(self.dataset_utkface):
            if utkface_count >= max_images_per_dataset:
                break
            for img_name in files:
                if utkface_count >= max_images_per_dataset:
                    break

                try:
                    # Skip non-image files
                    if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        continue
                    
                    age, gender, etnicity = img_name.split('_')[:3]
                    age, gender, etnicity = int(age), int(gender), int(etnicity)
                    
                    # Convert gender to string (0=Male, 1=Female)
                    gender_str = "Male" if gender == 0 else "Female"
                    
                    # Convert ethnicity to string (White (0), Black (1), Asian (2), Indian (3), Others (4))
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
                        self.images.append(face_img)
                        self.ages.append(age)
                        self.genders.append(gender_str)
                        self.etnicity.append(etnicity_str)
                        utkface_count += 1
                        if utkface_count % 50 == 0:
                            print(f"Loaded {utkface_count}/{max_images_per_dataset} UTKFace images", end='\r')
                except Exception as e:
                    continue

        
        print(f"\n Loaded {utkface_count} images from UTKFace dataset")
        
        print(f"Loaded {utkface_count} images from UTKFace dataset")
        
        # Add facial age dataset
        facial_age_count = 0
        print(f"\n Loading Facial Age dataset from: {self.dataset_facial_age}")
        for root, dirs, files in os.walk(self.dataset_facial_age):
            if facial_age_count >= max_images_per_dataset:
                break
            # Get the age from the directory name
            age_folder = os.path.basename(root)
            
            for img_name in files:
                if facial_age_count >= max_images_per_dataset:
                    break

                try:
                    # Use directory name as age
                    age = int(age_folder)
                    
                    img_path = os.path.join(root, img_name)
                    frame = cv2.imread(img_path)
                    face_img = self.image_processor.detect_crop_faces(frame)
                    face_img = self.image_processor.image_enhancements(face_img)
                    face_img = cv2.resize(face_img, IMAGE_SIZE)

                    if face_img is not None:
                        self.images.append(face_img)
                        self.ages.append(age)
                        self.genders.append('unknown')
                        self.etnicity.append('unknown')
                        facial_age_count += 1
                        if facial_age_count % 50 == 0:
                            print(f" Loaded {facial_age_count}/{max_images_per_dataset} Facial Age images", end='\r')
                except:
                    continue  # Skip any invalid entries
        
        print(f"\nLoaded {facial_age_count} images from Facial Age dataset")
        
        # Convert to numpy arrays and normalize after loading both datasets
        self.images = np.array(self.images) / 255.0  # Normalize
        self.ages = np.array(self.ages, dtype=np.float32)  # Convert to float32
        
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

