import os
import kagglehub
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


class DatasetDownloader:
    """Class for downloading biometric datasets from Kaggle"""
    
    def __init__(self):
        self.dataset_utkface = r"C:\Users\stina\.cache\kagglehub\datasets\jangedoo\utkface-new\versions\1"
        self.dataset_facial_age = r"C:\Users\stina\.cache\kagglehub\datasets\frabbisw\facial-age\versions\1"
        self.images = []
        self.genders = []
        self.etnicity = []
        self.ages = []

    def download_facial_age_dataset(self):
        """Download Facial Age dataset from kaggle"""
        
        # Download latest version
        path = kagglehub.dataset_download("frabbisw/facial-age")
        print("Path to Facial Age dataset:", path)
        self.dataset_facial_age = path
        return path
    
    def download_utkface_dataset(self):
        """Download UTKFace dataset from kaggle"""
        
        # Download latest version
        path = kagglehub.dataset_download("jangedoo/utkface-new")
        print("Path to UTKFace dataset:", path)
        self.dataset_utkface = path
        return path
    
    def download_all(self):
        """Download all datasets"""
        self.download_utkface_dataset()
        self.download_facial_age_dataset()
        return self.dataset_utkface, self.dataset_facial_age
    
    def create_image_age_dict(self):
        # Start with UTK face dataset
        for root, dirs, files in os.walk(self.dataset_utkface):
            for img_name in files:
                try:
                    age, gender, etnicity = img_name.split('_')[:3]
                    age, gender, etnicity = int(age), int(gender), int(etnicity)
                    
                    # Convert gender to string (0=Male, 1=Female)
                    gender_str = "Male" if gender == 0 else "Female"
                    
                    # Convert ethnicity to string (White (0), Black (1), Asian (2), Indian (3), Others (4))
                    etnicity_map = {0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Others"}
                    etnicity_str = etnicity_map.get(etnicity, "Unknown")
                    
                    img_path = os.path.join(root, img_name)
                    img = self.resize_normalize_image(img_path)
                    if img is not None:
                        self.images.append(img)
                        self.ages.append(age)
                        self.genders.append(gender_str)
                        self.etnicity.append(etnicity_str)
                except:
                    continue  # Skip any invalid entries
        self.images = np.array(self.images) / 255.0  # Normalize
        self.ages = np.array(self.ages, dtype=np.float32)  # Convert to float32
        
        # Add facial age dataset
        """
        for root, dirs, files in os.walk(self.dataset_facial_age):
            # Get the age from the directory name
            age_folder = os.path.basename(root)
            
            for img_name in files:
                try:
                    # Use directory name as age
                    age = int(age_folder)
                    
                    img_path = os.path.join(root, img_name)
                    img = self.resize_normalize_image(img_path)
                    if img is not None:
                        self.age_image[img] = {'age':age}
                except:
                    continue  # Skip any invalid entries
        """
                
    def resize_normalize_image(self, img_path):
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (64, 64))  # Resize for uniform input size
        return img
    
    def generate_dataset(self):
        """Split age_image dictionary into train and test sets
        """
        print("Loading and processing images...")
        self.create_image_age_dict()
        print(f"Loaded {len(self.images)} images")
        
        # Split the data - returns in pairs (train, test) for each input array
        X_train, X_test, age_train, age_test, gen_train, gen_test, etn_train, etn_test = train_test_split(
            self.images, self.ages, self.genders, self.etnicity, 
            test_size=0.2, random_state=42
        )
        
        print(f"Split complete: {len(X_train)} train, {len(X_test)} test")
        print(f"Age train dtype: {type(age_train[0])}, example: {age_train[0]}")
        
        return X_train, X_test, age_train, age_test, gen_train, gen_test, etn_train, etn_test

