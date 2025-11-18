import os
import kagglehub
import cv2
import numpy as np

class DatasetDownloader:
    """Class for downloading biometric datasets from Kaggle"""
    
    def __init__(self):
        self.dataset_utkface = r"C:\Users\stina\.cache\kagglehub\datasets\jangedoo\utkface-new\versions\1"
        self.dataset_facial_age = r"C:\Users\stina\.cache\kagglehub\datasets\frabbisw\facial-age\versions\1"
        self.age_image = {}
    
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
                    age, gender, _ = img_name.split('_')[:3]
                    age, gender = int(age), int(gender)
                    
                    img_path = os.path.join(root, img_name)
                    img = self.resize_normalize_image(img_path)
                    if img is not None:
                        self.age_image[age] = img
                except:
                    continue  # Skip any invalid entries
        
        # Add facial age dataset
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
                        self.age_image[age] = img
                except:
                    continue  # Skip any invalid entries

                
    def resize_normalize_image(self, img_path):
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (64, 64))  # Resize for uniform input size
            img = np.array(img) / 255.0  # Normalize
        return img

if __name__ == "__main__":
    # Example usage
    downloader = DatasetDownloader()
    downloader.create_image_age_dict()
    print(downloader.age_image)