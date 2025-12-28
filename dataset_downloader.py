import os
import kagglehub


# Define data folder paths
DATA_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
UTKFACE_PATH = os.path.join(DATA_FOLDER, 'utkface')
FACIAL_AGE_PATH = os.path.join(DATA_FOLDER, 'facial_age')
PROCESSED_FACES_PATH = os.path.join(DATA_FOLDER, 'processed_faces')
PROCESSED_UTKFACE_PATH = os.path.join(PROCESSED_FACES_PATH, 'utkface')
PROCESSED_FACIAL_AGE_PATH = os.path.join(PROCESSED_FACES_PATH, 'facial_age')

class DatasetDownloader:
    """Class for downloading biometric datasets from Kaggle"""
    
    def __init__(self):
        # Ensure data folder exists
        os.makedirs(DATA_FOLDER, exist_ok=True)
        
        # Use data folder paths
        self.dataset_utkface = UTKFACE_PATH
        self.dataset_facial_age = FACIAL_AGE_PATH
        self.processed_root = PROCESSED_FACES_PATH
        self.processed_utkface = PROCESSED_UTKFACE_PATH
        self.processed_facial_age = PROCESSED_FACIAL_AGE_PATH
        
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
