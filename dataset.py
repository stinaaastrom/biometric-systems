import kagglehub

def download_dataset():
    """Download dataset from kaggle"""
    
    # Download latest version
    path = kagglehub.dataset_download("frabbisw/facial-age")

    #TODO: Dowload UTK face dataset as well

    print("Path to dataset files:", path)