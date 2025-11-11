"""
Prepare VisDrone dataset for training
Downloads and organizes the dataset
"""

import os
import requests
from pathlib import Path
from tqdm import tqdm
import zipfile

VISDRONE_URLS = {
    'train': 'https://github.com/VisDrone/VisDrone-Dataset/releases/download/v1.0/VisDrone2019-DET-train.zip',
    'val': 'https://github.com/VisDrone/VisDrone-Dataset/releases/download/v1.0/VisDrone2019-DET-val.zip',
    'test': 'https://github.com/VisDrone/VisDrone-Dataset/releases/download/v1.0/VisDrone2019-DET-test-dev.zip'
}

def download_file(url, dest_path):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as file, tqdm(
        desc=dest_path.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            pbar.update(size)

def extract_zip(zip_path, extract_to):
    """Extract zip file"""
    print(f"Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def main():
    print("Preparing VisDrone dataset for AerialVision...")
    
    # Create directories
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    downloads_dir = data_dir / 'downloads'
    downloads_dir.mkdir(exist_ok=True)
    
    # Download and extract
    for split, url in VISDRONE_URLS.items():
        print(f"\n{'='*50}")
        print(f"Processing {split} split...")
        print(f"{'='*50}")
        
        zip_path = downloads_dir / f'visdrone_{split}.zip'
        
        # Download
        if not zip_path.exists():
            print(f"Downloading from {url}...")
            download_file(url, zip_path)
        else:
            print(f"Using cached {zip_path.name}")
        
        # Extract
        extract_to = data_dir / split
        if not extract_to.exists():
            extract_zip(zip_path, extract_to)
        else:
            print(f"Already extracted to {extract_to}")
    
    # Create dataset YAML
    yaml_content = """
# VisDrone dataset for AerialVision
# Urban traffic monitoring from aerial imagery

path: data  # dataset root dir
train: train/images  # train images
val: val/images  # val images
test: test/images  # test images

# Classes (10 vehicle/pedestrian types)
names:
  0: pedestrian
  1: people
  2: bicycle
  3: car
  4: van
  5: truck
  6: tricycle
  7: awning-tricycle
  8: bus
  9: motor

# Dataset info
nc: 10  # number of classes
"""
    
    yaml_path = data_dir / 'visdrone.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n{'='*50}")
    print(f"Dataset preparation complete!")
    print(f"{'='*50}")
    print(f"Dataset config: {yaml_path}")
    print(f"Train images: {data_dir}/train/images")
    print(f"Val images: {data_dir}/val/images")
    print(f"Test images: {data_dir}/test/images")
    print(f"\nYou can now start training with:")
    print(f"  python train.py --data {yaml_path}")

if __name__ == '__main__':
    main()
