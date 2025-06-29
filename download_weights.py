"""
Download PartPacker model weights from Hugging Face Hub
"""

import os
import sys
from huggingface_hub import hf_hub_download

MODEL_REPO = "nvidia/PartPacker"
PRETRAINED_DIR = "./pretrained"

def download_weights():
    """Download model weights from Hugging Face Hub"""
    
    print(f"Downloading PartPacker weights from {MODEL_REPO}...")
    
    # Create pretrained directory
    os.makedirs(PRETRAINED_DIR, exist_ok=True)
    
    try:
        # Download VAE weights
        print("Downloading VAE weights...")
        vae_path = hf_hub_download(
            repo_id=MODEL_REPO, 
            filename="vae.pt",
            local_dir=PRETRAINED_DIR
        )
        print(f"VAE weights downloaded: {vae_path}")
        
        # Download Flow weights  
        print("Downloading Flow weights...")
        flow_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename="flow.pt", 
            local_dir=PRETRAINED_DIR
        )
        print(f"Flow weights downloaded: {flow_path}")
        
        print("Model weights downloaded successfully!")
        
        # Verify downloads
        if verify_downloads():
            print("All weights verified successfully!")
        else:
            print("Warning: Some weights may be missing")
            
    except Exception as e:
        print(f"Error downloading weights: {e}")
        sys.exit(1)

def verify_downloads():
    """Verify that all required model files are present"""
    
    required_files = ["vae.pt", "flow.pt"]
    
    for filename in required_files:
        file_path = os.path.join(PRETRAINED_DIR, filename)
        
        if not os.path.exists(file_path):
            print(f"Missing required file: {file_path}")
            return False
            
        # Check file size (should not be empty)
        if os.path.getsize(file_path) == 0:
            print(f"Empty file detected: {file_path}")
            return False
    
    return True

def check_disk_space():
    """Check available disk space before downloading"""
    
    # Get available disk space
    statvfs = os.statvfs('.')
    available_space_gb = (statvfs.f_bavail * statvfs.f_frsize) / (1024**3)

    required_space_gb = 10.0  # Estimated space needed for PartPacker weights
    
    if available_space_gb < required_space_gb:
        print(f"Warning: Low disk space. Available: {available_space_gb:.1f}GB, Required: {required_space_gb:.1f}GB")
        return False
    
    print(f"Disk space check passed. Available: {available_space_gb:.1f}GB")
    return True

if __name__ == "__main__":
    print("PartPacker Weight Downloader")
    print("=" * 40)
    
    # Check disk space
    if not check_disk_space():
        print("Warning: Proceeding with limited disk space...")
    
    # Download weights
    download_weights()
    
    print("Weight download completed!") 