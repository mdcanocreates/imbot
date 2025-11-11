"""
Data setup utilities for automatic dataset download.

This module provides functionality to automatically download and extract
the cell image dataset from Dropbox if it's not already present locally.
"""

import os
import zipfile
from pathlib import Path
from typing import Optional
import urllib.request
import urllib.error


# Dropbox direct download URL for cell images ZIP
DATASET_URL = "https://www.dropbox.com/scl/fi/na732v7y9nt542rnjm17u/cell_images.zip?rlkey=sg0kzmdlqtvnq8w68yxo5qrvr&st=3gzh6e7h&dl=1"


def ensure_data_available(
    data_root: str = "img_model",
    dataset_url: str = DATASET_URL,
    zip_name: str = "cell_images.zip",
) -> str:
    """
    Ensure that `data_root` (default 'img_model') exists and is non-empty.
    If not, download a zip of the images from `dataset_url` and unzip it so
    that `data_root` is available.
    
    Parameters
    ----------
    data_root : str
        Path to the data directory (default: "img_model")
    dataset_url : str
        URL to download the ZIP file from
    zip_name : str
        Name for the downloaded ZIP file (default: "cell_images.zip")
    
    Returns
    -------
    str
        Status message indicating what happened:
        - "exists" if data_root already exists and is non-empty
        - "downloaded" if data was downloaded and extracted
        - "error: <message>" if an error occurred
    
    Steps:
      - If `data_root` exists and has some files/subdirs, do nothing.
      - Else:
          * download `zip_name` from `dataset_url` using urllib
          * save to repo root
          * unzip it into the repo root
          * after unzip, we should have `img_model/...` present.
    """
    data_path = Path(data_root)
    
    # Check if data_root exists and is non-empty
    if data_path.exists() and data_path.is_dir():
        # Check if it has any subdirectories or files
        contents = list(data_path.iterdir())
        if len(contents) > 0:
            return "exists"
    
    # Data doesn't exist or is empty - need to download
    import sys
    # Print to stderr so it shows in Streamlit console
    print(f"📥 Dataset not found in {data_root}. Downloading from Dropbox...", file=sys.stderr)
    
    try:
        # Download the ZIP file
        zip_path = Path(zip_name)
        print(f"  Downloading {zip_name}...", file=sys.stderr)
        
        def show_progress(block_num, block_size, total_size):
            """Show download progress."""
            downloaded = block_num * block_size
            percent = min(100, (downloaded / total_size) * 100) if total_size > 0 else 0
            if block_num % 10 == 0:  # Print every 10 blocks to avoid spam
                print(f"  Progress: {percent:.1f}%", end='\r', file=sys.stderr)
        
        urllib.request.urlretrieve(dataset_url, zip_path, reporthook=show_progress)
        print(f"  ✓ Downloaded {zip_name}", file=sys.stderr)
        
        # Extract the ZIP file
        print(f"  Extracting {zip_name}...", file=sys.stderr)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
        print(f"  ✓ Extracted to {data_root}/", file=sys.stderr)
        
        # Clean up the ZIP file
        if zip_path.exists():
            zip_path.unlink()
            print(f"  ✓ Removed {zip_name}", file=sys.stderr)
        
        # Verify extraction was successful
        if data_path.exists() and data_path.is_dir():
            contents = list(data_path.iterdir())
            if len(contents) > 0:
                print(f"✅ Dataset ready in {data_root}/", file=sys.stderr)
                return "downloaded"
            else:
                return "error: Extracted ZIP but directory is empty"
        else:
            return f"error: Extraction completed but {data_root} not found"
            
    except urllib.error.URLError as e:
        error_msg = f"Network error: {e}"
        print(f"❌ {error_msg}", file=sys.stderr)
        return f"error: {error_msg}"
    except zipfile.BadZipFile as e:
        error_msg = f"Invalid ZIP file: {e}"
        print(f"❌ {error_msg}", file=sys.stderr)
        return f"error: {error_msg}"
    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        print(f"❌ {error_msg}", file=sys.stderr)
        return f"error: {error_msg}"

