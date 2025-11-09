"""
I/O utilities for loading images and mapping cell IDs to file paths.

This module handles loading of fluorescence and brightfield images
for each cell, with configurable file path patterns.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
from skimage import io, color, transform


# Default file naming pattern
DEFAULT_CHANNELS = {
    'actin': 'Actin',
    'microtubules': 'Microtubules',
    'nuclei': 'Nuclei',
    'combo': 'Combo',
    'bf': 'BF'
}

# Cell ID to filename prefix mapping (for cells with non-standard naming)
# If a cell ID is not in this map, it uses the cell_id as the prefix
CELL_ID_TO_PREFIX = {
    'CellB': 'PhenotypeI',
    'CellC': 'PhenotypeII'
}


def get_cell_filepaths(
    cell_id: str,
    data_root: Path,
    channel_map: Optional[Dict[str, str]] = None
) -> Dict[str, Path]:
    """
    Get file paths for all channels of a given cell.
    
    Parameters
    ----------
    cell_id : str
        Identifier for the cell (e.g., "CellA", "CellB", "CellC")
    data_root : Path
        Root directory containing cell subdirectories
    channel_map : dict, optional
        Custom mapping from channel names to filename suffixes.
        If None, uses DEFAULT_CHANNELS.
    
    Returns
    -------
    dict
        Dictionary mapping channel names to file paths.
        Keys: 'actin', 'microtubules', 'nuclei', 'combo', 'bf'
    
    Raises
    ------
    FileNotFoundError
        If the cell directory or any required image file is not found.
    """
    if channel_map is None:
        channel_map = DEFAULT_CHANNELS
    
    cell_dir = data_root / cell_id
    if not cell_dir.exists():
        raise FileNotFoundError(f"Cell directory not found: {cell_dir}")
    
    # Get filename prefix (use mapping if available, otherwise use cell_id)
    filename_prefix = CELL_ID_TO_PREFIX.get(cell_id, cell_id)
    
    filepaths = {}
    for channel_key, suffix in channel_map.items():
        # Try common image extensions
        for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
            # Try both patterns: {prefix}_{suffix} and {cell_id}_{suffix}
            candidates = [
                cell_dir / f"{filename_prefix}_{suffix}{ext}",
                cell_dir / f"{cell_id}_{suffix}{ext}"
            ]
            for candidate in candidates:
                if candidate.exists():
                    filepaths[channel_key] = candidate
                    break
            if channel_key in filepaths:
                break
        else:
            raise FileNotFoundError(
                f"Image file not found for {cell_id} channel {channel_key} "
                f"(tried patterns: {filename_prefix}_{suffix}.* and {cell_id}_{suffix}.*)"
            )
    
    return filepaths


def load_cell_images(
    cell_id: str,
    data_root: Path,
    channel_map: Optional[Dict[str, str]] = None
) -> Dict[str, np.ndarray]:
    """
    Load all images for a given cell.
    
    Parameters
    ----------
    cell_id : str
        Identifier for the cell
    data_root : Path
        Root directory containing cell subdirectories
    channel_map : dict, optional
        Custom mapping from channel names to filename suffixes
    
    Returns
    -------
    dict
        Dictionary mapping channel names to image arrays.
        All images are converted to grayscale (2D arrays) except
        'combo' which remains RGB (3D array).
        Keys: 'actin', 'microtubules', 'nuclei', 'combo', 'bf'
    """
    filepaths = get_cell_filepaths(cell_id, data_root, channel_map)
    
    images = {}
    for channel_key, filepath in filepaths.items():
        img = io.imread(str(filepath))
        
        # Convert to grayscale if needed (except combo which should be RGB)
        if channel_key == 'combo':
            # Combo image should remain RGB
            if len(img.shape) == 2:
                # If it's grayscale, convert to RGB
                img = color.gray2rgb(img)
            images[channel_key] = img
        else:
            # All other channels should be grayscale
            if len(img.shape) == 3:
                img = color.rgb2gray(img)
            images[channel_key] = img
    
    return images


def resize_image_to_target(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize an image to a target size.
    
    Parameters
    ----------
    image : np.ndarray
        Image to resize (2D or 3D array)
    target_size : tuple
        Target (height, width)
    
    Returns
    -------
    np.ndarray
        Resized image
    """
    if len(image.shape) == 2:
        # Grayscale image
        resized = transform.resize(image, target_size, preserve_range=True, anti_aliasing=True)
        # Preserve original dtype
        if image.dtype == np.uint8:
            resized = (resized * 255).astype(np.uint8)
        else:
            resized = resized.astype(image.dtype)
    else:
        # RGB image
        resized = transform.resize(image, target_size + (image.shape[2],), 
                                   preserve_range=True, anti_aliasing=True)
        if image.dtype == np.uint8:
            resized = (resized * 255).astype(np.uint8)
        else:
            resized = resized.astype(image.dtype)
    
    return resized


def normalize_image_sizes(images: Dict[str, np.ndarray]) -> Tuple[int, int]:
    """
    Normalize image sizes by resizing all images to match the most common size.
    The combo image is excluded from size determination but will be resized separately.
    
    Parameters
    ----------
    images : dict
        Dictionary of image arrays
    
    Returns
    -------
    tuple
        (height, width) of the normalized images
    """
    # Get sizes for analysis images (exclude combo which is only for visualization)
    analysis_channels = ['actin', 'microtubules', 'nuclei', 'bf']
    sizes = {}
    for channel in analysis_channels:
        if channel in images:
            img = images[channel]
            if len(img.shape) == 2:
                size = img.shape
            else:
                size = img.shape[:2]
            sizes[channel] = size
    
    if not sizes:
        raise ValueError("No analysis images found")
    
    # Find the most common size (or use the first one if all are different)
    from collections import Counter
    size_counts = Counter(sizes.values())
    target_size = size_counts.most_common(1)[0][0]
    
    # Resize all analysis images to target size
    for channel in analysis_channels:
        if channel in images:
            img = images[channel]
            current_size = img.shape[:2] if len(img.shape) == 3 else img.shape
            if current_size != target_size:
                print(f"  Resizing {channel} from {current_size} to {target_size}")
                images[channel] = resize_image_to_target(images[channel], target_size)
    
    # Resize combo image separately if it exists (for visualization)
    if 'combo' in images:
        combo_img = images['combo']
        combo_size = combo_img.shape[:2]
        if combo_size != target_size:
            print(f"  Resizing combo from {combo_size} to {target_size}")
            images['combo'] = resize_image_to_target(combo_img, target_size)
    
    return target_size

