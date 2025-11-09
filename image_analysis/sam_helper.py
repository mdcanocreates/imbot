"""
Optional Segment Anything Model (SAM) integration for improved segmentation.

This module provides optional integration with Meta's Segment Anything Model
for potentially improved cell and nuclear segmentation. SAM is a foundation
model that can generate high-quality masks from prompts.

Note: This is an optional enhancement. The classical skimage-based pipeline
remains the primary segmentation method. SAM can be used as a refinement step
or alternative method if available.

Requirements:
- PyTorch
- segment-anything package
- SAM model weights (download separately)

Usage:
    If SAM is available and model weights are provided, this can be used to:
    1. Generate initial masks using SAM's automatic mask generator
    2. Refine existing masks using SAM's predictor with point/box prompts
    3. Use SAM as a fallback if classical segmentation fails
"""

from typing import Optional, Tuple
import numpy as np
from pathlib import Path

# Try to import SAM (optional dependency)
try:
    from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
    import torch
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("Note: Segment Anything Model (SAM) not available. Install with: pip install segment-anything")


def segment_with_sam_automatic(
    image: np.ndarray,
    model_path: Optional[Path] = None,
    model_type: str = "vit_h",
    points_per_side: int = 32
) -> list:
    """
    Use SAM's automatic mask generator to segment all objects in an image.
    
    This generates masks for all objects in the image without prompts.
    Useful for initial segmentation or as a fallback method.
    
    Parameters
    ----------
    image : np.ndarray
        Input image (HWC format, uint8, RGB)
    model_path : Path, optional
        Path to SAM model checkpoint (.pth file)
    model_type : str
        SAM model type: "vit_h", "vit_l", or "vit_b"
    points_per_side : int
        Number of points per side for automatic mask generation
    
    Returns
    -------
    list
        List of mask dictionaries with keys: 'segmentation', 'area', 'bbox', etc.
    """
    if not SAM_AVAILABLE:
        raise ImportError("SAM not available. Install segment-anything package.")
    
    if model_path is None or not model_path.exists():
        raise FileNotFoundError(f"SAM model not found at {model_path}")
    
    # Load SAM model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=str(model_path))
    sam.to(device=device)
    
    # Create automatic mask generator
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=points_per_side,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
    )
    
    # Generate masks
    masks = mask_generator.generate(image)
    
    return masks


def segment_with_sam_prompt(
    image: np.ndarray,
    point_coords: np.ndarray,
    point_labels: np.ndarray,
    model_path: Optional[Path] = None,
    model_type: str = "vit_h"
) -> Tuple[np.ndarray, float]:
    """
    Use SAM with point prompts to segment a specific object.
    
    This is useful when you have prior knowledge of object location
    (e.g., from nuclei detection) and want SAM to generate a precise mask.
    
    Parameters
    ----------
    image : np.ndarray
        Input image (HWC format, uint8, RGB)
    point_coords : np.ndarray
        Point coordinates (N, 2) in image space
    point_labels : np.ndarray
        Point labels (N,) where 1 = foreground, 0 = background
    model_path : Path, optional
        Path to SAM model checkpoint (.pth file)
    model_type : str
        SAM model type: "vit_h", "vit_l", or "vit_b"
    
    Returns
    -------
    tuple
        (mask, score) where mask is binary mask and score is prediction quality
    """
    if not SAM_AVAILABLE:
        raise ImportError("SAM not available. Install segment-anything package.")
    
    if model_path is None or not model_path.exists():
        raise FileNotFoundError(f"SAM model not found at {model_path}")
    
    # Load SAM model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=str(model_path))
    sam.to(device=device)
    
    # Create predictor
    predictor = SamPredictor(sam)
    predictor.set_image(image)
    
    # Predict mask from points
    masks, scores, logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )
    
    # Return best mask (highest score)
    best_idx = np.argmax(scores)
    return masks[best_idx], float(scores[best_idx])


def refine_cell_mask_with_sam(
    image: np.ndarray,
    initial_mask: np.ndarray,
    model_path: Optional[Path] = None,
    model_type: str = "vit_h"
) -> np.ndarray:
    """
    Use SAM to refine an existing cell mask.
    
    This can improve masks generated by classical methods by using SAM
    with prompts derived from the initial mask (e.g., centroid point).
    
    Parameters
    ----------
    image : np.ndarray
        Input image (HWC format, uint8, RGB)
    initial_mask : np.ndarray
        Initial binary mask to refine
    model_path : Path, optional
        Path to SAM model checkpoint (.pth file)
    model_type : str
        SAM model type
    
    Returns
    -------
    np.ndarray
        Refined binary mask
    """
    if not SAM_AVAILABLE:
        return initial_mask  # Return original if SAM not available
    
    if model_path is None or not model_path.exists():
        return initial_mask  # Return original if model not found
    
    try:
        # Get centroid of initial mask as prompt
        from skimage import measure
        from skimage.measure import regionprops
        labeled = measure.label(initial_mask)
        if labeled.max() == 0:
            return initial_mask
        
        regions = regionprops(labeled)
        largest = max(regions, key=lambda r: r.area)
        centroid = largest.centroid
        
        # Use centroid as positive point prompt
        point_coords = np.array([[centroid[1], centroid[0]]])  # (x, y) format
        point_labels = np.array([1])  # Foreground point
        
        # Get refined mask from SAM
        refined_mask, score = segment_with_sam_prompt(
            image, point_coords, point_labels, model_path, model_type
        )
        
        return refined_mask.astype(bool)
        
    except Exception as e:
        print(f"Warning: SAM refinement failed: {e}")
        return initial_mask  # Return original on error

