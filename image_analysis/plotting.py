"""
Plotting utilities for QC overlays and visualization.

This module provides functions to overlay masks on images for
visual quality control of segmentation results.
"""

from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure


def overlay_mask_on_image(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    alpha: float = 0.3,
    contour_color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    contour_width: float = 2.0
) -> np.ndarray:
    """
    Overlay a binary mask on an image with semi-transparent fill and contour.
    
    Parameters
    ----------
    image : np.ndarray
        Grayscale or RGB image (2D or 3D array)
    mask : np.ndarray
        Binary mask to overlay (2D array, same shape as image or image[:2])
    color : tuple
        RGB color for the mask overlay (values 0-1)
    alpha : float
        Transparency of the mask overlay (0-1)
    contour_color : tuple
        RGB color for the contour outline
    contour_width : float
        Width of the contour line
    
    Returns
    -------
    np.ndarray
        Image with mask overlay (RGB, 3D array)
    """
    # Ensure image is RGB
    if len(image.shape) == 2:
        # Grayscale to RGB
        image_rgb = np.stack([image, image, image], axis=-1)
    else:
        image_rgb = image.copy()
    
    # Normalize image to 0-1 range if needed
    if image_rgb.max() > 1.0:
        image_rgb = image_rgb.astype(np.float32) / 255.0
    
    # Ensure mask matches image spatial dimensions
    if mask.shape != image_rgb.shape[:2]:
        raise ValueError(
            f"Mask shape {mask.shape} does not match image shape {image_rgb.shape[:2]}"
        )
    
    # Create overlay
    overlay = image_rgb.copy()
    
    # Apply semi-transparent mask overlay
    # When indexing with a 2D boolean mask into a 3D array, we get shape (N, 3)
    # where N is the number of True pixels
    color_array = np.array(color)  # Shape (3,)
    mask_pixels = overlay[mask]  # Shape (N, 3)
    overlay[mask] = (
        alpha * color_array + (1 - alpha) * mask_pixels
    )
    
    # Add contour outline
    contours = measure.find_contours(mask.astype(float), 0.5)
    for contour in contours:
        # Clip contour to image bounds
        contour[:, 0] = np.clip(contour[:, 0], 0, image_rgb.shape[0] - 1)
        contour[:, 1] = np.clip(contour[:, 1], 0, image_rgb.shape[1] - 1)
        
        # Draw contour pixels (simplified approach)
        contour_int = contour.astype(int)
        # Remove duplicates
        contour_int = np.unique(contour_int, axis=0)
        
        # Draw contour with specified width
        for r, c in contour_int:
            # Draw a small circle/square around each contour point
            radius = int(np.ceil(contour_width))
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    rr, cc = r + dr, c + dc
                    if (0 <= rr < image_rgb.shape[0] and 
                        0 <= cc < image_rgb.shape[1] and
                        np.sqrt(dr**2 + dc**2) <= contour_width):
                        overlay[rr, cc] = contour_color
    
    return overlay


def plot_combo_with_masks(
    combo_image: np.ndarray,
    cell_mask: np.ndarray,
    nucleus_mask: np.ndarray,
    output_path: str,
    cell_color: Tuple[float, float, float] = (0.0, 1.0, 0.0),
    nucleus_color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    alpha: float = 0.3
) -> None:
    """
    Create and save a QC image showing combo image with cell and nuclear outlines.
    
    Parameters
    ----------
    combo_image : np.ndarray
        RGB composite image (3D array)
    cell_mask : np.ndarray
        Binary mask of the cell
    nucleus_mask : np.ndarray
        Binary mask of all nuclei
    output_path : str
        Path to save the output PNG
    cell_color : tuple
        RGB color for cell mask overlay
    nucleus_color : tuple
        RGB color for nuclear mask overlay
    alpha : float
        Transparency of mask overlays
    """
    # Start with combo image
    overlay = combo_image.copy()
    
    # Normalize to 0-1 if needed
    if overlay.max() > 1.0:
        overlay = overlay.astype(np.float32) / 255.0
    
    # Overlay cell mask
    if np.any(cell_mask):
        cell_overlay = overlay_mask_on_image(
            overlay, cell_mask, color=cell_color, alpha=alpha,
            contour_color=cell_color, contour_width=2.0
        )
        overlay = cell_overlay
    
    # Overlay nucleus mask
    if np.any(nucleus_mask):
        nucleus_overlay = overlay_mask_on_image(
            overlay, nucleus_mask, color=nucleus_color, alpha=alpha,
            contour_color=nucleus_color, contour_width=2.0
        )
        overlay = nucleus_overlay
    
    # Create figure and save
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(overlay)
    ax.axis('off')
    ax.set_title('Combo image with cell (green) and nuclear (red) masks')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_actin_with_cell_mask(
    actin_image: np.ndarray,
    cell_mask: np.ndarray,
    output_path: str,
    cell_color: Tuple[float, float, float] = (0.0, 1.0, 0.0),
    alpha: float = 0.3
) -> None:
    """
    Create and save a QC image showing Actin image with cell mask overlay.
    
    Parameters
    ----------
    actin_image : np.ndarray
        Grayscale Actin channel image
    cell_mask : np.ndarray
        Binary mask of the cell
    output_path : str
        Path to save the output PNG
    cell_color : tuple
        RGB color for cell mask overlay
    alpha : float
        Transparency of mask overlay
    """
    # Overlay cell mask on actin image
    overlay = overlay_mask_on_image(
        actin_image, cell_mask, color=cell_color, alpha=alpha,
        contour_color=cell_color, contour_width=2.0
    )
    
    # Create figure and save
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(overlay)
    ax.axis('off')
    ax.set_title('Actin image with cell mask overlay')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_nuclei_with_nuclear_mask(
    nuclei_image: np.ndarray,
    nucleus_mask: np.ndarray,
    cell_mask: np.ndarray,
    output_path: str,
    nucleus_color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    cell_color: Tuple[float, float, float] = (0.0, 1.0, 0.0),
    alpha: float = 0.3
) -> None:
    """
    Create and save a QC image showing Nuclei image with nuclear mask overlay.
    
    Parameters
    ----------
    nuclei_image : np.ndarray
        Grayscale Nuclei channel image
    nucleus_mask : np.ndarray
        Binary mask of all nuclei
    cell_mask : np.ndarray
        Binary mask of the cell (shown as outline)
    output_path : str
        Path to save the output PNG
    nucleus_color : tuple
        RGB color for nuclear mask overlay
    cell_color : tuple
        RGB color for cell outline
    alpha : float
        Transparency of mask overlay
    """
    # Overlay nuclear mask on nuclei image
    overlay = overlay_mask_on_image(
        nuclei_image, nucleus_mask, color=nucleus_color, alpha=alpha,
        contour_color=nucleus_color, contour_width=2.0
    )
    
    # Add cell outline
    from skimage import measure
    contours = measure.find_contours(cell_mask.astype(float), 0.5)
    for contour in contours:
        contour_int = contour.astype(int)
        contour_int[:, 0] = np.clip(contour_int[:, 0], 0, overlay.shape[0] - 1)
        contour_int[:, 1] = np.clip(contour_int[:, 1], 0, overlay.shape[1] - 1)
        for r, c in contour_int:
            radius = 2
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    rr, cc = r + dr, c + dc
                    if (0 <= rr < overlay.shape[0] and 
                        0 <= cc < overlay.shape[1] and
                        np.sqrt(dr**2 + dc**2) <= radius):
                        overlay[rr, cc] = cell_color
    
    # Create figure and save
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(overlay)
    ax.axis('off')
    ax.set_title('Nuclei image with nuclear mask overlay (cell outline in green)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

