"""
Segmentation functions for cell and nuclear masks.

This module implements segmentation strategies for:
- Cell segmentation from Actin channel (nuclei-anchored ROI method)
- Nuclear segmentation from Nuclei channel
- Cytoplasm mask derivation

All functions use scikit-image and are parameterized for reproducibility.

OLD APPROACH (DEPRECATED - why it fails):
- Segmented cell from full Actin image using global Otsu thresholding
- Kept largest connected component
- PROBLEM: Mask hugs background speckle and borders; misses true cortex
- PROBLEM: Cell occupies only part of field, so "largest component" ends up being
  a noisy region, not the true cell
- PROBLEM: Illumination is non-uniform, global thresholding fails
- RESULT: Cell outline snakes around noisy background and image borders

NEW APPROACH (CORRECTED):
- First segment nuclei robustly to get bounding box
- Define ROI around nuclei (expanded by margin)
- Segment cell within ROI using Actin channel with adaptive thresholding
- Pick component whose centroid is closest to nuclei centroid
- This guarantees we pick the cell containing the nuclei, not random speckle
- RESULT: Cell outline follows actual actin cortex, excludes background noise
"""

from typing import Optional, Tuple, Dict
import numpy as np
from skimage import filters, morphology, measure, segmentation, feature
from skimage.morphology import disk, closing, opening, remove_small_objects
from skimage.segmentation import watershed
from skimage.measure import label, regionprops
from scipy import ndimage
import warnings


# Default segmentation parameters
DEFAULT_PARAMS = {
    'blur_sigma': 2.0,  # Gaussian blur sigma for noise reduction
    'blur_sigma_cell': 2.5,  # Gaussian blur for cell segmentation (slightly larger)
    'min_size_cell': 200,  # Minimum area (pixels) for cell objects in ROI
    'min_size_nucleus': 50,  # Minimum area (pixels) for nuclear objects (DEPRECATED - now adaptive)
    'closing_radius_cell': 5,  # Morphological closing radius for cell mask
    'opening_radius_nucleus': 2,  # Morphological opening radius for nuclear mask
    'nucleus_erosion_radius': 2,  # Erosion radius to separate nucleus from cytoplasm
    # New adaptive parameters for nuclear segmentation
    'nuclear_area_min_fraction': 0.01,  # Minimum nuclear area as fraction of cell area
    'nuclear_area_max_fraction': 0.25,  # Maximum nuclear area as fraction of cell area (reduced from 0.5)
    # ROI parameters for nuclei-anchored cell segmentation
    'roi_margin': 150,  # Margin in pixels to expand around nuclei bounding box (increased for better cell capture)
}


def segment_cell_nuclei_anchored(
    actin_image: np.ndarray,
    nuclei_image: np.ndarray,
    blur_sigma: float = DEFAULT_PARAMS['blur_sigma_cell'],
    min_size: int = DEFAULT_PARAMS['min_size_cell'],
    closing_radius: int = DEFAULT_PARAMS['closing_radius_cell'],
    roi_margin: int = DEFAULT_PARAMS['roi_margin']
) -> np.ndarray:
    """
    Segment cell using nuclei-anchored ROI approach.
    
    This function uses a two-stage approach:
    1. First segments nuclei to define a Region of Interest (ROI)
    2. Then segments the cell within that ROI using the Actin channel
    3. Picks the component whose centroid is closest to the nuclei centroid
    
    Why this approach:
    - Nuclei are bright, well-localized objects that reliably mark cell location
    - ROI excludes background noise and image borders
    - Picking component closest to nuclei guarantees we get the correct cell
    - Prevents mask from hugging background speckle and borders
    
    Parameters
    ----------
    actin_image : np.ndarray
        Grayscale Actin channel image (2D array)
    nuclei_image : np.ndarray
        Grayscale Nuclei channel image (2D array)
    blur_sigma : float
        Standard deviation for Gaussian blur (noise reduction)
    min_size : int
        Minimum area in pixels for cell objects (smaller objects removed)
    closing_radius : int
        Radius for morphological closing (fills gaps along cortex)
    roi_margin : int
        Margin in pixels to expand around nuclei bounding box
    
    Returns
    -------
    np.ndarray
        Binary mask of the cell (True = cell, False = background)
    """
    # Normalize images to 0-1 range if needed
    if actin_image.max() > 1.0:
        actin_image = actin_image.astype(np.float32) / 255.0
    if nuclei_image.max() > 1.0:
        nuclei_image = nuclei_image.astype(np.float32) / 255.0
    
    # Step 1: Segment nuclei to get bounding box
    # Use a simple, robust nuclear segmentation for ROI definition
    nuclei_blurred = ndimage.gaussian_filter(nuclei_image, sigma=1.5)
    
    # Threshold nuclei (use percentile to be robust)
    nuclei_pixels = nuclei_blurred[nuclei_image > 0.1]  # Ignore very dark pixels
    if len(nuclei_pixels) == 0:
        # Fallback: use global threshold
        nuclei_threshold = np.percentile(nuclei_blurred, 85)
    else:
        nuclei_threshold = np.percentile(nuclei_pixels, 85)
    
    nuclei_binary = nuclei_blurred > nuclei_threshold
    
    # Remove small noise
    nuclei_binary = remove_small_objects(nuclei_binary, min_size=100)
    
    # Get bounding box of nuclei
    labeled_nuclei = label(nuclei_binary)
    if labeled_nuclei.max() == 0:
        # No nuclei found, fall back to center of image
        h, w = actin_image.shape
        min_row, max_row = h // 4, 3 * h // 4
        min_col, max_col = w // 4, 3 * w // 4
        nuclei_centroid = (h / 2, w / 2)  # Center of image
    else:
        regions_nuclei = regionprops(labeled_nuclei)
        # Get bounding box of all nuclei
        min_row = min(r.bbox[0] for r in regions_nuclei)
        min_col = min(r.bbox[1] for r in regions_nuclei)
        max_row = max(r.bbox[2] for r in regions_nuclei)
        max_col = max(r.bbox[3] for r in regions_nuclei)
        
        # Compute nuclei centroid (for later component selection)
        total_area = sum(r.area for r in regions_nuclei)
        nuclei_centroid = (
            sum(r.centroid[0] * r.area for r in regions_nuclei) / total_area,
            sum(r.centroid[1] * r.area for r in regions_nuclei) / total_area
        )
    
    # Step 2: Define ROI around nuclei bounding box
    h, w = actin_image.shape
    roi_min_row = max(0, min_row - roi_margin)
    roi_min_col = max(0, min_col - roi_margin)
    roi_max_row = min(h, max_row + roi_margin)
    roi_max_col = min(w, max_col + roi_margin)
    
    # Step 3: Extract Actin ROI
    actin_roi = actin_image[roi_min_row:roi_max_row, roi_min_col:roi_max_col]
    
    # Step 4: Process Actin ROI
    # Apply stronger Gaussian blur for better denoising (as suggested by Gemini)
    actin_roi_blurred = ndimage.gaussian_filter(actin_roi, sigma=blur_sigma)
    
    # Apply median filter for additional noise reduction (as suggested by Gemini)
    from skimage.filters import median
    actin_roi_denoised = median(actin_roi_blurred, footprint=disk(3))
    
    # Contrast enhancement: normalize to full dynamic range
    actin_min = actin_roi_denoised.min()
    actin_max = actin_roi_denoised.max()
    if actin_max > actin_min:
        actin_roi_normalized = (actin_roi_denoised - actin_min) / (actin_max - actin_min)
    else:
        actin_roi_normalized = actin_roi_denoised
    
    # Adaptive thresholding: try multiple strategies (as suggested by Gemini)
    # Strategy 1: Try Otsu on ROI
    threshold = None
    try:
        threshold = filters.threshold_otsu(actin_roi_normalized)
    except:
        pass
    
    # Strategy 2: If Otsu fails or gives unreasonable threshold, use percentile
    if threshold is None or threshold < 0.1 or threshold > 0.9:
        # Use percentile-based threshold (more conservative to avoid background noise)
        # Try multiple percentiles to find a good balance
        for percentile in [50, 45, 40, 35]:
            threshold_candidate = np.percentile(actin_roi_normalized, percentile)
            # Check if this threshold gives reasonable segmentation
            pixels_above = (actin_roi_normalized > threshold_candidate).sum()
            roi_area = actin_roi_normalized.size
            # Want between 10% and 60% of ROI to be cell (avoid too much background)
            if 0.1 * roi_area < pixels_above < 0.6 * roi_area:
                threshold = threshold_candidate
                break
        
        # If no good threshold found, use default
        if threshold is None or threshold < 0.1 or threshold > 0.9:
            threshold = np.percentile(actin_roi_normalized, 40)
    
    # Strategy 3: If still no good threshold, try lower percentile
    if threshold is None or threshold < 0.05:
        # Fallback: use a very low threshold to ensure we capture something
        threshold = np.percentile(actin_roi_normalized, 30)
    
    # Create binary mask
    binary_roi = actin_roi_normalized > threshold
    
    # Morphological operations
    selem = disk(closing_radius)
    binary_roi = closing(binary_roi, selem)
    
    # Remove small objects (increase min_size to filter out more noise, as suggested by Gemini)
    binary_roi = remove_small_objects(binary_roi, min_size=max(min_size, 500))
    
    # Fill holes
    binary_roi = ndimage.binary_fill_holes(binary_roi)
    
    # Step 5: Label components and pick the one closest to nuclei centroid
    labeled_roi = label(binary_roi)
    
    if labeled_roi.max() == 0:
        # No components found - try lower threshold as fallback
        threshold_fallback = np.percentile(actin_roi_normalized, 25)
        binary_roi = actin_roi_normalized > threshold_fallback
        binary_roi = remove_small_objects(binary_roi, min_size=min_size // 2)
        labeled_roi = label(binary_roi)
        
        if labeled_roi.max() == 0:
            # Still no components, return empty mask
            cell_mask = np.zeros_like(actin_image, dtype=bool)
            return cell_mask
    
    regions_roi = regionprops(labeled_roi)
    
    # Find component whose centroid is closest to nuclei centroid
    # Convert nuclei centroid to ROI coordinates
    nuclei_centroid_roi = (
        nuclei_centroid[0] - roi_min_row,
        nuclei_centroid[1] - roi_min_col
    )
    
    min_distance = float('inf')
    best_label = None
    
    # Prefer larger components that are close to nuclei
    for region in regions_roi:
        # Compute distance from region centroid to nuclei centroid
        dist = np.sqrt(
            (region.centroid[0] - nuclei_centroid_roi[0])**2 +
            (region.centroid[1] - nuclei_centroid_roi[1])**2
        )
        # Prefer components that are both close to nuclei and reasonably large
        if dist < min_distance:
            # If closer, always prefer it
            min_distance = dist
            best_label = region.label
        elif dist < min_distance * 1.5 and region.area > 500:
            # If somewhat close and large, prefer it
            min_distance = dist
            best_label = region.label
    
    # If no component found, use largest component as fallback
    if best_label is None and len(regions_roi) > 0:
        best_label = max(regions_roi, key=lambda r: r.area).label
    
    # Step 6: Create cell mask in full image coordinates
    cell_mask = np.zeros_like(actin_image, dtype=bool)
    if best_label is not None:
        cell_mask_roi = labeled_roi == best_label
        cell_mask[roi_min_row:roi_max_row, roi_min_col:roi_max_col] = cell_mask_roi
    
    # Step 7: Refinement - smooth boundary (as suggested by Gemini)
    # Apply stronger morphological operations to smooth jagged boundaries
    # First opening to remove small protrusions
    cell_mask = opening(cell_mask, disk(3))
    # Then closing to fill gaps and smooth contours
    cell_mask = closing(cell_mask, disk(5))
    # Additional closing for very jagged boundaries
    cell_mask = closing(cell_mask, disk(3))
    
    # If cell mask is too small (likely wrong), try to expand it
    cell_area = cell_mask.sum()
    if cell_area < 500:  # Very small cell mask, likely incorrect
        # Try dilation to expand the mask
        from skimage.morphology import dilation
        cell_mask = dilation(cell_mask, disk(10))
        # Then close to smooth
        cell_mask = closing(cell_mask, disk(5))
    
    return cell_mask


def segment_cell(
    actin_image: np.ndarray,
    bounding_box: Optional[Tuple[int, int, int, int]] = None,
    blur_sigma: float = DEFAULT_PARAMS['blur_sigma'],
    min_size: int = DEFAULT_PARAMS['min_size_cell'],
    closing_radius: int = DEFAULT_PARAMS['closing_radius_cell']
) -> np.ndarray:
    """
    DEPRECATED: Old cell segmentation method.
    
    This method is kept for backward compatibility but should not be used.
    Use segment_cell_nuclei_anchored() instead.
    
    The old approach fails because:
    - Mask hugs background speckle and borders; misses true cortex
    - Global Otsu thresholding doesn't work with non-uniform illumination
    - Largest component often ends up being noisy background, not the cell
    """
    # Optionally crop to bounding box
    if bounding_box is not None:
        min_row, min_col, max_row, max_col = bounding_box
        actin_image = actin_image[min_row:max_row, min_col:max_col]
    
    # Normalize to 0-1 range if needed
    if actin_image.max() > 1.0:
        actin_image = actin_image.astype(np.float32) / 255.0
    
    # Apply Gaussian blur to reduce noise
    blurred = ndimage.gaussian_filter(actin_image, sigma=blur_sigma)
    
    # Otsu thresholding to obtain initial binary mask
    threshold = filters.threshold_otsu(blurred)
    binary = blurred > threshold
    
    # Morphological closing to fill small gaps along the cortex
    selem = disk(closing_radius)
    binary = closing(binary, selem)
    
    # Remove small objects below area threshold
    binary = remove_small_objects(binary, min_size=min_size)
    
    # Fill holes within the main region
    binary = ndimage.binary_fill_holes(binary)
    
    # Label connected components
    labeled = label(binary)
    
    # If multiple components exist, keep the largest one
    if labeled.max() > 0:
        regions = regionprops(labeled)
        largest_region = max(regions, key=lambda r: r.area)
        cell_mask = labeled == largest_region.label
    else:
        # No regions found, return empty mask
        cell_mask = np.zeros_like(binary, dtype=bool)
    
    return cell_mask


def segment_nuclei_robust(
    nuclei_image: np.ndarray,
    cell_mask: np.ndarray,
    cell_area_pixels: float,
    blur_sigma: float = 1.5,
    opening_radius: int = 1,
    nuclear_area_min_fraction: float = DEFAULT_PARAMS['nuclear_area_min_fraction'],
    nuclear_area_max_fraction: float = DEFAULT_PARAMS['nuclear_area_max_fraction'],
    use_watershed: bool = True
) -> Tuple[np.ndarray, Dict]:
    """
    Robust nuclear segmentation tuned for single-cell fields.
    
    This function implements a principled nuclear segmentation pipeline that:
    1. Uses adaptive size filtering based on cell area (nuclei should be 1-50% of cell area)
    2. Applies watershed separation for touching nuclei
    3. Restricts all nuclei to be within the cell mask
    4. Filters components by area range derived from biological constraints
    
    The size filters are chosen based on biological knowledge:
    - Minimum: ~1% of cell area (very small speckles are noise)
    - Maximum: ~50% of cell area (a single nucleus shouldn't exceed half the cell)
    - Typical endothelial cell nuclei: 5-20% of cell area
    
    Parameters
    ----------
    nuclei_image : np.ndarray
        Grayscale Nuclei channel image (2D array, normalized 0-1)
    cell_mask : np.ndarray
        Binary mask of the cell (required for restricting nuclei)
    cell_area_pixels : float
        Total cell area in pixels (used for adaptive size filtering)
    blur_sigma : float
        Standard deviation for Gaussian blur (default 1.5, smaller than cell segmentation)
    opening_radius : int
        Radius for morphological opening (removes small noise)
    nuclear_area_min_fraction : float
        Minimum nuclear area as fraction of cell area (default 0.01 = 1%)
    nuclear_area_max_fraction : float
        Maximum nuclear area as fraction of cell area (default 0.5 = 50%)
    use_watershed : bool
        Whether to use watershed separation for touching nuclei (default True)
    
    Returns
    -------
    tuple
        (nucleus_mask, nuclei_props)
        - nucleus_mask: Binary mask of all nuclei (union of all nuclear objects)
        - nuclei_props: Dictionary with 'count', 'areas' (list), and 'warnings' (list)
    """
    # Normalize to 0-1 range if needed
    if nuclei_image.max() > 1.0:
        nuclei_image = nuclei_image.astype(np.float32) / 255.0
    
    # Apply strong denoising first (as suggested by Gemini)
    # Use median filter for noise reduction
    from skimage.filters import median
    nuclei_denoised = median(nuclei_image, footprint=disk(5))
    
    # Apply Gaussian blur (as suggested by Gemini: sigma=1-2 pixels for smoothing)
    # Use slightly larger sigma for better noise reduction
    blur_sigma_effective = max(blur_sigma, 1.5)  # At least 1.5 pixels
    blurred = ndimage.gaussian_filter(nuclei_denoised, sigma=blur_sigma_effective)
    
    # Restrict processing to cell interior only (but be less restrictive)
    # If cell mask is too small or wrong, use a more permissive approach
    cell_area_actual = cell_mask.sum()
    if cell_area_actual < 1000:  # Cell mask might be wrong, be more permissive
        # Use a dilated cell mask or just use the full image
        from skimage.morphology import dilation
        cell_mask_dilated = dilation(cell_mask, disk(20))
        cell_pixels = blurred[cell_mask_dilated]
        use_mask = cell_mask_dilated
    else:
        cell_pixels = blurred[cell_mask]
        use_mask = cell_mask
    
    if len(cell_pixels) == 0:
        # Fallback: use full image
        cell_pixels = blurred.flatten()
        use_mask = np.ones_like(cell_mask, dtype=bool)
    
    # Adaptive thresholding: try multiple strategies (as suggested by Gemini)
    threshold = None
    
    # Strategy 1: Try Otsu on cell region
    try:
        threshold = filters.threshold_otsu(cell_pixels)
        # Check if threshold is reasonable
        if threshold > np.percentile(cell_pixels, 95) or threshold < np.percentile(cell_pixels, 5):
            threshold = None  # Threshold unreasonable, try other methods
    except:
        pass
    
    # Strategy 2: Use percentile-based threshold (more conservative)
    if threshold is None:
        # Try multiple percentiles to find a good threshold
        # Start with lower percentiles to capture more nuclei (as suggested by Gemini)
        for percentile in [80, 75, 70, 65, 60]:  # Lower percentiles to catch faint nuclei
            threshold_candidate = np.percentile(cell_pixels, percentile)
            # Check if this threshold captures reasonable amount of pixels
            pixels_above = (blurred > threshold_candidate).sum()
            # More permissive: between 50 and 50% of cell pixels
            if 50 < pixels_above < len(cell_pixels) * 0.5:
                threshold = threshold_candidate
                break
        
        # If still no threshold, use default (lower percentile)
        if threshold is None:
            threshold = np.percentile(cell_pixels, 75)  # Lower default
    
    # Create binary mask
    binary = blurred > threshold
    
    # Restrict to mask (but be less restrictive if cell mask is wrong)
    binary = binary & use_mask
    
    # Morphological opening to remove small noise and smooth boundaries (as suggested by Gemini)
    # Use larger opening radius for better smoothing
    selem = disk(max(opening_radius, 3))  # At least 3 pixels
    binary = opening(binary, selem)
    
    # Fill all holes within nuclei (as suggested by Gemini)
    binary = ndimage.binary_fill_holes(binary)
    
    # Apply additional smoothing with opening/closing (as suggested by Gemini)
    binary = opening(binary, disk(2))  # Small opening to smooth
    binary = closing(binary, disk(2))  # Small closing to fill gaps
    
    # Restrict again to cell mask (but be less restrictive if cell mask is wrong)
    if cell_mask.sum() > 1000:  # Only restrict if cell mask is reasonable
        binary = binary & use_mask
    # Otherwise, keep the full binary mask
    
    # Calculate adaptive size thresholds based on cell area
    min_nuclear_area = int(cell_area_pixels * nuclear_area_min_fraction)
    max_nuclear_area = int(cell_area_pixels * nuclear_area_max_fraction)
    
    # Remove objects that are too small (noise) or too large (artifacts)
    # First remove small objects
    binary = remove_small_objects(binary, min_size=min_nuclear_area)
    
    # Label components to check sizes
    labeled = label(binary)
    regions = regionprops(labeled)
    
    # Filter by maximum size
    valid_labels = []
    for region in regions:
        if min_nuclear_area <= region.area <= max_nuclear_area:
            valid_labels.append(region.label)
    
    # Create filtered binary mask
    if valid_labels:
        binary_filtered = np.isin(labeled, valid_labels)
    else:
        binary_filtered = np.zeros_like(binary, dtype=bool)
    
    # Apply watershed separation if requested and if we have multiple potential nuclei
    if use_watershed and binary_filtered.sum() > 0:
        # Use distance transform to find peaks
        distance = ndimage.distance_transform_edt(binary_filtered)
        
        # Find local maxima (potential nuclear centers)
        local_maxima = feature.peak_local_max(
            distance, 
            min_distance=10,  # Minimum distance between peaks (pixels)
            threshold_abs=distance.max() * 0.3,  # At least 30% of max distance
            exclude_border=False
        )
        
        # Create markers from local maxima
        # peak_local_max returns coordinates as a tuple (row_indices, col_indices)
        markers = np.zeros_like(distance, dtype=np.int32)
        if isinstance(local_maxima, tuple) and len(local_maxima) == 2:
            # Tuple of (row_indices, col_indices)
            row_indices, col_indices = local_maxima
            if len(row_indices) > 0:
                # Set markers at peak locations
                for i, (r, c) in enumerate(zip(row_indices, col_indices)):
                    markers[r, c] = i + 1
                
                # Apply watershed
                labels = watershed(-distance, markers, mask=binary_filtered)
                
                # Update binary mask from watershed labels
                binary_filtered = labels > 0
        else:
            # No peaks found or unexpected format, use original binary
            pass
    
    # Final labeling and area calculation
    labeled_final = label(binary_filtered)
    regions_final = regionprops(labeled_final)
    
    # Extract nuclear properties
    nuclei_areas = [r.area for r in regions_final if r.area > 0]
    nuclear_count = len(nuclei_areas)
    
    # Create union mask
    nucleus_mask = labeled_final > 0
    
    # Collect warnings
    warnings_list = []
    if nuclear_count == 0:
        warnings_list.append(f"No nuclei detected (threshold={threshold:.3f}, min_area={min_nuclear_area}, max_area={max_nuclear_area})")
    elif nuclear_count > 4:
        warnings_list.append(f"Unusually high nuclear count: {nuclear_count} (expected 1-4)")
    
    # Check total nuclear area
    total_nuclear_area = sum(nuclei_areas)
    if total_nuclear_area > cell_area_pixels:
        warnings_list.append(f"Total nuclear area ({total_nuclear_area:.0f}) exceeds cell area ({cell_area_pixels:.0f})")
    
    nuclei_props = {
        'count': nuclear_count,
        'areas': nuclei_areas,
        'warnings': warnings_list,
        'min_area': min_nuclear_area,
        'max_area': max_nuclear_area,
        'threshold': threshold
    }
    
    return nucleus_mask, nuclei_props


def segment_nuclei(
    nuclei_image: np.ndarray,
    cell_mask: Optional[np.ndarray] = None,
    blur_sigma: float = DEFAULT_PARAMS['blur_sigma'],
    min_size: int = DEFAULT_PARAMS['min_size_nucleus'],
    opening_radius: int = DEFAULT_PARAMS['opening_radius_nucleus']
) -> Tuple[np.ndarray, Dict]:
    """
    DEPRECATED: Legacy nuclear segmentation function.
    
    This function is kept for backward compatibility but should not be used
    for new analyses. Use segment_nuclei_robust() instead.
    
    See segment_nuclei_robust() for the improved implementation.
    """
    # If cell_mask is provided, use robust segmentation
    if cell_mask is not None:
        cell_area = np.sum(cell_mask)
        return segment_nuclei_robust(
            nuclei_image, 
            cell_mask, 
            cell_area,
            blur_sigma=blur_sigma,
            opening_radius=opening_radius
        )
    
    # Fallback to old method if no cell_mask
    # Normalize to 0-1 range if needed
    if nuclei_image.max() > 1.0:
        nuclei_image = nuclei_image.astype(np.float32) / 255.0
    
    # Apply Gaussian blur
    blurred = ndimage.gaussian_filter(nuclei_image, sigma=blur_sigma)
    
    # Otsu thresholding
    threshold = filters.threshold_otsu(blurred)
    binary = blurred > threshold
    
    # Morphological opening to clean noise
    selem = disk(opening_radius)
    binary = opening(binary, selem)
    
    # Morphological closing to fill small gaps
    binary = closing(binary, selem)
    
    # Remove small objects
    binary = remove_small_objects(binary, min_size=min_size)
    
    # Fill holes
    binary = ndimage.binary_fill_holes(binary)
    
    # Label connected components
    labeled = label(binary)
    
    # Extract properties of each nuclear component
    regions = regionprops(labeled)
    nuclei_areas = [r.area for r in regions if r.area > 0]
    nuclear_count = len(nuclei_areas)
    
    # Create union mask of all nuclei
    nucleus_mask = labeled > 0
    
    nuclei_props = {
        'count': nuclear_count,
        'areas': nuclei_areas,
        'warnings': []
    }
    
    return nucleus_mask, nuclei_props


def create_cytoplasm_mask(
    cell_mask: np.ndarray,
    nucleus_mask: np.ndarray,
    erosion_radius: int = DEFAULT_PARAMS['nucleus_erosion_radius']
) -> np.ndarray:
    """
    Create cytoplasm mask by subtracting nucleus from cell.
    
    Optionally erodes the nucleus mask slightly before subtracting to avoid
    mixing nuclear pixels into cytoplasm measurements.
    
    Parameters
    ----------
    cell_mask : np.ndarray
        Binary mask of the cell
    nucleus_mask : np.ndarray
        Binary mask of the nuclei
    erosion_radius : int
        Radius for eroding nucleus mask before subtraction (0 = no erosion)
    
    Returns
    -------
    np.ndarray
        Binary mask of the cytoplasm (cell AND NOT nucleus)
    """
    # Ensure masks have same shape
    if cell_mask.shape != nucleus_mask.shape:
        raise ValueError(
            f"Shape mismatch: cell_mask {cell_mask.shape} vs nucleus_mask {nucleus_mask.shape}"
        )
    
    # Optionally erode nucleus mask to avoid edge contamination
    if erosion_radius > 0:
        selem = disk(erosion_radius)
        nucleus_mask_eroded = morphology.binary_erosion(nucleus_mask, selem)
    else:
        nucleus_mask_eroded = nucleus_mask
    
    # Cytoplasm = cell AND NOT nucleus
    cytoplasm_mask = cell_mask & (~nucleus_mask_eroded)
    
    return cytoplasm_mask
