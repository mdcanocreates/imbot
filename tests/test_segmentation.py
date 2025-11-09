"""
Unit tests for segmentation functions.
"""

import numpy as np
import pytest
from skimage.draw import ellipse
from image_analysis.segmentation import (
    segment_cell, segment_nuclei, create_cytoplasm_mask
)


def test_segment_cell_simple_ellipse():
    """Test cell segmentation on a synthetic ellipse."""
    # Create a synthetic binary image with an ellipse
    img = np.zeros((200, 200), dtype=np.float32)
    rr, cc = ellipse(100, 100, 50, 30, shape=img.shape)  # Center at (100, 100), radii 50, 30
    img[rr, cc] = 1.0
    
    # Add some noise
    noise = np.random.normal(0, 0.1, img.shape)
    img = np.clip(img + noise, 0, 1)
    
    # Scale to 0-255 range
    img = (img * 255).astype(np.uint8)
    
    # Segment cell
    cell_mask = segment_cell(img)
    
    # Check that we get a mask
    assert cell_mask.dtype == bool
    assert cell_mask.shape == img.shape
    
    # Check that the mask has reasonable area (should be close to ellipse area)
    # Ellipse area ≈ π * a * b = π * 50 * 30 ≈ 4712 pixels
    mask_area = np.sum(cell_mask)
    expected_area = np.pi * 50 * 30
    
    # Allow some tolerance (segmentation may not be perfect)
    assert mask_area > expected_area * 0.7
    assert mask_area < expected_area * 1.3


def test_segment_nuclei_multiple_blobs():
    """Test nuclear segmentation with multiple nuclei."""
    # Create a synthetic image with two bright nuclei
    img = np.zeros((200, 200), dtype=np.float32)
    
    # First nucleus (circle at (50, 50))
    rr1, cc1 = ellipse(50, 50, 15, 15, shape=img.shape)
    img[rr1, cc1] = 1.0
    
    # Second nucleus (circle at (150, 150))
    rr2, cc2 = ellipse(150, 150, 20, 20, shape=img.shape)
    img[rr2, cc2] = 1.0
    
    # Add some noise
    noise = np.random.normal(0, 0.1, img.shape)
    img = np.clip(img + noise, 0, 1)
    
    # Scale to 0-255 range
    img = (img * 255).astype(np.uint8)
    
    # Segment nuclei
    nucleus_mask, nuclei_props = segment_nuclei(img)
    
    # Check that we get a mask
    assert nucleus_mask.dtype == bool
    assert nucleus_mask.shape == img.shape
    
    # Check that we detect multiple nuclei
    assert nuclei_props['count'] >= 2  # Should detect at least 2 nuclei
    
    # Check that areas are reasonable
    for area in nuclei_props['areas']:
        assert area > 100  # Each nucleus should have reasonable area


def test_create_cytoplasm_mask():
    """Test cytoplasm mask creation."""
    # Create synthetic masks
    cell_mask = np.zeros((100, 100), dtype=bool)
    cell_mask[20:80, 20:80] = True  # Cell is a square
    
    nucleus_mask = np.zeros((100, 100), dtype=bool)
    nucleus_mask[40:60, 40:60] = True  # Nucleus is a smaller square inside
    
    # Create cytoplasm mask
    cytoplasm_mask = create_cytoplasm_mask(cell_mask, nucleus_mask)
    
    # Check that cytoplasm mask is correct
    assert cytoplasm_mask.dtype == bool
    assert cytoplasm_mask.shape == cell_mask.shape
    
    # Cytoplasm should be cell minus nucleus
    # Cell area: 60x60 = 3600
    # Nucleus area: 20x20 = 400
    # Cytoplasm area should be approximately 3600 - 400 = 3200
    cytoplasm_area = np.sum(cytoplasm_mask)
    expected_area = 3600 - 400
    
    assert cytoplasm_area == expected_area
    
    # Cytoplasm should not overlap with nucleus
    assert np.sum(cytoplasm_mask & nucleus_mask) == 0

