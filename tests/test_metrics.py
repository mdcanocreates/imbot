"""
Unit tests for metrics computation.
"""

import numpy as np
import pytest
from skimage.draw import ellipse, circle
from image_analysis.metrics import (
    compute_cell_area, compute_circularity, compute_aspect_ratio,
    compute_nuclear_metrics, compute_nc_ratio,
    compute_orientation_order_parameter,
    compute_actin_metrics, compute_microtubule_intensity
)


def test_compute_cell_area():
    """Test cell area computation."""
    # Create a mask with known area
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:80, 20:80] = True  # 60x60 square = 3600 pixels
    
    area = compute_cell_area(mask, pixel_size_um=1.0)
    assert area == 3600.0
    
    # Test with pixel size
    area_um2 = compute_cell_area(mask, pixel_size_um=0.5)
    assert area_um2 == 3600.0 * (0.5 ** 2) == 900.0


def test_compute_circularity_perfect_circle():
    """Test circularity computation on a perfect circle."""
    # Create a perfect circle
    mask = np.zeros((200, 200), dtype=bool)
    rr, cc = circle(100, 100, 50, shape=mask.shape)  # Center at (100, 100), radius 50
    mask[rr, cc] = True
    
    circularity = compute_circularity(mask)
    
    # Perfect circle should have circularity close to 1.0
    assert circularity > 0.9
    assert circularity <= 1.0


def test_compute_aspect_ratio_ellipse():
    """Test aspect ratio computation on an ellipse."""
    # Create an ellipse with known major and minor axes
    mask = np.zeros((200, 200), dtype=bool)
    rr, cc = ellipse(100, 100, 50, 25, shape=mask.shape)  # Major axis 50, minor axis 25
    mask[rr, cc] = True
    
    aspect_ratio = compute_aspect_ratio(mask)
    
    # Expected aspect ratio ≈ 50/25 = 2.0
    # Allow some tolerance
    assert aspect_ratio > 1.8
    assert aspect_ratio < 2.2


def test_compute_nuclear_metrics():
    """Test nuclear metrics computation."""
    # Create synthetic masks
    cell_mask = np.zeros((100, 100), dtype=bool)
    cell_mask[10:90, 10:90] = True  # Cell
    
    nucleus_mask = np.zeros((100, 100), dtype=bool)
    # Two nuclei
    rr1, cc1 = circle(30, 30, 10, shape=nucleus_mask.shape)
    rr2, cc2 = circle(70, 70, 12, shape=nucleus_mask.shape)
    nucleus_mask[rr1, cc1] = True
    nucleus_mask[rr2, cc2] = True
    
    nuclear_count, nuclear_area = compute_nuclear_metrics(
        nucleus_mask, cell_mask, pixel_size_um=1.0
    )
    
    # Should detect 2 nuclei
    assert nuclear_count == 2
    
    # Areas should be reasonable (approximately π * r² for each)
    assert nuclear_area > 500  # Sum of two circles


def test_compute_nc_ratio():
    """Test N:C ratio computation."""
    cell_area = 10000.0  # µm²
    nuclear_area = 1000.0  # µm²
    
    # Expected: 1000 / (10000 - 1000) = 1000 / 9000 ≈ 0.111
    nc_ratio = compute_nc_ratio(cell_area, nuclear_area)
    
    assert abs(nc_ratio - 0.111) < 0.01


def test_compute_orientation_order_parameter_aligned():
    """Test orientation order parameter on aligned lines."""
    # Create synthetic image with aligned lines
    img = np.zeros((200, 200), dtype=np.float32)
    
    # Create parallel vertical lines
    for i in range(10, 200, 20):
        img[:, i:i+5] = 1.0
    
    # Create mask covering the lines
    mask = np.ones((200, 200), dtype=bool)
    
    # Compute anisotropy
    anisotropy = compute_orientation_order_parameter(img, mask)
    
    # Aligned lines should have high anisotropy
    assert anisotropy > 0.7


def test_compute_orientation_order_parameter_random():
    """Test orientation order parameter on random orientations."""
    # Create synthetic image with random noise
    np.random.seed(42)
    img = np.random.rand(200, 200).astype(np.float32)
    
    # Create mask
    mask = np.ones((200, 200), dtype=bool)
    
    # Compute anisotropy
    anisotropy = compute_orientation_order_parameter(img, mask)
    
    # Random orientations should have low anisotropy
    assert anisotropy < 0.5


def test_compute_actin_metrics():
    """Test actin metrics computation."""
    # Create synthetic actin image
    actin_image = np.random.rand(100, 100).astype(np.float32)
    
    # Create cytoplasm mask
    cytoplasm_mask = np.zeros((100, 100), dtype=bool)
    cytoplasm_mask[20:80, 20:80] = True
    
    actin_mean, actin_anisotropy = compute_actin_metrics(
        actin_image, cytoplasm_mask
    )
    
    # Mean intensity should be in [0, 1]
    assert 0.0 <= actin_mean <= 1.0
    
    # Anisotropy should be in [0, 1]
    assert 0.0 <= actin_anisotropy <= 1.0


def test_compute_microtubule_intensity():
    """Test microtubule intensity computation."""
    # Create synthetic microtubule image
    mtub_image = np.random.rand(100, 100).astype(np.float32)
    
    # Create cytoplasm mask
    cytoplasm_mask = np.zeros((100, 100), dtype=bool)
    cytoplasm_mask[20:80, 20:80] = True
    
    mtub_mean = compute_microtubule_intensity(mtub_image, cytoplasm_mask)
    
    # Mean intensity should be in [0, 1]
    assert 0.0 <= mtub_mean <= 1.0

