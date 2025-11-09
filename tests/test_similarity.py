"""
Unit tests for similarity analysis.
"""

import numpy as np
import pandas as pd
import pytest
from image_analysis.similarity import (
    normalize_metrics, compute_pairwise_distances,
    find_most_similar_pair, analyze_similarity
)


def test_normalize_metrics():
    """Test z-score normalization of metrics."""
    # Create a simple metrics DataFrame
    data = {
        'cell_area': [1000, 2000, 3000],
        'circularity': [0.5, 0.6, 0.7],
        'aspect_ratio': [1.5, 2.0, 2.5]
    }
    df = pd.DataFrame(data, index=['CellA', 'CellB', 'CellC'])
    
    # Normalize
    normalized = normalize_metrics(df, ['cell_area', 'circularity', 'aspect_ratio'])
    
    # Check that means are approximately 0
    for col in ['cell_area', 'circularity', 'aspect_ratio']:
        mean = normalized[col].mean()
        assert abs(mean) < 1e-10
    
    # Check that standard deviations are approximately 1
    for col in ['cell_area', 'circularity', 'aspect_ratio']:
        std = normalized[col].std()
        assert abs(std - 1.0) < 1e-10


def test_compute_pairwise_distances():
    """Test pairwise distance computation."""
    # Create normalized metrics DataFrame
    data = {
        'cell_area': [0.0, 1.0, -1.0],  # Already normalized
        'circularity': [0.0, 0.0, 0.0]  # All same
    }
    df = pd.DataFrame(data, index=['CellA', 'CellB', 'CellC'])
    
    # Compute distances
    distances = compute_pairwise_distances(
        df, ['cell_area', 'circularity'], ['CellA', 'CellB', 'CellC']
    )
    
    # Check that distances are computed
    assert ('CellA', 'CellB') in distances
    assert ('CellA', 'CellC') in distances
    assert ('CellB', 'CellC') in distances
    
    # Distance between A and B should be 1.0 (only cell_area differs by 1)
    assert abs(distances[('CellA', 'CellB')] - 1.0) < 1e-6
    
    # Distance between A and C should be 1.0 (only cell_area differs by 1)
    assert abs(distances[('CellA', 'CellC')] - 1.0) < 1e-6
    
    # Distance between B and C should be 2.0 (cell_area differs by 2)
    assert abs(distances[('CellB', 'CellC')] - 2.0) < 1e-6


def test_find_most_similar_pair():
    """Test finding the most similar pair."""
    # Create distances where A and B are most similar
    distances = {
        ('CellA', 'CellB'): 1.0,
        ('CellA', 'CellC'): 5.0,
        ('CellB', 'CellC'): 4.0
    }
    
    most_similar, min_dist = find_most_similar_pair(
        distances, ['CellA', 'CellB', 'CellC']
    )
    
    # Should find A and B as most similar
    assert set(most_similar) == {'CellA', 'CellB'}
    assert min_dist == 1.0


def test_analyze_similarity():
    """Test complete similarity analysis."""
    # Create metrics DataFrame where A and B should be most similar
    data = {
        'cell_area': [1000, 1100, 5000],  # A and B similar, C different
        'circularity': [0.5, 0.52, 0.3],  # A and B similar, C different
        'aspect_ratio': [1.5, 1.6, 3.0]   # A and B similar, C different
    }
    df = pd.DataFrame(data, index=['CellA', 'CellB', 'CellC'])
    
    # Analyze similarity
    results = analyze_similarity(
        df, ['cell_area', 'circularity', 'aspect_ratio'],
        ['CellA', 'CellB', 'CellC']
    )
    
    # Check results structure
    assert 'normalized_metrics' in results
    assert 'distances' in results
    assert 'most_similar_pair' in results
    assert 'min_distance' in results
    
    # Most similar pair should be A and B
    most_similar = results['most_similar_pair']
    assert set(most_similar) == {'CellA', 'CellB'}
    
    # Distance between A and B should be smallest
    dist_AB = results['distances'][('CellA', 'CellB')]
    dist_AC = results['distances'][('CellA', 'CellC')]
    dist_BC = results['distances'][('CellB', 'CellC')]
    
    assert dist_AB < dist_AC
    assert dist_AB < dist_BC

