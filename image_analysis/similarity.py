"""
Similarity analysis between cells based on computed metrics.

This module implements:
- Z-score normalization of metrics
- Euclidean distance calculation between cells
- Identification of most similar pair
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def normalize_metrics(
    metrics_df: pd.DataFrame,
    metric_columns: List[str]
) -> pd.DataFrame:
    """
    Normalize metrics using z-score normalization.
    
    For each metric k, compute:
    m_tilde_ik = (m_ik - μ_k) / σ_k
    
    where μ_k is the mean and σ_k is the standard deviation across all cells.
    
    Parameters
    ----------
    metrics_df : pd.DataFrame
        DataFrame with one row per cell and columns for each metric.
        Must have an index or column identifying cells.
    metric_columns : list
        List of column names to normalize
    
    Returns
    -------
    pd.DataFrame
        DataFrame with normalized metrics (same structure as input)
    """
    # Create a copy to avoid modifying the original
    normalized_df = metrics_df.copy()
    
    # Normalize each metric column
    for col in metric_columns:
        if col not in metrics_df.columns:
            raise ValueError(f"Column '{col}' not found in metrics DataFrame")
        
        values = metrics_df[col].values
        mean = np.mean(values)
        std = np.std(values)
        
        # Avoid division by zero
        if std > 1e-10:
            normalized_df[col] = (values - mean) / std
        else:
            # If std is zero, all values are the same, set to 0
            normalized_df[col] = 0.0
    
    return normalized_df


def compute_pairwise_distances(
    metrics_df: pd.DataFrame,
    metric_columns: List[str],
    cell_ids: List[str]
) -> Dict[Tuple[str, str], float]:
    """
    Compute Euclidean distance between each pair of cells.
    
    Distance between cells A and B:
    d(A,B) = sqrt(sum_k (m_tilde_Ak - m_tilde_Bk)²)
    
    Parameters
    ----------
    metrics_df : pd.DataFrame
        DataFrame with normalized metrics (one row per cell)
    metric_columns : list
        List of column names to use for distance calculation
    cell_ids : list
        List of cell identifiers (must match DataFrame index or a column)
    
    Returns
    -------
    dict
        Dictionary mapping (cell_id1, cell_id2) tuples to distances
    """
    # Ensure cell_ids are in the DataFrame
    if isinstance(metrics_df.index, pd.Index):
        # If cell_ids are in the index
        if all(cid in metrics_df.index for cid in cell_ids):
            cell_data = {cid: metrics_df.loc[cid, metric_columns].values 
                        for cid in cell_ids}
        else:
            # Try to find a column with cell IDs
            raise ValueError("Cell IDs not found in DataFrame index")
    else:
        # Assume cell_ids are in a column or index
        cell_data = {cid: metrics_df.loc[cid, metric_columns].values 
                    for cid in cell_ids}
    
    # Compute pairwise distances
    distances = {}
    n = len(cell_ids)
    
    for i in range(n):
        for j in range(i + 1, n):
            cell1 = cell_ids[i]
            cell2 = cell_ids[j]
            
            vec1 = cell_data[cell1]
            vec2 = cell_data[cell2]
            
            # Euclidean distance
            distance = np.sqrt(np.sum((vec1 - vec2) ** 2))
            
            # Store both orderings for convenience
            distances[(cell1, cell2)] = distance
            distances[(cell2, cell1)] = distance
    
    return distances


def find_most_similar_pair(
    distances: Dict[Tuple[str, str], float],
    cell_ids: List[str]
) -> Tuple[Tuple[str, str], float]:
    """
    Find the pair of cells with the smallest distance (most similar).
    
    Parameters
    ----------
    distances : dict
        Dictionary mapping (cell_id1, cell_id2) tuples to distances
    cell_ids : list
        List of cell identifiers
    
    Returns
    -------
    tuple
        ((cell_id1, cell_id2), distance) for the most similar pair
    """
    # Find minimum distance
    min_distance = float('inf')
    most_similar_pair = None
    
    n = len(cell_ids)
    for i in range(n):
        for j in range(i + 1, n):
            cell1 = cell_ids[i]
            cell2 = cell_ids[j]
            
            if (cell1, cell2) in distances:
                dist = distances[(cell1, cell2)]
                if dist < min_distance:
                    min_distance = dist
                    most_similar_pair = (cell1, cell2)
    
    if most_similar_pair is None:
        raise ValueError("No distances found for the given cell IDs")
    
    return most_similar_pair, min_distance


def analyze_similarity(
    metrics_df: pd.DataFrame,
    metric_columns: List[str],
    cell_ids: List[str]
) -> Dict:
    """
    Complete similarity analysis: normalize, compute distances, find most similar pair.
    
    Parameters
    ----------
    metrics_df : pd.DataFrame
        DataFrame with metrics (one row per cell)
    metric_columns : list
        List of column names to use for similarity analysis
    cell_ids : list
        List of cell identifiers
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'normalized_metrics': DataFrame with normalized metrics
        - 'distances': Dictionary of pairwise distances
        - 'most_similar_pair': Tuple of (cell_id1, cell_id2)
        - 'min_distance': Distance for the most similar pair
    """
    # Normalize metrics
    normalized_df = normalize_metrics(metrics_df, metric_columns)
    
    # Compute pairwise distances
    distances = compute_pairwise_distances(normalized_df, metric_columns, cell_ids)
    
    # Find most similar pair
    most_similar_pair, min_distance = find_most_similar_pair(distances, cell_ids)
    
    return {
        'normalized_metrics': normalized_df,
        'distances': distances,
        'most_similar_pair': most_similar_pair,
        'min_distance': min_distance
    }

