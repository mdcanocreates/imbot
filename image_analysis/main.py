"""
Main pipeline entry point for cell image analysis.

This script processes fluorescence and brightfield images for multiple cells,
computes morphological and cytoskeletal metrics, and performs similarity analysis.

CURRENT LOGIC SUMMARY:

1. Image Loading (io_utils.py):
   - Loads 5 channels per cell: Actin, Microtubules, Nuclei, Combo, BF
   - Automatically resizes images to match most common size
   - Handles non-standard filename patterns (CellB→PhenotypeI, CellC→PhenotypeII)

2. Cell Segmentation (segmentation.py):
   - Uses Actin channel (strong cortical boundaries)
   - Otsu thresholding + morphological closing + hole filling
   - Keeps largest connected component as cell_mask

3. Nuclear Segmentation (segmentation.py):
   - CORRECTED: Uses segment_nuclei_robust() with adaptive size filtering
   - Adaptive thresholds: nuclei must be 1-50% of cell area
   - Watershed separation for touching nuclei
   - Restricted to cell interior only
   - Sanity checks: count 1-4, N:C ratio 0.05-0.7, nuclear_area < cell_area

4. Metrics Computation (metrics.py):
   - Cell area, circularity, aspect ratio (from cell_mask)
   - Nuclear count, nuclear area, N:C ratio (from nucleus_mask)
   - Actin mean intensity, actin anisotropy (from cytoplasm_mask)
   - Microtubule mean intensity (from cytoplasm_mask)

5. Similarity Analysis (similarity.py):
   - Z-score normalization per metric
   - Euclidean distance between cell pairs
   - Identifies most similar pair

6. QC Images (plotting.py):
   - Combo image with cell and nuclear masks
   - Actin image with cell mask
   - Nuclei image with nuclear mask (NEW)

7. Dashboard Generation (generate_dashboard.py):
   - Interactive HTML dashboard with metrics table, charts, and QC images
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd

from image_analysis.io_utils import load_cell_images, normalize_image_sizes
from image_analysis.segmentation import (
    segment_cell_nuclei_anchored, segment_nuclei_robust, create_cytoplasm_mask
)
from image_analysis.metrics import compute_all_metrics
from image_analysis.similarity import analyze_similarity
from image_analysis.plotting import (
    plot_combo_with_masks, plot_actin_with_cell_mask, plot_nuclei_with_nuclear_mask
)
from image_analysis.generate_dashboard import generate_html_dashboard
from image_analysis.gemini_qc import evaluate_segmentation_with_gemini
from image_analysis.sam_wrapper import refine_cell_mask_with_sam


def process_cell(
    cell_id: str,
    data_root: Path,
    output_root: Path,
    pixel_size_um: float = 1.0
) -> Dict[str, float]:
    """
    Process a single cell: load images, segment, compute metrics, save QC images.
    
    Parameters
    ----------
    cell_id : str
        Identifier for the cell (e.g., "CellA")
    data_root : Path
        Root directory containing cell subdirectories
    output_root : Path
        Output directory for QC images
    pixel_size_um : float
        Pixel size in microns per pixel
    
    Returns
    -------
    dict
        Dictionary of computed metrics for this cell
    """
    print(f"Processing {cell_id}...")
    
    # Load images
    images = load_cell_images(cell_id, data_root)
    # Normalize image sizes (resize if needed)
    normalize_image_sizes(images)
    
    # Extract individual channels
    actin_image = images['actin']
    microtubule_image = images['microtubules']
    nuclei_image = images['nuclei']
    combo_image = images['combo']
    
    # Segment cell using nuclei-anchored ROI approach
    print(f"  Segmenting cell using nuclei-anchored ROI approach...")
    cell_mask = segment_cell_nuclei_anchored(actin_image, nuclei_image)
    cell_area_pixels = np.sum(cell_mask)
    print(f"    Cell area: {cell_area_pixels:.0f} pixels")
    
    # Store original cell mask for potential SAM refinement
    original_cell_mask = cell_mask.copy()
    
    # Segment nuclei from Nuclei channel using robust method (now with corrected cell_mask)
    print(f"  Segmenting nuclei from Nuclei channel (robust method)...")
    nucleus_mask, nuclei_props = segment_nuclei_robust(
        nuclei_image, 
        cell_mask, 
        cell_area_pixels
    )
    
    # Log nuclear segmentation results
    print(f"    Nuclear count: {nuclei_props['count']}")
    print(f"    Nuclear areas: {nuclei_props['areas']}")
    if nuclei_props['warnings']:
        for warning in nuclei_props['warnings']:
            print(f"    WARNING: {warning}")
    
    # Create cytoplasm mask
    cytoplasm_mask = create_cytoplasm_mask(cell_mask, nucleus_mask)
    
    # Compute all metrics
    print(f"  Computing metrics...")
    metrics = compute_all_metrics(
        cell_mask=cell_mask,
        nucleus_mask=nucleus_mask,
        cytoplasm_mask=cytoplasm_mask,
        actin_image=actin_image,
        microtubule_image=microtubule_image,
        pixel_size_um=pixel_size_um
    )
    
    # Add cell_id to metrics
    metrics['cell_id'] = cell_id
    
    # Sanity checks for nuclear metrics
    print(f"  Running sanity checks...")
    sanity_checks_passed = True
    
    # Check 1: Nuclear count should be between 1 and 4
    if not (1 <= metrics['nuclear_count'] <= 4):
        print(f"    ⚠️  SANITY CHECK FAILED: nuclear_count={metrics['nuclear_count']} (expected 1-4)")
        sanity_checks_passed = False
    
    # Check 2: N:C ratio should be between 0.05 and 0.7
    nc_ratio = metrics['nc_ratio']
    if not (0.05 <= nc_ratio <= 0.7):
        print(f"    ⚠️  SANITY CHECK WARNING: nc_ratio={nc_ratio:.3f} (expected 0.05-0.7)")
        if nc_ratio > 1.0:
            print(f"    ⚠️  CRITICAL: N:C ratio > 1.0 indicates nuclear area exceeds cell area!")
            sanity_checks_passed = False
    
    # Check 3: Nuclear area must be less than cell area
    if metrics['nuclear_area'] >= metrics['cell_area']:
        print(f"    ⚠️  SANITY CHECK FAILED: nuclear_area ({metrics['nuclear_area']:.0f}) >= cell_area ({metrics['cell_area']:.0f})")
        sanity_checks_passed = False
    
    if sanity_checks_passed:
        print(f"    ✓ All sanity checks passed")
    
    # Generate QC images
    print(f"  Generating QC images...")
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Combo image with masks
    combo_path = output_root / f"{cell_id}_combo_with_masks.png"
    plot_combo_with_masks(
        combo_image=combo_image,
        cell_mask=cell_mask,
        nucleus_mask=nucleus_mask,
        output_path=str(combo_path)
    )
    
    # Actin image with cell mask
    actin_path = output_root / f"{cell_id}_actin_with_cell_mask.png"
    plot_actin_with_cell_mask(
        actin_image=actin_image,
        cell_mask=cell_mask,
        output_path=str(actin_path)
    )
    
    # Nuclei image with nuclear mask
    nuclei_path = output_root / f"{cell_id}_nuclei_with_nuclear_mask.png"
    plot_nuclei_with_nuclear_mask(
        nuclei_image=nuclei_image,
        nucleus_mask=nucleus_mask,
        cell_mask=cell_mask,
        output_path=str(nuclei_path)
    )
    
    # Gemini QC evaluation (for nuclei channel)
    print(f"  Running Gemini QC evaluation...")
    try:
        # Get raw nuclei image path (handle non-standard naming)
        from image_analysis.io_utils import CELL_ID_TO_PREFIX
        filename_prefix = CELL_ID_TO_PREFIX.get(cell_id, cell_id)
        raw_nuclei_path = data_root / cell_id / f"{filename_prefix}_Nuclei.jpg"
        if not raw_nuclei_path.exists():
            # Try alternative naming
            raw_nuclei_path = data_root / cell_id / f"{cell_id}_Nuclei.jpg"
        
        qc_result = evaluate_segmentation_with_gemini(
            cell_id=cell_id,
            raw_image_path=str(raw_nuclei_path),
            overlay_image_path=str(nuclei_path),
            channel="nuclei"
        )
        
        # Log QC results
        cell_score = qc_result.get('cell_mask_score')
        nucleus_score = qc_result.get('nucleus_mask_score')
        
        if cell_score is not None:
            print(f"    Cell mask score: {cell_score:.2f}")
        if nucleus_score is not None:
            print(f"    Nuclear mask score: {nucleus_score:.2f}")
        
        # Check if scores are below threshold
        qc_threshold = 0.7
        cell_mask_needs_refinement = False
        if cell_score is not None and cell_score < qc_threshold:
            print(f"    ⚠️  Cell mask score below threshold ({qc_threshold}) - needs review")
            cell_mask_needs_refinement = True
        if nucleus_score is not None and nucleus_score < qc_threshold:
            print(f"    ⚠️  Nuclear mask score below threshold ({qc_threshold}) - needs review")
        
        # Log issues and suggestions
        if qc_result.get('issues'):
            print(f"    Issues: {', '.join(qc_result['issues'])}")
        if qc_result.get('suggested_ops'):
            print(f"    Suggestions: {', '.join(qc_result['suggested_ops'])}")
        
        # Store QC result in metrics for later saving
        metrics['_gemini_qc'] = qc_result
        
        # SAM refinement if cell mask score is low
        if cell_mask_needs_refinement:
            print(f"  Attempting SAM refinement for {cell_id}...")
            try:
                # Get Actin or Combo image path for SAM
                from image_analysis.io_utils import CELL_ID_TO_PREFIX
                filename_prefix = CELL_ID_TO_PREFIX.get(cell_id, cell_id)
                actin_path = data_root / cell_id / f"{filename_prefix}_Actin.jpg"
                if not actin_path.exists():
                    actin_path = data_root / cell_id / f"{cell_id}_Actin.jpg"
                
                # Try Combo image if Actin not found
                if not actin_path.exists():
                    combo_path = data_root / cell_id / f"{filename_prefix}_Combo.jpg"
                    if not combo_path.exists():
                        combo_path = data_root / cell_id / f"{cell_id}_Combo.jpg"
                    if combo_path.exists():
                        actin_path = combo_path
                
                if actin_path.exists():
                    # Generate seed points from nuclei centroids if available
                    seed_points = None
                    if nucleus_mask.sum() > 0:
                        from skimage.measure import regionprops, label
                        labeled_nuclei = label(nucleus_mask)
                        if labeled_nuclei.max() > 0:
                            regions = regionprops(labeled_nuclei)
                            seed_points = [(int(r.centroid[0]), int(r.centroid[1])) for r in regions]
                    
                    # Save original mask for comparison
                    before_sam_path = output_root / f"{cell_id}_cellmask_before_sam.png"
                    plot_actin_with_cell_mask(
                        actin_image=actin_image,
                        cell_mask=original_cell_mask,
                        output_path=str(before_sam_path)
                    )
                    
                    # Refine mask with SAM
                    refined_mask = refine_cell_mask_with_sam(
                        image_path=str(actin_path),
                        initial_mask=original_cell_mask,
                        points=seed_points
                    )
                    
                    # Save refined mask for comparison
                    after_sam_path = output_root / f"{cell_id}_cellmask_after_sam.png"
                    plot_actin_with_cell_mask(
                        actin_image=actin_image,
                        cell_mask=refined_mask,
                        output_path=str(after_sam_path)
                    )
                    
                    # Re-run Gemini QC on refined mask
                    print(f"    Re-running Gemini QC on SAM-refined mask...")
                    refined_nuclei_path = output_root / f"{cell_id}_nuclei_with_nuclear_mask_refined.png"
                    plot_nuclei_with_nuclear_mask(
                        nuclei_image=nuclei_image,
                        nucleus_mask=nucleus_mask,
                        cell_mask=refined_mask,
                        output_path=str(refined_nuclei_path)
                    )
                    
                    refined_qc_result = evaluate_segmentation_with_gemini(
                        cell_id=f"{cell_id}_SAM_refined",
                        raw_image_path=str(raw_nuclei_path),
                        overlay_image_path=str(refined_nuclei_path),
                        channel="nuclei"
                    )
                    
                    refined_cell_score = refined_qc_result.get('cell_mask_score')
                    if refined_cell_score is not None:
                        print(f"    SAM-refined cell mask score: {refined_cell_score:.2f}")
                    
                    # Replace mask if improved
                    if refined_cell_score is not None and refined_cell_score >= qc_threshold:
                        print(f"    ✓ SAM refinement improved mask (score: {refined_cell_score:.2f} >= {qc_threshold})")
                        cell_mask = refined_mask
                        cell_area_pixels = np.sum(cell_mask)
                        # Recompute metrics with refined mask
                        cytoplasm_mask = create_cytoplasm_mask(cell_mask, nucleus_mask)
                        metrics = compute_all_metrics(
                            cell_mask=cell_mask,
                            nucleus_mask=nucleus_mask,
                            cytoplasm_mask=cytoplasm_mask,
                            actin_image=actin_image,
                            microtubule_image=microtubule_image,
                            pixel_size_um=pixel_size_um
                        )
                        metrics['cell_id'] = cell_id
                        # Update QC result
                        metrics['_gemini_qc'] = refined_qc_result
                        # Regenerate QC images with refined mask
                        plot_combo_with_masks(
                            combo_image=combo_image,
                            cell_mask=cell_mask,
                            nucleus_mask=nucleus_mask,
                            output_path=str(combo_path)
                        )
                        plot_actin_with_cell_mask(
                            actin_image=actin_image,
                            cell_mask=cell_mask,
                            output_path=str(actin_path)
                        )
                        plot_nuclei_with_nuclear_mask(
                            nuclei_image=nuclei_image,
                            nucleus_mask=nucleus_mask,
                            cell_mask=cell_mask,
                            output_path=str(nuclei_path)
                        )
                    else:
                        print(f"    ⚠️  SAM refinement did not improve mask (score: {refined_cell_score:.2f} < {qc_threshold})")
                        print(f"    Using original mask for metrics computation")
                else:
                    print(f"    Warning: Could not find Actin or Combo image for SAM refinement")
            except Exception as e:
                print(f"    Warning: SAM refinement failed: {e}")
                print(f"    Using original mask for metrics computation")
        
    except Exception as e:
        print(f"    Warning: Gemini QC failed: {e}")
        metrics['_gemini_qc'] = None
    
    print(f"  Completed {cell_id}")
    
    return metrics


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(
        description="Quantitative comparison of endothelial cells from fluorescence images"
    )
    parser.add_argument(
        '--data-root',
        type=str,
        default='data',
        help='Root directory containing cell subdirectories (default: data)'
    )
    parser.add_argument(
        '--cell-ids',
        type=str,
        nargs='+',
        default=['CellA', 'CellB', 'CellC'],
        help='List of cell IDs to analyze (default: CellA CellB CellC)'
    )
    parser.add_argument(
        '--pixel-size',
        type=float,
        default=1.0,
        help='Pixel size in microns per pixel (default: 1.0)'
    )
    parser.add_argument(
        '--output-root',
        type=str,
        default='outputs',
        help='Output directory for metrics CSV and QC images (default: outputs)'
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    
    # Validate data root exists
    if not data_root.exists():
        print(f"Error: Data root directory does not exist: {data_root}")
        sys.exit(1)
    
    # Process each cell
    all_metrics = []
    all_gemini_qc = []
    for cell_id in args.cell_ids:
        try:
            metrics = process_cell(
                cell_id=cell_id,
                data_root=data_root,
                output_root=output_root,
                pixel_size_um=args.pixel_size
            )
            
            # Extract Gemini QC results if present
            if '_gemini_qc' in metrics and metrics['_gemini_qc'] is not None:
                all_gemini_qc.append(metrics['_gemini_qc'])
                # Remove from metrics dict (don't save to CSV)
                del metrics['_gemini_qc']
            
            all_metrics.append(metrics)
        except Exception as e:
            print(f"Error processing {cell_id}: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # Combine metrics into DataFrame
    metrics_df = pd.DataFrame(all_metrics)
    
    # Set cell_id as index
    metrics_df.set_index('cell_id', inplace=True)
    
    # Save metrics to CSV
    metrics_path = output_root / 'metrics.csv'
    metrics_df.to_csv(metrics_path)
    print(f"\nMetrics saved to {metrics_path}")
    
    # Save Gemini QC results to JSON
    if all_gemini_qc:
        import json
        qc_path = output_root / 'gemini_qc_results.json'
        with open(qc_path, 'w') as f:
            json.dump(all_gemini_qc, f, indent=2)
        print(f"Gemini QC results saved to {qc_path}")
        
        # Create a summary CSV for QC scores
        qc_summary = []
        for qc in all_gemini_qc:
            qc_summary.append({
                'cell_id': qc.get('cell_id'),
                'channel': qc.get('channel'),
                'cell_mask_score': qc.get('cell_mask_score'),
                'nucleus_mask_score': qc.get('nucleus_mask_score'),
                'needs_review': (
                    (qc.get('cell_mask_score') is not None and qc.get('cell_mask_score') < 0.7) or
                    (qc.get('nucleus_mask_score') is not None and qc.get('nucleus_mask_score') < 0.7)
                )
            })
        qc_summary_df = pd.DataFrame(qc_summary)
        qc_summary_path = output_root / 'gemini_qc_summary.csv'
        qc_summary_df.to_csv(qc_summary_path, index=False)
        print(f"Gemini QC summary saved to {qc_summary_path}")
    
    # Perform similarity analysis
    print("\nPerforming similarity analysis...")
    
    # Select metrics for similarity analysis (exclude cell_id)
    metric_columns = [
        'cell_area', 'circularity', 'aspect_ratio', 'nuclear_count',
        'nuclear_area', 'nc_ratio', 'actin_mean_intensity',
        'actin_anisotropy', 'mtub_mean_intensity'
    ]
    
    # Ensure all columns exist
    available_columns = [col for col in metric_columns if col in metrics_df.columns]
    if len(available_columns) < len(metric_columns):
        missing = set(metric_columns) - set(available_columns)
        print(f"Warning: Missing metric columns: {missing}")
    
    similarity_results = analyze_similarity(
        metrics_df=metrics_df,
        metric_columns=available_columns,
        cell_ids=args.cell_ids
    )
    
    # Print results
    print("\n" + "="*60)
    print("CELL METRICS")
    print("="*60)
    for cell_id in args.cell_ids:
        if cell_id in metrics_df.index:
            print(f"\n{cell_id}:")
            for col in metrics_df.columns:
                value = metrics_df.loc[cell_id, col]
                print(f"  {col}: {value:.4f}")
    
    print("\n" + "="*60)
    print("NORMALIZED METRIC VECTORS")
    print("="*60)
    normalized_df = similarity_results['normalized_metrics']
    for cell_id in args.cell_ids:
        if cell_id in normalized_df.index:
            values = normalized_df.loc[cell_id, available_columns].values
            print(f"\nm̃_{cell_id} = {values}")
    
    print("\n" + "="*60)
    print("PAIRWISE DISTANCES")
    print("="*60)
    distances = similarity_results['distances']
    n = len(args.cell_ids)
    for i in range(n):
        for j in range(i + 1, n):
            cell1 = args.cell_ids[i]
            cell2 = args.cell_ids[j]
            if (cell1, cell2) in distances:
                dist = distances[(cell1, cell2)]
                print(f"d({cell1}, {cell2}) = {dist:.4f}")
    
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    most_similar_pair = similarity_results['most_similar_pair']
    min_distance = similarity_results['min_distance']
    
    # Find the outlier (the cell not in the most similar pair)
    outlier = [cid for cid in args.cell_ids if cid not in most_similar_pair][0]
    
    print(f"\nCells {most_similar_pair[0]} and {most_similar_pair[1]} are most similar")
    print(f"(distance = {min_distance:.4f})")
    print(f"Cell {outlier} is the outlier based on the chosen metrics.")
    print("="*60 + "\n")
    
    # Generate HTML dashboard
    print("Generating HTML dashboard...")
    try:
        gemini_qc_path = output_root / 'gemini_qc_results.json'
        generate_html_dashboard(
            metrics_csv_path=metrics_path,
            output_dir=output_root,
            qc_images_dir=output_root,
            gemini_qc_path=gemini_qc_path if gemini_qc_path.exists() else None
        )
        print(f"\n✓ Dashboard generated: {output_root / 'dashboard.html'}")
        print(f"  Open this file in your browser to view the results!")
    except Exception as e:
        print(f"Warning: Could not generate dashboard: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

