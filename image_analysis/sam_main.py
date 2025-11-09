"""
SAM-first pipeline entry point for cell image analysis.

This script uses Segment Anything Model (SAM) as the PRIMARY method for cell
segmentation, with classical methods used only for nuclei segmentation and refinement.

This is a fork of the classical pipeline (main.py) that allows aggressive use
of SAM without affecting the safer classical pipeline.

Pipeline flow:
1. Load images (Actin, Microtubules, Nuclei, Combo, BF)
2. Use SAM to segment cell from Actin/Combo image (PRIMARY METHOD)
3. Use classical methods to segment nuclei inside SAM cell mask
4. Compute metrics (same as classical pipeline)
5. Perform similarity analysis
6. Generate QC images and dashboard
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd
import os

from image_analysis.io_utils import load_cell_images, normalize_image_sizes, CELL_ID_TO_PREFIX
from image_analysis.segmentation import (
    segment_nuclei_robust, create_cytoplasm_mask
)
from image_analysis.metrics import compute_all_metrics
from image_analysis.similarity import analyze_similarity
from image_analysis.plotting import (
    plot_combo_with_masks, plot_actin_with_cell_mask, plot_nuclei_with_nuclear_mask
)
from image_analysis.generate_dashboard import generate_html_dashboard
from image_analysis.gemini_qc import evaluate_segmentation_with_gemini
from image_analysis.sam_wrapper import sam_segment_cell


def process_cell_sam_first(
    cell_id: str,
    data_root: Path,
    output_root: Path,
    pixel_size_um: float = 1.0
) -> Dict[str, float]:
    """
    Process a single cell using SAM-first approach.
    
    This function:
    1. Uses SAM as the PRIMARY method to segment the cell
    2. Uses classical methods to segment nuclei inside the SAM mask
    3. Computes metrics and generates QC images
    
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
    print(f"Processing {cell_id} (SAM-first approach)...")
    
    # Load images
    images = load_cell_images(cell_id, data_root)
    # Normalize image sizes (resize if needed)
    normalize_image_sizes(images)
    
    # Extract individual channels
    actin_image = images['actin']
    microtubule_image = images['microtubules']
    nuclei_image = images['nuclei']
    combo_image = images['combo']
    
    # Get image paths for SAM
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
    
    # Get nuclei image path
    nuclei_path = data_root / cell_id / f"{filename_prefix}_Nuclei.jpg"
    if not nuclei_path.exists():
        nuclei_path = data_root / cell_id / f"{cell_id}_Nuclei.jpg"
    
    # Step 1: Segment cell using SAM (PRIMARY METHOD)
    print(f"  Segmenting cell using SAM (primary method)...")
    try:
        cell_mask = sam_segment_cell(
            image_path=str(actin_path),
            nuclei_image_path=str(nuclei_path) if nuclei_path.exists() else None
        )
        
        # Resize SAM mask to match normalized image dimensions
        if cell_mask.shape != actin_image.shape:
            from skimage.transform import resize
            cell_mask = resize(cell_mask, actin_image.shape, order=0, preserve_range=True).astype(bool)
        
        cell_area_pixels = np.sum(cell_mask)
        print(f"    Cell area: {cell_area_pixels:.0f} pixels")
    except Exception as e:
        print(f"    ERROR: SAM segmentation failed: {e}")
        print(f"    Falling back to empty mask")
        cell_mask = np.zeros_like(actin_image, dtype=bool)
        cell_area_pixels = 0
    
    # Step 2: Segment nuclei from Nuclei channel using classical method (inside SAM mask)
    print(f"  Segmenting nuclei from Nuclei channel (classical method, inside SAM mask)...")
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
    combo_path = output_root / f"{cell_id}_combo_with_masks_sam.png"
    plot_combo_with_masks(
        combo_image=combo_image,
        cell_mask=cell_mask,
        nucleus_mask=nucleus_mask,
        output_path=str(combo_path)
    )
    
    # Actin image with cell mask
    actin_path_qc = output_root / f"{cell_id}_actin_with_cell_mask_sam.png"
    plot_actin_with_cell_mask(
        actin_image=actin_image,
        cell_mask=cell_mask,
        output_path=str(actin_path_qc)
    )
    
    # Nuclei image with nuclear mask
    nuclei_path_qc = output_root / f"{cell_id}_nuclei_with_nuclear_mask_sam.png"
    plot_nuclei_with_nuclear_mask(
        nuclei_image=nuclei_image,
        nucleus_mask=nucleus_mask,
        cell_mask=cell_mask,
        output_path=str(nuclei_path_qc)
    )
    
    # Gemini QC evaluation (optional, for comparison)
    print(f"  Running Gemini QC evaluation (for comparison)...")
    try:
        raw_nuclei_path = data_root / cell_id / f"{filename_prefix}_Nuclei.jpg"
        if not raw_nuclei_path.exists():
            raw_nuclei_path = data_root / cell_id / f"{cell_id}_Nuclei.jpg"
        
        qc_result = evaluate_segmentation_with_gemini(
            cell_id=f"{cell_id}_SAM_first",
            raw_image_path=str(raw_nuclei_path),
            overlay_image_path=str(nuclei_path_qc),
            channel="nuclei"
        )
        
        # Log QC results
        cell_score = qc_result.get('cell_mask_score')
        nucleus_score = qc_result.get('nucleus_mask_score')
        
        if cell_score is not None:
            print(f"    Cell mask score: {cell_score:.2f}")
        if nucleus_score is not None:
            print(f"    Nuclear mask score: {nucleus_score:.2f}")
        
        # Store QC result in metrics for later saving
        metrics['_gemini_qc'] = qc_result
        
    except Exception as e:
        print(f"    Warning: Gemini QC failed: {e}")
        metrics['_gemini_qc'] = None
    
    print(f"  Completed {cell_id}")
    
    return metrics


def main():
    """Main entry point for SAM-first pipeline."""
    parser = argparse.ArgumentParser(
        description="Cell image analysis pipeline using SAM-first approach",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This pipeline uses Segment Anything Model (SAM) as the PRIMARY method for cell
segmentation. Classical methods are used only for nuclei segmentation inside
the SAM cell mask.

Example:
  python -m image_analysis.sam_main --data-root img_model --output-root outputs_sam
        """
    )
    
    parser.add_argument(
        '--data-root',
        type=str,
        default='img_model',
        help='Root directory containing cell subdirectories (default: img_model)'
    )
    
    parser.add_argument(
        '--output-root',
        type=str,
        default='outputs_sam',
        help='Output directory for results (default: outputs_sam)'
    )
    
    parser.add_argument(
        '--cell-ids',
        nargs='+',
        default=['CellA', 'CellB', 'CellC'],
        help='List of cell IDs to process (default: CellA CellB CellC)'
    )
    
    parser.add_argument(
        '--pixel-size',
        type=float,
        default=1.0,
        help='Pixel size in microns per pixel (default: 1.0)'
    )
    
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Running SAM-first cell image analysis pipeline...")
    print("="*60)
    print(f"Data root: {data_root}")
    print(f"Output root: {output_root}")
    print(f"Cell IDs: {', '.join(args.cell_ids)}")
    print()
    
    # Check SAM availability
    try:
        from image_analysis.sam_wrapper import SAM_AVAILABLE
        if not SAM_AVAILABLE:
            print("ERROR: SAM is not available. Please install segment-anything package.")
            sys.exit(1)
        
        checkpoint_path = os.getenv('SAM_CHECKPOINT_PATH')
        if not checkpoint_path or not Path(checkpoint_path).exists():
            print("ERROR: SAM_CHECKPOINT_PATH not set or checkpoint not found.")
            print("Please set SAM_CHECKPOINT_PATH environment variable.")
            sys.exit(1)
        
        print(f"SAM checkpoint: {checkpoint_path}")
        print()
    except Exception as e:
        print(f"ERROR: Failed to check SAM availability: {e}")
        sys.exit(1)
    
    # Process each cell
    all_metrics = []
    all_gemini_qc = []
    
    for cell_id in args.cell_ids:
        try:
            metrics = process_cell_sam_first(
                cell_id=cell_id,
                data_root=data_root,
                output_root=output_root,
                pixel_size_um=args.pixel_size
            )
            
            # Store Gemini QC separately
            if metrics.get('_gemini_qc'):
                all_gemini_qc.append(metrics['_gemini_qc'])
                del metrics['_gemini_qc']
            
            all_metrics.append(metrics)
            
        except Exception as e:
            print(f"Error processing {cell_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save metrics to CSV
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        metrics_csv = output_root / "metrics_sam.csv"
        df.to_csv(metrics_csv, index=False)
        print(f"\nMetrics saved to {metrics_csv}")
    
    # Save Gemini QC results
    if all_gemini_qc:
        import json
        qc_json = output_root / "gemini_qc_results_sam.json"
        with open(qc_json, 'w') as f:
            json.dump(all_gemini_qc, f, indent=2)
        print(f"Gemini QC results saved to {qc_json}")
        
        # Create summary CSV
        qc_summary = []
        for qc in all_gemini_qc:
            qc_summary.append({
                'cell_id': qc.get('cell_id', ''),
                'channel': qc.get('channel', ''),
                'cell_mask_score': qc.get('cell_mask_score'),
                'nucleus_mask_score': qc.get('nucleus_mask_score')
            })
        qc_df = pd.DataFrame(qc_summary)
        qc_csv = output_root / "gemini_qc_summary_sam.csv"
        qc_df.to_csv(qc_csv, index=False)
        print(f"Gemini QC summary saved to {qc_csv}")
    
    # Perform similarity analysis
    if len(all_metrics) >= 2:
        print("\nPerforming similarity analysis...")
        # Convert list of dicts to DataFrame
        metrics_df = pd.DataFrame(all_metrics)
        
        # Define metric columns (exclude non-metric columns)
        exclude_cols = {'cell_id', '_gemini_qc'}
        metric_columns = [col for col in metrics_df.columns if col not in exclude_cols]
        
        # Get cell IDs
        if 'cell_id' in metrics_df.columns:
            cell_ids = metrics_df['cell_id'].tolist()
        else:
            cell_ids = args.cell_ids
        
        # Set cell_id as index for similarity analysis
        if 'cell_id' in metrics_df.columns:
            metrics_df = metrics_df.set_index('cell_id')
        
        similarity_results = analyze_similarity(
            metrics_df=metrics_df,
            metric_columns=metric_columns,
            cell_ids=cell_ids
        )
        
        # Print results
        print("\n" + "="*60)
        print("CELL METRICS")
        print("="*60)
        for cell_id in cell_ids:
            if cell_id in metrics_df.index:
                row = metrics_df.loc[cell_id]
                print(f"{cell_id}:")
                for col in metric_columns:
                    if col in row:
                        print(f"  {col}: {row[col]:.4f}")
        
        print("\n" + "="*60)
        print("NORMALIZED METRIC VECTORS")
        print("="*60)
        norm_df = similarity_results['normalized_metrics']
        for cell_id in cell_ids:
            if cell_id in norm_df.index:
                row = norm_df.loc[cell_id, metric_columns]
                print(f"m̃_{cell_id} = {row.values}")
        
        print("\n" + "="*60)
        print("PAIRWISE DISTANCES")
        print("="*60)
        distances = similarity_results['distances']
        n = len(cell_ids)
        for i in range(n):
            for j in range(i + 1, n):
                cell1 = cell_ids[i]
                cell2 = cell_ids[j]
                if (cell1, cell2) in distances:
                    print(f"d({cell1}, {cell2}) = {distances[(cell1, cell2)]:.4f}")
        
        print("\n" + "="*60)
        print("CONCLUSION")
        print("="*60)
        most_similar = similarity_results['most_similar_pair']
        min_dist = similarity_results['min_distance']
        print(f"\nCells {most_similar[0]} and {most_similar[1]} are most similar")
        print(f"(distance = {min_dist:.4f})")
    
    # Generate HTML dashboard
    if all_metrics:
        print("\nGenerating HTML dashboard...")
        metrics_csv_path = output_root / "metrics_sam.csv"
        qc_images_dir = output_root
        gemini_qc_path = output_root / "gemini_qc_results_sam.json" if all_gemini_qc else None
        
        generate_html_dashboard(
            metrics_csv_path=metrics_csv_path,
            output_dir=output_root,
            qc_images_dir=qc_images_dir,
            gemini_qc_path=gemini_qc_path
        )
        
        dashboard_path = output_root / "dashboard.html"
        print(f"✓ Dashboard generated: {dashboard_path}")
        print(f"  Open this file in your browser to view the results!")
    
    print("\n" + "="*60)
    print("✓ SAM-first analysis completed successfully!")
    print("="*60)
    print(f"\nResults saved to: {output_root}/")
    print(f"Dashboard available at: {output_root}/dashboard.html")


if __name__ == "__main__":
    main()

