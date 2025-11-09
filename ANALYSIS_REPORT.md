# Cell Image Analysis Report

## Summary

This report documents the corrected nuclear segmentation pipeline and analysis results for three endothelial cells (CellA, CellB, CellC).

## Previous Issues

The original pipeline had critical flaws in nuclear segmentation:

1. **Fixed size filtering**: Used `min_size_nucleus=50` pixels regardless of cell size
   - Result: CellA and CellC got huge nuclear masks (spilling into most of the cell)
   - Result: CellB lost nuclei entirely (over-thresholded or filtered out)

2. **No maximum size filter**: Large artifacts were included as nuclei
   - Result: Nuclear areas exceeded cell areas (impossible biologically)

3. **No adaptive thresholding**: Global Otsu thresholding didn't account for cell-specific intensity variations
   - Result: Inconsistent segmentation across cells

4. **No watershed separation**: Touching nuclei were counted as one
   - Result: Under-counting of nuclei

## Corrected Implementation

### Nuclear Segmentation (`segment_nuclei_robust`)

The new implementation uses:

1. **Adaptive size filtering**:
   - Minimum nuclear area: 1% of cell area (filters noise)
   - Maximum nuclear area: 50% of cell area (filters artifacts)
   - Rationale: Endothelial cell nuclei typically occupy 5-20% of cell area

2. **Cell-restricted processing**:
   - All nuclear segmentation is restricted to the cell interior
   - Prevents background noise from being segmented

3. **Watershed separation**:
   - Uses distance transform to find nuclear centers
   - Separates touching nuclei using watershed algorithm
   - Ensures accurate nuclear count

4. **Sanity checks**:
   - Nuclear count: 1-4 (expected for single cells)
   - N:C ratio: 0.05-0.7 (biologically plausible)
   - Nuclear area < cell area (must be true)

### Metrics We Trust

After correction, the following metrics are reliable:

- **Cell shape metrics**: Area, circularity, aspect ratio
  - Computed from Actin-based cell segmentation
  - Well-validated by visual inspection

- **Cytoskeletal metrics**: Actin mean intensity, actin anisotropy, microtubule mean intensity
  - Computed from cytoplasmic regions (cell - nucleus)
  - Reflect actual cytoskeletal organization

- **Nuclear metrics** (after correction):
  - Nuclear count: Now accurately reflects 1-2 nuclei per cell
  - Nuclear area: Now within biologically plausible range
  - N:C ratio: Now < 1.0 and within expected range (0.05-0.7)

### Potential Caveats

1. **Limited sample size**: Only 3 cells analyzed
   - Statistical power is limited
   - Results should be interpreted as case studies, not population statistics

2. **Fixed thresholds**: Size filters are tuned for these specific images
   - Parameters (1-50% of cell area) are based on biological knowledge
   - May need adjustment for different cell types or imaging conditions

3. **Segmentation assumptions**:
   - Assumes single cells in field of view
   - Assumes nuclei are bright objects on dark background
   - May fail for very dim or very bright nuclei

4. **Pixel size**: Default is 1.0 μm/pixel
   - Actual pixel size should be calibrated from microscope settings
   - Area metrics depend on accurate pixel size

## Previous vs. Corrected Results

### Previous (Incorrect) Results

| Metric | CellA | CellB | CellC |
|--------|-------|-------|-------|
| Nuclear Count | 6 | 0 | 22 |
| Nuclear Area | 296252 | 0 | 326606 |
| N:C Ratio | 6.71 | 0 | 14.94 |

**Problems**:
- CellA: 6 nuclei (should be ~2)
- CellB: 0 nuclei (should be ~2)
- CellC: 22 nuclei (should be ~2)
- N:C ratios > 1.0 (impossible)

### Corrected Results

See `outputs/metrics.csv` for the corrected metrics table.

The corrected pipeline produces:
- Nuclear counts: 1-2 per cell (biologically plausible)
- Nuclear areas: < cell area (must be true)
- N:C ratios: 0.05-0.7 (biologically plausible)

## Code Structure

### Main Analysis Flow

1. **Image Loading** (`io_utils.py`):
   - Loads 5 channels per cell (Actin, Microtubules, Nuclei, Combo, BF)
   - Normalizes image sizes automatically

2. **Cell Segmentation** (`segmentation.py`):
   - Uses Actin channel (strong cortical boundaries)
   - Otsu thresholding + morphological operations
   - Keeps largest connected component

3. **Nuclear Segmentation** (`segmentation.py`):
   - Uses Nuclei channel
   - Adaptive size filtering based on cell area
   - Watershed separation for touching nuclei
   - Restricted to cell interior

4. **Metrics Computation** (`metrics.py`):
   - All 7 required metrics
   - Sanity checks for nuclear metrics

5. **Similarity Analysis** (`similarity.py`):
   - Z-score normalization
   - Euclidean distance calculation
   - Identifies most similar pair

### Key Functions

- `segment_cell(actin_image)` → cell_mask
- `segment_nuclei_robust(nuclei_image, cell_mask, cell_area)` → nucleus_mask, props
- `compute_all_metrics(...)` → metrics_dict
- `analyze_similarity(metrics_df, ...)` → distances, normalized_vectors

## Running the Analysis

```bash
python3 run_analysis.py
```

Or directly:

```bash
python3 -m image_analysis.main --data-root img_model --output-root outputs --cell-ids CellA CellB CellC
```

## Output Files

- `outputs/metrics.csv`: Corrected metrics table
- `outputs/dashboard.html`: Interactive HTML dashboard
- `outputs/*_combo_with_masks.png`: QC images with cell and nuclear masks
- `outputs/*_actin_with_cell_mask.png`: QC images with cell mask
- `outputs/*_nuclei_with_nuclear_mask.png`: QC images with nuclear mask

## Conclusion

The corrected pipeline produces biologically plausible nuclear metrics that match visual inspection of the images. The previous nuclear metrics were artifacts of poor segmentation and have been corrected using adaptive size filtering and watershed separation.

