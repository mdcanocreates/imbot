# Cell Image Analysis Pipeline

Final production pipeline for quantitative comparison of endothelial cells using fluorescence and brightfield images.

## Overview

This pipeline uses **SAM-first segmentation** as the primary method for cell segmentation, with classical methods used for nuclei segmentation inside the SAM masks. The pipeline computes 6-7 robust metrics per cell and performs similarity analysis using z-score normalization and Euclidean distance.

## Pipeline Architecture

### Segmentation Strategy (Frozen)

1. **Cell Mask (Primary - SAM)**
   - Input: Actin image only
   - Method: Segment Anything Model (SAM)
   - Preprocessing: Light Gaussian/median filter
   - Postprocessing: `remove_small_objects`, `binary_closing`, `binary_opening`, `binary_fill_holes`

2. **Nuclear Mask (Classical, inside cell)**
   - Input: Nuclei channel, constrained to SAM cell mask
   - Method: Adaptive thresholding (Sauvola/Niblack) + watershed
   - Postprocessing: Fixed pixel bounds (300-5000 px), distance transform + watershed for touching nuclei

3. **Cytoplasm Mask**
   - `cytoplasm_mask = cell_mask & ~nuclear_mask`

### Metrics (6-7 Core Metrics)

1. **Cell area** - from cell_mask
2. **Cell circularity** - (4πA / P²)
3. **Aspect ratio** - major / minor axis
4. **Actin mean intensity (cytoplasm)** - mean actin in cytoplasm_mask
5. **Actin anisotropy** - orientation order parameter from gradients
6. **Microtubule mean intensity (cytoplasm)** - mean microtubule intensity
7. **N:C area ratio** (optional, if nuclear segmentation is reliable)

### Similarity Analysis

- Z-score normalization per metric across all cells
- Euclidean distance between cell pairs
- Identification of most similar pair

## Installation

### Prerequisites

- Python 3.12 (for SAM support)
- SAM checkpoint file (`sam_vit_b_01ec64.pth`)

### Setup

1. Clone the repository:
```bash
git clone git@github.com:mdcanocreates/imbot.git
cd imbot
```

2. Create virtual environment:
```bash
python3.12 -m venv venv_sam
source venv_sam/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set SAM checkpoint path:
```bash
export SAM_CHECKPOINT_PATH="/path/to/sam_vit_b_01ec64.pth"
```

## Usage

### Run Streamlit App (Recommended)

The Streamlit app will automatically download the cell images from Dropbox on first run if `img_model/` doesn't exist:

```bash
source venv_sam/bin/activate
streamlit run sam_refine_ui.py
```

The dataset will be automatically downloaded to `img_model/` if it's not already present. No manual download needed!

### Run Final Production Pipeline

```bash
source venv_sam/bin/activate
export SAM_CHECKPOINT_PATH="/path/to/sam_vit_b_01ec64.pth"
python3 final_analysis.py --data-root img_model --output-root final_outputs
```

### Output Files

- `final_outputs/final_metrics.csv` - All computed metrics per cell
- `final_outputs/similarity_results.csv` - Pairwise distances
- `final_outputs/*_actin_with_cell_mask.png` - QC images (Actin + cell mask)
- `final_outputs/*_nuclei_with_nuclear_mask.png` - QC images (Nuclei + nuclear mask + cell outline)
- `final_outputs/*_combo_with_masks.png` - QC images (Combo + both masks)

## Results Summary

The pipeline processes three cells (CellA, CellB, CellC) and computes:

- **Cell metrics**: Area, circularity, aspect ratio, actin intensity, actin anisotropy, microtubule intensity
- **Similarity analysis**: Z-score normalized vectors and pairwise Euclidean distances
- **Most similar pair**: Identified based on minimum distance

## Project Structure

```
imbot/
├── final_analysis.py          # Final production pipeline entry point
├── image_analysis/
│   ├── io_utils.py            # Image loading and normalization
│   ├── segmentation.py        # Classical nuclei segmentation
│   ├── sam_wrapper.py         # SAM integration
│   ├── metrics.py             # Metric computation
│   ├── similarity.py          # Similarity analysis
│   └── plotting.py            # QC image generation
├── segment-anything-main/     # SAM repository
├── tests/                      # Unit tests
└── requirements.txt            # Python dependencies
```

## Key Features

- **SAM-first segmentation**: Uses Segment Anything Model as primary cell segmenter
- **Classical refinement**: Nuclei segmentation using adaptive thresholding and watershed
- **Robust metrics**: 6-7 core metrics with biological relevance
- **Similarity analysis**: Z-score normalization and Euclidean distance
- **QC visualization**: Automatic generation of overlay images for validation

## Limitations

- Small sample size (n=3 cells)
- Segmentation accuracy depends on image quality
- Nuclear segmentation may fail for low-contrast images
- SAM requires GPU for optimal performance (CPU fallback available)

## Citation

If you use this pipeline, please cite:
- Segment Anything Model (SAM) by Meta AI
- This repository: https://github.com/mdcanocreates/imbot

## License

See LICENSE file for details.

