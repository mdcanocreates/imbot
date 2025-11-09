# Segment Anything Model (SAM) Integration

This repository integrates Meta's Segment Anything Model (SAM) in two ways:

1. **Classical Pipeline with SAM Refinement** (`main.py`): SAM is used as a fallback/refinement step when Gemini QC flags poor-quality masks (score < 0.7).

2. **SAM-First Pipeline** (`sam_main.py`): SAM is used as the PRIMARY method for cell segmentation, with classical methods used only for nuclei segmentation and refinement.

## Overview

### Classical Pipeline (main.py)
- Uses classical threshold-based segmentation as the primary method
- SAM is used as an optional refinement step when Gemini QC flags low-quality masks
- This is the safer, more conservative approach

### SAM-First Pipeline (sam_main.py)
- Uses SAM as the PRIMARY method for cell segmentation
- Classical methods are used only for nuclei segmentation inside the SAM mask
- This allows aggressive use of SAM without affecting the classical pipeline
- Useful for comparing "classical-first" vs "SAM-first" approaches

## Setup

### 1. Clone the SAM Repository

The SAM repository should be cloned into the workspace as `segment-anything-main/`:

```bash
# The repo structure should be:
# imbot/
#   ├── segment-anything-main/
#   │   ├── segment_anything/
#   │   │   ├── predictor.py
#   │   │   ├── build_sam.py
#   │   │   └── ...
#   │   └── ...
#   └── image_analysis/
#       └── sam_wrapper.py
```

### 2. Download SAM Model Checkpoint

Download a SAM model checkpoint (ViT-B recommended for speed):

- **ViT-B**: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
- **ViT-L**: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
- **ViT-H**: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

### 3. Set Environment Variable

Set the `SAM_CHECKPOINT_PATH` environment variable to point to your downloaded checkpoint:

```bash
export SAM_CHECKPOINT_PATH="/path/to/sam_vit_b_01ec64.pth"
```

Or add it to your shell profile (`.bashrc`, `.zshrc`, etc.):

```bash
echo 'export SAM_CHECKPOINT_PATH="/path/to/sam_vit_b_01ec64.pth"' >> ~/.bashrc
source ~/.bashrc
```

### 4. Install Dependencies

Install the required packages:

```bash
pip install torch opencv-python segment-anything
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

## How It Works

### Normal Pipeline Flow

1. **Cell Segmentation**: Uses actin/nuclei segmentation pipeline
2. **Gemini QC**: Evaluates mask quality (score 0-1)
3. **Metrics Computation**: Calculates cell metrics

### SAM Refinement Flow (When Triggered)

1. **Gemini QC Flags Low Quality**: Cell mask score < 0.7
2. **SAM Refinement**:
   - Loads Actin or Combo image
   - Uses nuclei centroids as seed points (if available)
   - Generates refined mask using SAM
3. **Re-QC**: Re-runs Gemini QC on refined mask
4. **Mask Replacement**: If refined mask score ≥ 0.7, replaces original mask
5. **Metrics Re-computation**: Recalculates metrics with refined mask

### Output Files

When SAM refinement is triggered, the following files are generated:

- `outputs/<cell_id>_cellmask_before_sam.png`: Original mask (for comparison)
- `outputs/<cell_id>_cellmask_after_sam.png`: SAM-refined mask (for comparison)
- `outputs/<cell_id>_nuclei_with_nuclear_mask_refined.png`: Refined mask overlay

## Usage

### Classical Pipeline (with SAM Refinement)

SAM refinement is **automatic** and **conditional**:

- It only runs when Gemini QC flags a low-quality mask
- No manual intervention required
- Gracefully handles missing SAM dependencies or checkpoints

Run the classical pipeline:
```bash
python3 run_analysis.py
# or
python -m image_analysis.main --data-root img_model --output-root outputs
```

### SAM-First Pipeline

Run the SAM-first pipeline:
```bash
cd "/Users/michael.cano/Desktop/Bme Data/imbot"
source venv_sam/bin/activate
export SAM_CHECKPOINT_PATH="/Users/michael.cano/Desktop/Bme Data/imbot/sam_vit_b_01ec64.pth"
python -m image_analysis.sam_main --data-root img_model --output-root outputs_sam
```

The SAM-first pipeline:
- Uses SAM as the PRIMARY method to segment cells
- Uses classical methods to segment nuclei inside the SAM mask
- Computes the same metrics as the classical pipeline
- Generates QC images with "_sam" suffix for comparison
- Saves results to `outputs_sam/` directory

### Manual Testing

To test SAM refinement manually, you can call the wrapper function:

```python
from image_analysis.sam_wrapper import refine_cell_mask_with_sam
import numpy as np

# Load your image and mask
image_path = "path/to/actin_image.jpg"
initial_mask = np.load("path/to/mask.npy")  # Boolean array

# Refine mask with SAM
refined_mask = refine_cell_mask_with_sam(
    image_path=image_path,
    initial_mask=initial_mask,
    points=None,  # Optional: list of (row, col) seed points
    model_type="vit_b"  # or "vit_l", "vit_h"
)
```

## Troubleshooting

### SAM Not Available

If SAM is not installed or not available, the pipeline will:
- Log a warning
- Continue with the original mask
- Not fail the analysis

### Missing Checkpoint

If `SAM_CHECKPOINT_PATH` is not set or the checkpoint file is missing:
- Logs a warning
- Returns the original mask
- Continues with normal pipeline

### GPU/CPU

SAM automatically uses GPU if available, otherwise falls back to CPU:
- GPU: Faster inference (recommended)
- CPU: Slower but works without GPU

## Configuration

### Model Type

Default model type is `vit_b` (ViT-B). To use a different model:

1. Download the corresponding checkpoint
2. Set `SAM_CHECKPOINT_PATH` to point to it
3. The wrapper will automatically detect the model type from the checkpoint

### Seed Points

SAM uses seed points to guide segmentation:
- **Automatic**: Generated from nuclei centroids (if available)
- **Manual**: Can be provided as `points` parameter (list of (row, col) tuples)

## Notes

- SAM is **optional** - the pipeline works without it
- SAM is **conditional** - only runs when needed
- SAM is **non-destructive** - original masks are preserved
- All results are logged and saved for comparison

