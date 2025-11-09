# Gemini QC Integration

This repository includes Google Gemini API integration for automated quality control of segmentation masks.

## Setup

1. Install the required package:
```bash
pip install google-generativeai
```

2. Get a Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

3. Set the API key as an environment variable:
```bash
export GEMINI_API_KEY="AIzaSyC4yL8JpfDYt_NS7VCfGEX0ZJVvmOzFCc8"
```

## Usage

Run the analysis pipeline as usual:

```bash
python -m image_analysis.main --data-root img_model --output-root outputs
```

The pipeline will automatically:
- Generate segmentation masks using the classical skimage-based pipeline
- Create overlay visualizations (green = cell mask, magenta = nuclear mask)
- Send raw + overlay images to Gemini for QC evaluation
- Save QC results to `outputs/gemini_qc_results.json`
- Create a summary CSV at `outputs/gemini_qc_summary.csv`

## Output Files

- `gemini_qc_results.json`: Full QC evaluation results with scores, issues, and suggestions
- `gemini_qc_summary.csv`: Summary table with scores and "needs_review" flag

## Important Notes

- **Gemini NEVER modifies masks directly** - it only evaluates overlays and suggests parameter tweaks
- The classical skimage-based pipeline owns all segmentation decisions
- If `GEMINI_API_KEY` is not set, the pipeline will skip Gemini QC and continue normally
- Scores below 0.7 are flagged as "needs review" in the summary CSV

## Troubleshooting

If Gemini QC fails:
- Check that `GEMINI_API_KEY` is set correctly
- Verify that `google-generativeai` is installed
- Check API quota/rate limits
- The pipeline will continue even if Gemini QC fails

