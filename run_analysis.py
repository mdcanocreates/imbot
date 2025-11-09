"""
Simple script to run the analysis on img_model folder and generate HTML dashboard.

This script automatically uses the venv_sam virtual environment if available,
which includes Python 3.12 and all SAM dependencies.
"""

import subprocess
import sys
from pathlib import Path
import webbrowser
import os

def main():
    """Run the analysis pipeline."""
    data_root = "img_model"
    output_root = "outputs"
    
    # Check if venv_sam exists and use it
    venv_path = Path(__file__).parent / "venv_sam"
    python_exe = sys.executable
    
    if venv_path.exists():
        # Use Python from venv_sam if available
        venv_python = venv_path / "bin" / "python"
        if venv_python.exists():
            python_exe = str(venv_python)
            print("Using Python 3.12 from venv_sam (includes SAM support)")
    
    # Set SAM_CHECKPOINT_PATH if not already set
    if 'SAM_CHECKPOINT_PATH' not in os.environ:
        checkpoint_path = Path(__file__).parent / "sam_vit_b_01ec64.pth"
        if checkpoint_path.exists():
            os.environ['SAM_CHECKPOINT_PATH'] = str(checkpoint_path)
            print(f"Set SAM_CHECKPOINT_PATH to: {checkpoint_path}")
    
    print("="*60)
    print("Running cell image analysis pipeline...")
    print("="*60)
    print(f"Data root: {data_root}")
    print(f"Output root: {output_root}")
    print()
    
    # Run the main pipeline
    cmd = [
        python_exe, "-m", "image_analysis.main",
        "--data-root", data_root,
        "--output-root", output_root,
        "--cell-ids", "CellA", "CellB", "CellC"
    ]
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode == 0:
        print("\n" + "="*60)
        print("✓ Analysis completed successfully!")
        print("="*60)
        print(f"\nResults saved to: {output_root}/")
        
        dashboard_path = Path(output_root) / "dashboard.html"
        if dashboard_path.exists():
            print(f"\nDashboard available at: {dashboard_path}")
            print(f"Opening dashboard in your browser...")
            try:
                webbrowser.open(f"file://{dashboard_path.absolute()}")
            except:
                print(f"Could not open browser automatically.")
                print(f"Please open {dashboard_path.absolute()} manually in your browser.")
        else:
            print(f"\nWarning: Dashboard not found at {dashboard_path}")
    else:
        print("\n" + "="*60)
        print("✗ Analysis failed. Check the error messages above.")
        print("="*60)
        sys.exit(1)

if __name__ == "__main__":
    main()

