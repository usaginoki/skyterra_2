#!/usr/bin/env python3
"""
Convenience script to run multicrop inference following TerraTorch documentation
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from model_inference_1 import run_inference_directory

def main():
    # Project-specific paths
    base_dir = Path(__file__).parent
    
    # Configuration and checkpoint paths
    config_file = base_dir / "multicrop.yaml"
    checkpoint_path = base_dir / "multicrop_example" / "multicrop_example" / "checkpoints" / "best-checkpoint-epoch=42-val_loss=0.00.ckpt"
    
    # Example input and output directories
    input_dir = base_dir.parent / "data"
    output_dir = base_dir / "inference_output"
    
    print("=== Quick Multicrop Inference (TerraTorch Style) ===")
    print(f"Config: {config_file}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Check if paths exist
    if not config_file.exists():
        print(f"Error: Config file not found: {config_file}")
        return
        
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        return
        
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        print("Please modify the input_dir path in this script to point to your data")
        return
    
    # Run inference following TerraTorch documentation
    run_inference_directory(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        config_file=str(config_file),
        checkpoint_path=str(checkpoint_path),
        visualize=True
    )

if __name__ == "__main__":
    main() 