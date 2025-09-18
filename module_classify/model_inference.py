import argparse
import os
from typing import List, Union
import re
import datetime
import numpy as np
import rasterio
import torch
import rioxarray
import yaml
from pathlib import Path
import glob
import matplotlib.pyplot as plt
from einops import rearrange
from terratorch.cli_tools import LightningInferenceModel
from terratorch.utils import view_api

from terratorch.tasks import SemanticSegmentationTask

def run_inference_directory(input_dir: str, output_dir: str, config_file: str, checkpoint_path: str, 
                           visualize: bool = False):
    """
    Run inference on all files in input directory following TerraTorch documentation
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Dataset bands based on multicrop.yaml configuration
    predict_dataset_bands = [
        "BLUE",
        "GREEN", 
        "RED",
        "NIR_NARROW",
        "SWIR_1",
        "SWIR_2"
    ]
    
    predict_output_bands = predict_dataset_bands
    
    print(f"Loading model from config: {config_file}")
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Instantiate the model following the documentation
    try:
        lightning_model = LightningInferenceModel.from_config(
            config_file, 
            checkpoint_path, 
        )
        # lightning_model = SemanticSegmentationTask.load_from_checkpoint(
        #     checkpoint_path,
        #     model_factory="EncoderDecoderFactory",
            
        # )
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Find all TIFF files in input directory
    tiff_extensions = ['*.tif', '*.tiff', '*.TIF', '*.TIFF']
    input_files = []
    for ext in tiff_extensions:
        input_files.extend(glob.glob(os.path.join(input_dir, ext)))
        input_files.extend(glob.glob(os.path.join(input_dir, '**', ext), recursive=True))
    
    if not input_files:
        print(f"No TIFF files found in {input_dir}")
        return
    
    print(f"Found {len(input_files)} files to process")
    
    # Use the TerraTorch inference_on_dir method
    try:
        print("Running inference on directory...")
        predictions, file_names = lightning_model.inference_on_dir(input_dir)
        print(f"Inference completed for {len(predictions)} files")
    except Exception as e:
        print(f"Error during batch inference: {e}")
        # Fall back to single file processing
        predictions = []
        file_names = []
        
        for input_file in input_files:
            try:
                print(f"Processing: {os.path.basename(input_file)}")
                prediction = lightning_model.inference(input_file)
                predictions.append(prediction)
                file_names.append(input_file)
            except Exception as e:
                print(f"Error processing {input_file}: {e}")
                continue
    
    # Save predictions and create visualizations
    for pred, input_file in zip(predictions, file_names):
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        
        # Save prediction as TIFF
        output_file = os.path.join(output_dir, f"{base_name}_prediction.tif")
        save_prediction(pred, input_file, output_file)
        print(f"Saved prediction: {output_file}")
        
        # Create visualization if requested
        if visualize:
            try:
                create_visualization_terratorch_style(input_file, pred, output_dir, base_name)
            except Exception as e:
                print(f"Warning: Could not create visualization for {base_name}: {e}")
    
    print("Inference completed!")

def save_prediction(prediction: torch.Tensor, input_file: str, output_file: str):
    """Save prediction as GeoTIFF with spatial reference"""
    
    # Get spatial information from input file
    with rasterio.open(input_file) as src:
        transform = src.transform
        crs = src.crs
    
    # Convert prediction to numpy
    pred_np = prediction.numpy() if isinstance(prediction, torch.Tensor) else prediction
    
    # Save prediction with spatial reference
    with rasterio.open(
        output_file,
        'w',
        driver='GTiff',
        height=pred_np.shape[0],
        width=pred_np.shape[1],
        count=1,
        dtype='uint8',
        crs=crs,
        transform=transform,
        compress='lzw'
    ) as dst:
        dst.write(pred_np.astype('uint8'), 1)

def create_visualization_terratorch_style(input_file: str, prediction: torch.Tensor, output_dir: str, base_name: str):
    """Create visualization following TerraTorch documentation style"""
    
    # Load input file following the documentation
    fp = rioxarray.open_rasterio(input_file)
    
    # Convert prediction to numpy
    pred_np = prediction.numpy() if isinstance(prediction, torch.Tensor) else prediction
    
    # Create figure with input and prediction (following the doc style)
    f, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    # Input RGB visualization (following the documentation)
    # Use bands 2, 1, 0 (RED, GREEN, BLUE) with brightness adjustment
    if fp.shape[0] >= 3:
        rgb_bands = [2, 1, 0] if fp.shape[0] >= 6 else [min(2, fp.shape[0]-1), min(1, fp.shape[0]-1), 0]
        rgb_data = fp[rgb_bands] + 0.10  # Following doc: slight brightness adjustment
        rgb_data.plot.imshow(rgb="band", ax=ax[0])
        ax[0].set_title(f"Input RGB - {base_name}")
        ax[0].axis('off')
        
        # Prediction visualization
        im = ax[1].imshow(pred_np, cmap='tab20', vmin=0, vmax=12)
        ax[1].set_title(f"Crop Classification - {base_name}")
        ax[1].axis('off')
        plt.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
        
        # Overlay following the documentation style
        overlay = rgb_data + 0.5 * np.stack(3 * [pred_np], axis=0)
        overlay.plot.imshow(rgb="band", ax=ax[2])
        ax[2].set_title(f"RGB + Classification Overlay - {base_name}")
        ax[2].axis('off')
    else:
        print(f"Warning: Not enough bands for RGB visualization in {base_name}")
        return
    
    plt.tight_layout()
    
    # Save visualization
    vis_output = os.path.join(output_dir, f"{base_name}_visualization.png")
    plt.savefig(vis_output, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization: {vis_output}")

def main():
    parser = argparse.ArgumentParser(description="Run multicrop classification inference following TerraTorch documentation")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory containing input TIFF files")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save prediction outputs")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to model configuration YAML file")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint file")
    parser.add_argument("--visualize", action="store_true",
                       help="Create visualization outputs")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        return
    
    if not os.path.exists(args.config):
        print(f"Error: Config file {args.config} does not exist")
        return
        
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file {args.checkpoint} does not exist")
        return
    
    print("=== Multicrop Classification Inference (TerraTorch Style) ===")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Config file: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Create visualizations: {args.visualize}")
    print("=" * 60)
    
    run_inference_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        config_file=args.config,
        checkpoint_path=args.checkpoint,
        visualize=args.visualize
    )

if __name__ == "__main__":
    main()
