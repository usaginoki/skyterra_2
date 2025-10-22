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
from terratorch.tasks.tiled_inference import tiled_inference
from terratorch.utils import view_api
from dotenv import load_dotenv
from terratorch.tasks import SemanticSegmentationTask

load_dotenv('.env.train')

#* Defining variables from environment
DATASET_PATH = os.getenv('DATASET_PATH', './data_1')
REPO_ID = os.getenv('REPO_ID', 'ibm-nasa-geospatial/multi-temporal-crop-classification')
CACHE_DIR = os.getenv('CACHE_DIR', './cache')
OUT_DIR = os.getenv('OUT_DIR', './multicrop_example')

# Training parameters
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '16'))
EPOCHS = int(os.getenv('EPOCHS', '50'))
LR = float(os.getenv('LR', '2.0e-4'))
WEIGHT_DECAY = float(os.getenv('WEIGHT_DECAY', '0.1'))
HEAD_DROPOUT = float(os.getenv('HEAD_DROPOUT', '0.1'))
FREEZE_BACKBONE = os.getenv('FREEZE_BACKBONE', 'False').lower() == 'true'

# Model parameters
BANDS_STR = os.getenv('BANDS', 'BLUE,GREEN,RED,NIR_NARROW,SWIR_1,SWIR_2')
BANDS = [band.strip() for band in BANDS_STR.split(',')]
NUM_FRAMES = int(os.getenv('NUM_FRAMES', '3'))

# Class weights
CLASS_WEIGHTS_STR = os.getenv('CLASS_WEIGHTS', 
    '0.386375,0.661126,0.548184,0.640482,0.876862,0.925186,3.249462,1.542289,2.175141,2.272419,3.062762,3.626097,1.198702')
CLASS_WEIGHTS = [float(w.strip()) for w in CLASS_WEIGHTS_STR.split(',')]

# Model configuration
BACKBONE = os.getenv('BACKBONE', 'prithvi_eo_v2_300_tl')
BACKBONE_PRETRAINED = os.getenv('BACKBONE_PRETRAINED', 'True').lower() == 'true'
BACKBONE_COORDS_ENCODING_STR = os.getenv('BACKBONE_COORDS_ENCODING', 'time,location')
BACKBONE_COORDS_ENCODING = [enc.strip() for enc in BACKBONE_COORDS_ENCODING_STR.split(',')]

DECODER = os.getenv('DECODER', 'UperNetDecoder')
DECODER_CHANNELS = int(os.getenv('DECODER_CHANNELS', '256'))
DECODER_SCALE_MODULES = os.getenv('DECODER_SCALE_MODULES', 'True').lower() == 'true'

# Neck indices based on backbone
NECK_INDICES_STR = os.getenv('NECK_INDICES', '5,11,17,23')  # default for prithvi_eo_v2_300
NECK_INDICES = [int(idx.strip()) for idx in NECK_INDICES_STR.split(',')]

# Training configuration
SEED = int(os.getenv('SEED', '0'))
NUM_WORKERS = int(os.getenv('NUM_WORKERS', '7'))
PRECISION = os.getenv('PRECISION', 'bf16-mixed')
LOG_EVERY_N_STEPS = int(os.getenv('LOG_EVERY_N_STEPS', '10'))

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
    
    # print(f"Loading model from config: {config_file}")
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load model_args from config file
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # Extract model_args from the nested structure
    model_args = config.get('model', {}).get('init_args', {}).get('model_args', {})
    print(f"Extracted model args: {model_args}")
    
    # Instantiate the model following the documentation
    try:
        # lightning_model = LightningInferenceModel.from_config(
        #     config_file, 
        #     checkpoint_path, 
        # )
        # model = SemanticSegmentationTask.load_from_checkpoint(
        #     checkpoint_path,
        #     model_factory="EncoderDecoderFactory",
        #     # model_args=model_args,
        # )
        backbone_args = dict(
            backbone_pretrained=BACKBONE_PRETRAINED,
            backbone=BACKBONE, # prithvi_eo_v2_300, prithvi_eo_v2_300_tl, prithvi_eo_v2_600, prithvi_eo_v2_600_tl
            backbone_coords_encoding=BACKBONE_COORDS_ENCODING,
            backbone_bands=BANDS,
            backbone_num_frames=NUM_FRAMES,
        )

        decoder_args = dict(
            decoder=DECODER,
            decoder_channels=DECODER_CHANNELS,
            decoder_scale_modules=DECODER_SCALE_MODULES,
        )

        necks = [
            dict(
                    name="SelectIndices",
                    # indices=[2, 5, 8, 11]    # indices for prithvi_vit_100
                    indices=NECK_INDICES,   # indices for prithvi_eo_v2_300
                    # indices=[7, 15, 23, 31]  # indices for prithvi_eo_v2_600
                ),
            dict(
                    name="ReshapeTokensToImage",
                    effective_time_dim=NUM_FRAMES,
                )
            ]

        model_args = dict(
            **backbone_args,
            **decoder_args,
            num_classes=len(CLASS_WEIGHTS),
            head_dropout=HEAD_DROPOUT,
            necks=necks,
            rescale=True,
        )
            

        model = SemanticSegmentationTask(
            model_args=model_args,
            plot_on_val=False,
            class_weights=CLASS_WEIGHTS,
            loss="ce",
            ignore_index=-1,
            freeze_backbone=FREEZE_BACKBONE,
            freeze_decoder=False,
            model_factory="EncoderDecoderFactory",
        )
        model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
        # load the checkpoint
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
        # input_files.extend(glob.glob(os.path.join(input_dir, '**', ext), recursive=True))
    
    if not input_files:
        print(f"No TIFF files found in {input_dir}")
        return
    
    print(f"Found {len(input_files)} files to process")
    
    # Use the TerraTorch inference_on_dir method
    try:
        predictions = []
        original_predictions = []
        file_names = []
        def model_forward(x,  **kwargs):
            return model(x, **kwargs).output 
        for input_file in input_files:
            try:
                print(f"Processing: {os.path.basename(input_file)}")
                print(f"Loading input file: {input_file}")
                input = rioxarray.open_rasterio(input_file)
                print("Doing something else...")
                #preprocessing
                
                # input = (input - means[:, None, None]) / stds[:, None, None]
                # normalize the input (stupid)
                input = input.to_numpy()
                input = (input - input.mean()) / input.std()
                
                # reshape to (1, 6, 3, x1, x2)
                input = input.reshape(6, 3, input.shape[-2], input.shape[-1])
                input = torch.tensor(input, dtype=torch.float, device='cpu').unsqueeze(0)
                print(f"Input tensor: {input.shape}")
                print(f"Running inference...")
                # Add out_channels parameter and use tiled inference parameters from config
                prediction = tiled_inference(
                    model_forward, 
                    input, 
                    out_channels=13,  # Number of crop classes
                    h_crop=224,
                    w_crop=224,
                    h_stride=112,
                    w_stride=112,
                    average_patches=True,
                    verbose=True
                )
                print("Inference finished")
                print(f"Raw prediction shape: {prediction.shape}")
                # Create 2D image where each pixel contains the MAXIMUM VALUE from the 13 layers
                # prediction shape: (13, height, width) -> max_values shape: (height, width)
                prediction_original = prediction[0]  # Original tensor structure (13, H, W)
                print(f"Original prediction shape: {prediction_original.shape}")
                prediction_max_values = prediction[0].max(dim=0)[1]  # [0] gets values, [1] would get indices
                print(f"Max values prediction shape: {prediction_max_values.shape}")
                original_predictions.append(prediction_original)  # Original tensor structure
                predictions.append(prediction_max_values)  # 2D maximum values
                file_names.append(input_file)
            except Exception as e:
                print(f"Error processing {input_file}: {e}")
                continue
    except Exception as e:
        print(f"Error during batch inference: {e}")
        # Fall back to single file processing
    
    # Save predictions and create visualizations
    for pred, input_file in zip(predictions, file_names):
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        
       
        # Save prediction as TIFF (2D max values from all layers)
        output_file = os.path.join(output_dir, f"{base_name}_pred.tif")
        save_max_values_prediction(pred, input_file, output_file)
        print(f"Saved max values prediction: {output_file}")
        
        # Create visualization if requested
        if visualize:
            try:
                create_visualization_terratorch_style(input_file, pred, output_dir, base_name)
            except Exception as e:
                print(f"Warning: Could not create visualization for {base_name}: {e}")
                
    for pred, input_file in zip(original_predictions, file_names):
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        
        # save predictions as tiff (original tensor structure)
        output_file = os.path.join(output_dir, f"{base_name}_pred_full_original.tif")
        save_prediction(pred, input_file, output_file)
        print(f"Saved original full prediction: {output_file}")
    
    print("Inference completed!")

def save_prediction(prediction: torch.Tensor, input_file: str, output_file: str):
    """Save prediction as GeoTIFF with spatial reference, preserving original tensor structure"""
    
    # Get spatial information from input file
    with rasterio.open(input_file) as src:
        transform = src.transform
        crs = src.crs
    
    # Convert prediction to numpy
    pred_np = prediction.numpy() if isinstance(prediction, torch.Tensor) else prediction
    
    print(f"Original prediction shape: {pred_np.shape}")
    
    # Handle different tensor shapes
    if pred_np.ndim == 2:
        # Simple 2D case (height, width)
        height, width = pred_np.shape
        count = 1
        data_to_write = pred_np
    elif pred_np.ndim == 3:
        # 3D case - could be (channels, height, width) or (height, width, channels)
        if pred_np.shape[0] <= 13:  # Likely (channels, height, width)
            count, height, width = pred_np.shape
            data_to_write = pred_np
        else:  # Likely (height, width, channels)
            height, width, count = pred_np.shape
            data_to_write = np.transpose(pred_np, (2, 0, 1))  # Convert to (channels, height, width)
    else:
        # For higher dimensions, flatten extra dimensions and keep the last 2 as spatial
        original_shape = pred_np.shape
        # Reshape to (channels, height, width) by flattening leading dimensions
        spatial_dims = pred_np.shape[-2:]  # Last 2 dimensions are spatial
        channel_dims = np.prod(pred_np.shape[:-2])  # Product of all other dimensions
        pred_np = pred_np.reshape(channel_dims, spatial_dims[0], spatial_dims[1])
        count, height, width = pred_np.shape
        data_to_write = pred_np
        print(f"Reshaped from {original_shape} to {pred_np.shape}")
    
    print(f"Saving as: {count} bands, {height}x{width}")
    
    # Save prediction with spatial reference
    with rasterio.open(
        output_file,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=count,
        dtype='uint8',
        crs=crs,
        transform=transform,
        compress='lzw'
    ) as dst:
        if count == 1:
            dst.write(data_to_write.astype('uint8'), 1)
        else:
            for i in range(count):
                dst.write(data_to_write[i].astype('uint8'), i + 1)

def save_max_values_prediction(prediction: torch.Tensor, input_file: str, output_file: str):
    """Save 2D prediction where each pixel contains the maximum value from all 13 layers"""
    
    # Get spatial information from input file
    with rasterio.open(input_file) as src:
        transform = src.transform
        crs = src.crs
    
    # Convert prediction to numpy
    pred_np = prediction.numpy() if isinstance(prediction, torch.Tensor) else prediction
    
    print(f"Saving max values prediction with shape: {pred_np.shape}")
    print(f"Value range: min={pred_np.min():.4f}, max={pred_np.max():.4f}")
    
    # Ensure we have a 2D array (should be height x width with maximum values)
    if pred_np.ndim != 2:
        raise ValueError(f"Expected 2D array for max values prediction, got {pred_np.ndim}D with shape {pred_np.shape}")
    
    height, width = pred_np.shape
    
    # Determine appropriate data type based on value range
    if pred_np.min() >= 0 and pred_np.max() <= 255:
        dtype = 'uint8'
        data_to_save = pred_np.astype('uint8')
    elif pred_np.min() >= -32768 and pred_np.max() <= 32767:
        dtype = 'int16'
        data_to_save = pred_np.astype('int16')
    else:
        dtype = 'float32'
        data_to_save = pred_np.astype('float32')
    
    print(f"Using dtype: {dtype}")
    
    # Save as single-band GeoTIFF with maximum values
    with rasterio.open(
        output_file,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform,
        compress='lzw'
    ) as dst:
        dst.write(data_to_save, 1)
    
    print(f"Saved 2D maximum values image: {output_file}")

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
