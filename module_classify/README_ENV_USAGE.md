# Environment-based Training Configuration

## Overview
The training script has been modified to read parameters from a `.env.train` file instead of hardcoded values. This allows for easy configuration changes without modifying the source code.

## Usage

1. **Install required dependency:**
   ```bash
   pip install python-dotenv
   ```

2. **Configure training parameters:**
   Edit the `.env.train` file to customize your training settings:
   ```bash
   nano .env.train
   ```

3. **Run training:**
   ```bash
   python train.py
   ```

## Configuration Parameters

### Dataset Configuration
- `DATASET_PATH`: Local path for dataset storage
- `REPO_ID`: HuggingFace repository ID for the dataset
- `CACHE_DIR`: Cache directory for downloaded data
- `OUT_DIR`: Output directory for checkpoints and logs

### Training Parameters
- `BATCH_SIZE`: Batch size for training
- `EPOCHS`: Number of training epochs
- `LR`: Learning rate
- `WEIGHT_DECAY`: Weight decay for optimizer
- `HEAD_DROPOUT`: Dropout rate for model head
- `FREEZE_BACKBONE`: Whether to freeze backbone weights

### Model Parameters
- `BANDS`: Comma-separated list of spectral bands
- `NUM_FRAMES`: Number of temporal frames
- `CLASS_WEIGHTS`: Comma-separated class weights for loss function

### Model Architecture
- `BACKBONE`: Backbone model name
- `BACKBONE_PRETRAINED`: Use pretrained backbone weights
- `BACKBONE_COORDS_ENCODING`: Coordinate encoding types
- `DECODER`: Decoder architecture
- `DECODER_CHANNELS`: Number of decoder channels
- `DECODER_SCALE_MODULES`: Scale decoder modules
- `NECK_INDICES`: Comma-separated neck layer indices

### Training Configuration
- `SEED`: Random seed for reproducibility
- `NUM_WORKERS`: Number of data loading workers
- `PRECISION`: Training precision (e.g., "bf16-mixed")
- `LOG_EVERY_N_STEPS`: Logging frequency

## Model-Specific Configurations

### Backbone Models and Neck Indices
Different backbone models require different neck indices:

- **prithvi_eo_v2_300**: `NECK_INDICES=5,11,17,23`
- **prithvi_eo_v2_600**: `NECK_INDICES=7,15,23,31`
- **prithvi_vit_100**: `NECK_INDICES=2,5,8,11`

Make sure to update both `BACKBONE` and `NECK_INDICES` together when changing models.

## Example Configurations

### Quick Training (fewer epochs)
```
EPOCHS=10
BATCH_SIZE=8
```

### High-precision Training
```
PRECISION=32
LR=1.0e-4
WEIGHT_DECAY=0.05
```

### Using Different Backbone
```
BACKBONE=prithvi_eo_v2_600
NECK_INDICES=7,15,23,31
```
