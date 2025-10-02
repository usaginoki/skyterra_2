import os
import sys
import numpy as np
import torch
from dotenv import load_dotenv

import terratorch
from terratorch.datamodules import MultiTemporalCropClassificationDataModule
from terratorch.tasks import SemanticSegmentationTask
from terratorch.datasets.transforms import FlattenTemporalIntoChannels, UnflattenTemporalFromChannels

import albumentations
from albumentations import Compose, Flip, Rotate
from albumentations.pytorch import ToTensorV2

import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from huggingface_hub import snapshot_download


# Load environment variables from .env.train file
load_dotenv('.env.train')

#* Defining variables from environment
DATASET_PATH = os.getenv('DATASET_PATH', './data')
REPO_ID = os.getenv('REPO_ID', 'ibm-nasa-geospatial/multi-temporal-crop-classification')
CACHE_DIR = os.getenv('CACHE_DIR', './cache')
OUT_DIR = os.getenv('OUT_DIR', './multicrop_augmented_example')

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

# Download dataset
_ = snapshot_download(repo_id=REPO_ID, repo_type="dataset", cache_dir=CACHE_DIR, local_dir=DATASET_PATH)


# Adding augmentations for a temporal dataset requires additional transforms
train_transforms = [
    terratorch.datasets.transforms.FlattenTemporalIntoChannels(),
    albumentations.Flip(),
    albumentations.Rotate(),
    albumentations.pytorch.transforms.ToTensorV2(),
    terratorch.datasets.transforms.UnflattenTemporalFromChannels(n_timesteps=NUM_FRAMES),
]

data_module = MultiTemporalCropClassificationDataModule(
    data_root=DATASET_PATH,
    train_transform=train_transforms,
    expand_temporal_dimension=True,
)

data_module.setup("fit")
train_dataset = data_module.train_dataset
print("Train dataset size:", len(train_dataset))

if len(train_dataset) == 0:
    raise ValueError("Train dataset is empty")

val_dataset = data_module.val_dataset
print("Validation dataset size:", len(val_dataset))

data_module.setup("test")
test_dataset = data_module.test_dataset
print("Test dataset size:", len(test_dataset))


#* Setting up training

pl.seed_everything(SEED)

# Logger
logger = TensorBoardLogger(
    save_dir=OUT_DIR,
    name="multicrop_augmented_example",
)

# Callbacks
checkpoint_callback = ModelCheckpoint(
    monitor="val/mIoU",
    mode="max",
    dirpath=os.path.join(OUT_DIR, "multicrop_augmented_example", "checkpoints"),
    filename="best-checkpoint-{epoch:02d}-{val_loss:.2f}",
    save_top_k=1,
)

# Trainer
trainer = pl.Trainer(
    accelerator="auto",
    strategy="auto",
    devices="auto",
    precision=PRECISION,
    num_nodes=1,
    logger=logger,
    max_epochs=EPOCHS,
    check_val_every_n_epoch=1,
    log_every_n_steps=LOG_EVERY_N_STEPS,
    enable_checkpointing=True,
    callbacks=[checkpoint_callback],
    limit_predict_batches=1,  # predict only in the first batch for generating plots
)

# DataModule
data_module = MultiTemporalCropClassificationDataModule(
    batch_size=BATCH_SIZE,
    data_root=DATASET_PATH,
    train_transform=train_transforms,
    reduce_zero_label=True,
    expand_temporal_dimension=True,
    num_workers=NUM_WORKERS,
    use_metadata=True,
)

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
            # indices=    # indices for prithvi_vit_100
            indices=NECK_INDICES,   # indices for prithvi_eo_v2_300
            # indices=  # indices for prithvi_eo_v2_600
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
    lr=LR,
    optimizer="AdamW",
    optimizer_hparams=dict(weight_decay=WEIGHT_DECAY),
    ignore_index=-1,
    freeze_backbone=FREEZE_BACKBONE,
    freeze_decoder=False,
    model_factory="EncoderDecoderFactory",
)

#* Training
trainer.fit(model, datamodule=data_module)

ckpt_path = checkpoint_callback.best_model_path

print(f"Best checkpoint path: {ckpt_path}")

# Test results
test_results = trainer.test(model, datamodule=data_module, ckpt_path=ckpt_path)

print(test_results)