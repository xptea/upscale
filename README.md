# Image Upscaling with SRCNN

This project implements a Super-Resolution Convolutional Neural Network (SRCNN) for image upscaling.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- tqdm
- matplotlib

## Usage

1. Prepare your data in `newdata/low` and `newdata/high` directories.
2. Run training: `python train_newdata.py`
3. For inference: `python inference.py`

## Files

- `model.py`: SRCNN model definition
- `train_newdata.py`: Training script for new dataset
- `inference.py`: Inference script
- `data_loader_multiscale.py`: Data loader for multi-scale images

## Dataset 
- https://www.kaggle.com/code/quadeer15sh/image-super-resolution-using-autoencoders/input