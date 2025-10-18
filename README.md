# Image Upscaling with a Super-Resolution Convolutional Neural Network

This project implements a Super-Resolution Convolutional Neural Network (SRCNN) for image upscaling.
## Examples

| Low Resolution 0X | High Resolution 6x |
|----------------|-----------------|
| <img width="400" alt="Low Res 1" src="https://github.com/user-attachments/assets/0cdacc26-c900-45ed-a5f8-0522defea6a9" /> | <img width="400" alt="High Res 1" src="https://github.com/user-attachments/assets/5917151a-1f61-45d9-b8ae-a9ff587998bf" /> |
| <img width="400" alt="Low Res 2" src="https://github.com/user-attachments/assets/72ece0fd-d5dc-496d-8c8d-f79c09ef91a4" /> | <img width="400" alt="High Res 2" src="https://github.com/user-attachments/assets/59cf2c4e-ad5d-40d1-b9a3-8197e7634c3d" /> |
| <img width="400" alt="Low Res 3" src="https://github.com/user-attachments/assets/273f005c-840a-48ee-8634-40ff3600df4b" /> | <img width="400" alt="High Res 3" src="https://github.com/user-attachments/assets/ea3a3ba7-fc50-4896-bce2-fa938c09ba5b" /> |
| <img width="400" alt="Low Res 4" src="https://github.com/user-attachments/assets/5916bd6b-d07f-4c92-a525-205256e1d8fc" /> | <img width="400" alt="High Res 4" src="https://github.com/user-attachments/assets/1514575d-d508-464f-aed8-006966cb86f2" /> |

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
