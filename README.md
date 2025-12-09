# Quantized MRI Segmentation
A lightweight classical segmentation pipeline quantized to INT8 for deployment.


## Features
- Classical segmentation (threshold + morphology)
- ONNX conversion
- Dynamic INT8 quantization
- FastAPI inference server
- Lightweight Docker deployment
- CPU-only, <200MB memory


## Sample MRI Dataset Links
Use any of the following:
- **TCGA Glioma MRI (Kaggle)**: https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation
- You can extract any axial slice as a PNG for experiments.

### Sample Results (After Quantization)
<img width="1200" height="400" alt="image" src="https://github.com/user-attachments/assets/5c13491e-ca1a-4844-85ae-a831122f5597" />


## Running the Project

python model/train.py python model/quantize.py python model/eval_quantized.py

## Examine Results

see quant_results directory

## Docker (Reproduce Locally)
docker build -t mri-quantized . docker run -p 8000:8000 mri-quantized
