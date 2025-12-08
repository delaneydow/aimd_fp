# Quantized MRI Segmentation
A lightweight classical segmentation pipeline converted to ONNX and quantized to INT8 for deployment.


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
- **IXI Dataset**: https://brain-development.org/ixi-dataset/
- **FastMRI single-slice**: https://fastmri.org/


You can extract any axial slice as a PNG for experiments.


## Running the Project

python src/convert_to_onnx.py python src/quantize_model.py python src/inference.py

## Docker
docker build -t mri-quantized . docker run -p 8000:8000 mri-quantized