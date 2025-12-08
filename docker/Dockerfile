FROM python:3.11-slim


RUN pip install fastapi uvicorn pillow numpy onnxruntime scikit-image


COPY ../model/mri_seg_int8.onnx /model/model.onnx
COPY app.py /app/app.py


CMD ["python", "/app/app.py"]