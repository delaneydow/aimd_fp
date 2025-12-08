from fastapi import FastAPI, UploadFile
from PIL import Image
import numpy as np
import onnxruntime as ort


app = FastAPI()
session = ort.InferenceSession("/model/model.onnx")


@app.post("/predict")
async def predict(file: UploadFile):
img = Image.open(file.file).convert("L")
arr = np.array(img).astype(np.float32)
inp = arr[None, None, :, :]
mask = session.run(None, {session.get_inputs()[0].name: inp})[0][0, 0]
return {"mask_pixel_sum": float(mask.sum())}