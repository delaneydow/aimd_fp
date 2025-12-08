import numpy as np
import onnxruntime as ort
from helper import load_image

session = ort.InferenceSession("../model/mri_seg_int8.onnx")

def run_inference(image_path):
img = load_image(image_path)
inp = img[None, None, :, :]
output = session.run(None, {session.get_inputs()[0].name: inp})[0]
return output[0, 0]


if __name__ == "__main__":
mask = run_inference("../data/sample_mri_slice.png")
print("Mask sum:", mask.sum())