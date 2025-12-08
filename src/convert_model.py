import numpy as np
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType
import onnx
from segmentation import segment_mri_slice


def convert_model():
dummy = FloatTensorType([1, 1, 256, 256])
onnx_model = to_onnx(segment_mri_slice, dummy)
with open("../model/mri_seg.onnx", "wb") as f:
f.write(onnx_model.SerializeToString())


if __name__ == "__main__":
convert_model()