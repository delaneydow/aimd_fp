from onnxruntime.quantization import quantize_dynamic, QuantType


def quantize():
quantize_dynamic(
model_input="../model/mri_seg.onnx",
model_output="../model/mri_seg_int8.onnx",
weight_type=QuantType.QInt8
)


if __name__ == "__main__":
quantize()