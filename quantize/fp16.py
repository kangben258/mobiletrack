import onnx
from onnxruntime.quantization import quantize_q
model = onnx.load("/media/kb/2T5/hcat/HCAT/pysot_toolkit/models/mobiletrack/back_cov.onnx")
model_fp16 = float16.convert_float_to_float16(model)
onnx.save(model_fp16,"/media/kb/2T5/hcat/HCAT/pysot_toolkit/models/mobiletrack/back_cov_fp16.onnx")
model_com = onnx.load("/media/kb/2T5/hcat/HCAT/pysot_toolkit/models/mobiletrack/com_cov.onnx")
model_fp16_com = float16.convert_float_to_float16(model_com)
onnx.save(model_fp16_com,"/media/kb/2T5/hcat/HCAT/pysot_toolkit/models/mobiletrack/com_cov_fp16.onnx")