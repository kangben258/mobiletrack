import sys
import os
env_path = os.path.join(os.path.dirname(__file__), '..')
print(env_path)
if env_path not in sys.path:
    sys.path.append(env_path)
import torch
import time
import onnx
import onnxruntime
import numpy as np
import torch.nn as nn
from ltr.models.tracking.hcat import MLP
from ltr.models.backbone.transt_backbone import build_backbone,Backbone
from ltr.models.neck.featurefusion_network_simple import build_featurefusion_network
import ltr.admin.settings as ws_settings
from ltr.models.backbone.mobileone import reparameterize_model
from ltr.models.neck.position_encoding import build_position_encoding
os.environ["CUDA_VISIBLE_DEVICES"]= "0"

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

sess_options = onnxruntime.SessionOptions()
sess_options.intra_op_num_threads = 1
sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
def getdata(b=1,x=256,z=8):
    x = torch.randn(1,3,x,x)
    zf = torch.randn(1,3,128,128)
    feature = torch.randn(1,48,z,z)
    # feature = torch.randn(1,1024,z,z)
    # feature = torch.randn(1,256,z,z)
    # feature = torch.randn(1,96,z,z)
    pos = torch.randn(1,128,z,z)
    pos_fast = torch.randn(1, 64, z, z)
    return x, feature, pos, pos_fast, zf
def inference_speed_track():
    T_w = 100  # warmup
    T_t = 1000  # test

    ort_session_back = onnxruntime.InferenceSession("/media/kb/2T5/hcat/HCAT/pysot_toolkit/models/mobiletrack/back_cov.onnx",providers=['CPUExecutionProvider'], sess_options=sess_options)
    ort_session_back_int8 = onnxruntime.InferenceSession("/media/kb/2T5/hcat/HCAT/pysot_toolkit/models/mobiletrack/back_cov_int8.onnx", providers=['CPUExecutionProvider'],sess_options=sess_options)
    ort_session_com = onnxruntime.InferenceSession("/media/kb/2T5/hcat/HCAT/pysot_toolkit/models/mobiletrack/com_cov.onnx", providers=['CPUExecutionProvider'],sess_options=sess_options)
    ort_session_com_int8 = onnxruntime.InferenceSession("/media/kb/2T5/hcat/HCAT/pysot_toolkit/models/mobiletrack/com_cov_int8.onnx", providers=['CPUExecutionProvider'],sess_options=sess_options)
    ort_session_back_fast = onnxruntime.InferenceSession("/media/kb/2T5/hcat/HCAT/pysot_toolkit/models/mobiletrack_fast/back_cov.onnx",providers=['CPUExecutionProvider'], sess_options=sess_options)
    ort_session_back_fast_int8 = onnxruntime.InferenceSession("/media/kb/2T5/hcat/HCAT/pysot_toolkit/models/mobiletrack_fast/back_cov_int8.onnx", providers=['CPUExecutionProvider'],sess_options=sess_options)
    ort_session_com_fast = onnxruntime.InferenceSession("/media/kb/2T5/hcat/HCAT/pysot_toolkit/models/mobiletrack_fast/com_cov.onnx", providers=['CPUExecutionProvider'],sess_options=sess_options)
    ort_session_com_fast_int8 = onnxruntime.InferenceSession("/media/kb/2T5/hcat/HCAT/pysot_toolkit/models/mobiletrack_fast/com_cov_int8.onnx", providers=['CPUExecutionProvider'],sess_options=sess_options)
    print(onnxruntime.get_device())
    print("Tracking onnx model has done!")
    with torch.no_grad():
        x, feature, pos, pos_fast, z = getdata()
        ort_back_input = {'zf': to_numpy(z)}
        ort_com_input = {'x':to_numpy(x),
                      'feature_template':to_numpy(feature),
                      'pos_template':to_numpy(pos)}
        ort_com_fast_input= {'x':to_numpy(x),
                      'feature_template':to_numpy(feature),
                      'pos_template':to_numpy(pos_fast)}
        # test accuracy
        ort_back_outs = ort_session_back.run(None, ort_back_input)
        ort_back_int8_outs = ort_session_back_int8.run(None, ort_back_input)
        print("The deviation between back_outs, the first output: {} \n".format(np.average(np.abs(ort_back_outs[0]-ort_back_int8_outs[0]))))
        print("The deviation between back_outs, the second output: {} \n".format(np.max(np.abs(ort_back_outs[1]-ort_back_int8_outs[1]))))

        ort_back_fast_outs = ort_session_back_fast.run(None, ort_back_input)
        ort_back_fast_int8_outs = ort_session_back_fast_int8.run(None, ort_back_input)
        print("The deviation between back_fast_outs, the first output: {} \n".format(np.max(np.abs(ort_back_fast_outs[0]-ort_back_fast_int8_outs[0]))))
        print("The deviation between back_fast_outs, the second output: {} \n".format(np.max(np.abs(ort_back_fast_outs[1]-ort_back_fast_int8_outs[1]))))

        # ort_com_outs = ort_session_com.run(None, ort_com_input)
        # ort_com_int8_outs = ort_session_com_int8.run(None, ort_com_input)
        # print("The deviation between com_outs, the first output: {} \n".format(np.max(np.abs(ort_com_outs[0]-ort_com_int8_outs[0]))))
        # print("The deviation between com_outs, the second output: {} \n".format(np.max(np.abs(ort_com_outs[1]-ort_com_int8_outs[1]))))
        #
        # ort_com_fast_outs = ort_session_com_fast.run(None, ort_com_fast_input)
        # ort_com_fast_int8_outs = ort_session_com_fast_int8.run(None, ort_com_fast_input)
        # print("The deviation between com_fast_outs, the first output: {} \n".format(np.max(np.abs(ort_com_fast_outs[0]-ort_com_fast_int8_outs[0]))))
        # print("The deviation between com_fast_outs, the second output: {} \n".format(np.max(np.abs(ort_com_fast_outs[1]-ort_com_fast_int8_outs[1]))))


        for i in range(T_w):
            ort_back_outs = ort_session_back.run(None, ort_back_input)

        onnx_s = time.time()
        for i in range(T_t):
            ort_back_outs = ort_session_back.run(None, ort_back_input)
        onnx_e = time.time()
        onnxt = onnx_e - onnx_s
        speed = 1000/onnxt
        print('The tracking process inference speed of ort_back_outs model: %.2f FPS \n' % (speed))

        for i in range(T_w):
            ort_back_int8_outs = ort_session_back_int8.run(None, ort_back_input)

        onnx_s1 = time.time()
        for i in range(T_t):
            ort_back_int8_outs = ort_session_back_int8.run(None, ort_back_input)
        onnx_e1 = time.time()
        onnxt1 = onnx_e1 - onnx_s1
        speed = 1000/onnxt1
        print('The tracking process inference speed of ort_back_int8_outs model: %.2f FPS \n' % (speed))
#
#         for i in range(T_w):
#             ort_back_fast_outs = ort_session_back_fast.run(None, ort_back_input)
#
#         onnx_s = time.time()
#         for i in range(T_t):
#             ort_back_fast_outs = ort_session_back_fast.run(None, ort_back_input)
#         onnx_e = time.time()
#         onnxt = onnx_e - onnx_s
#         speed = 1000/onnxt
#         print('The tracking process inference speed of ort_back_fast_outs model: %.2f FPS \n' % (speed))
#
#         for i in range(T_w):
#             ort_back_fast_int8_outs = ort_session_back_fast_int8.run(None, ort_back_input)
#
#         onnx_s = time.time()
#         for i in range(T_t):
#             ort_back_fast_int8_outs = ort_session_back_fast_int8.run(None, ort_back_input)
#         onnx_e = time.time()
#         onnxt = onnx_e - onnx_s
#         speed = 1000/onnxt
#         print('The tracking process inference speed of ort_back_fast_int8_outs model: %.2f FPS \n' % (speed))
#
#
#         ####################################################################################################333
#         for i in range(T_w):
#             ort_com_outs = ort_session_com.run(None, ort_com_input)
#
#         onnx_s = time.time()
#         for i in range(T_t):
#             ort_com_outs = ort_session_com.run(None, ort_com_input)
#         onnx_e = time.time()
#         onnxt = onnx_e - onnx_s
#         speed = 1000 / onnxt
#         print('The tracking process inference speed of ort_com_outs model: %.2f FPS \n' % (speed))
#
#         for i in range(T_w):
#             ort_com_int8_outs = ort_session_com_int8.run(None, ort_com_input)
#
#         onnx_s = time.time()
#         for i in range(T_t):
#             ort_com_int8_outs = ort_session_com_int8.run(None, ort_com_input)
#         onnx_e = time.time()
#         onnxt = onnx_e - onnx_s
#         speed = 1000 / onnxt
#         print('The tracking process inference speed of ort_com_int8_outs model: %.2f FPS \n' % (speed))
# ###############################################################################################
#         for i in range(T_w):
#             ort_com_fast_outs = ort_session_com_fast.run(None, ort_com_fast_input)
#
#         onnx_s = time.time()
#         for i in range(T_t):
#             ort_com_fast_outs = ort_session_com_fast.run(None, ort_com_fast_input)
#         onnx_e = time.time()
#         onnxt = onnx_e - onnx_s
#         speed = 1000 / onnxt
#         print('The tracking process inference speed of ort_com_fast_outs model: %.2f FPS \n' % (speed))
#
#         for i in range(T_w):
#             ort_com_fast_int8_outs = ort_session_com_fast_int8.run(None, ort_com_fast_input)
#
#         onnx_s = time.time()
#         for i in range(T_t):
#             ort_com_fast_int8_outs = ort_session_com_fast_int8.run(None, ort_com_fast_input)
#         onnx_e = time.time()
#         onnxt = onnx_e - onnx_s
#         speed = 1000 / onnxt
#         print('The tracking process inference speed of ort_com_fast_int8_outs model: %.2f FPS \n' % (speed))




if __name__ == "__main__":
    inference_speed_track()