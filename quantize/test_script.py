# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, Shenzhen Institute of Artificial Intelligence and Robotics for Society, all rights reserved.
# Third party copyrights are property of their respective owners.

import os
import sys
import numpy as np
import cv2 as cv

import onnx
import torch
from onnx import version_converter
import onnxruntime
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat

from transform import Compose, Tracking
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
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
sess_options = onnxruntime.SessionOptions()
sess_options.intra_op_num_threads = 4
sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
def get_template_iunformation(img,bbox,net_path):
    transformer = Tracking(128,'backbone')
    zf = transformer(img,bbox).cuda()
    onnx_inputs = {'zf': to_numpy(zf).astype(np.float32)}
    onnx_backbone = onnx.load(net_path)
    onnx.checker.check_model(onnx_backbone)
    ort_session = onnxruntime.InferenceSession(net_path,providers=['CUDAExecutionProvider'])
    ort_outs = ort_session.run(None, onnx_inputs)
    feature_template = torch.from_numpy(ort_outs[0])
    pos_template = torch.from_numpy(ort_outs[1])
    return feature_template,pos_template
class DataReader_backbone(CalibrationDataReader):
    def __init__(self, model_path, image_dir, transforms, data_dim):
        model = onnx.load(model_path)
        # self.input_name = []
        # for i in range(len(model.graph.input)):
        #     input_name = model.graph.input[i].name
        #     self.input_name.append(input_name)
        self.input_name = model.graph.input[0].name
        self.transforms = transforms
        self.data_dim = data_dim
        self.data = self.get_calibration_data(image_dir)
        # for x in self.data:
        #
        self.enum_data_dicts = iter([{self.input_name: x} for x in self.data])

    def get_next(self):
        return next(self.enum_data_dicts, None)

    def get_calibration_data(self, image_dir):
        bbox_dir = image_dir + '/../groundtruth.txt'
        fp = open(bbox_dir,encoding='gb18030')
        lines = fp.readlines()  # 读取全部内容 ，并以列表方式返回
        results = []
        for line in lines:
            line = line.strip('\n').split(',')
            new_line = list(map(int, line))
            results.append(new_line)
        blobs = []
        supported = ["jpg", "png"]  # supported file suffix
        for i,image_name in enumerate(os.listdir(image_dir)):
            image_name_suffix = image_name.split('.')[-1].lower()
            if image_name_suffix not in supported:
                continue
            img = cv.imread(os.path.join(image_dir, image_name))
            img = self.transforms(img,results[i]).cuda()
            img = to_numpy(img).astype(np.float32)
            # if img is None:
            #     continue
            # blob = cv.dnn.blobFromImage(img)
            # if self.data_dim == 'hwc':
            #     blob = cv.transposeND(blob, [0, 2, 3, 1])
            blobs.append(img)
        return blobs

class DataReader_com(CalibrationDataReader):
    def __init__(self, model_path, image_dir, transforms, data_dim,net_path):
        self.net_path = net_path
        model = onnx.load(model_path)
        self.input_name = []
        for i in range(len(model.graph.input)):
            input_name = model.graph.input[i].name
            self.input_name.append(input_name)
        # self.input_name = model.graph.input[0].name
        self.transforms = transforms
        self.data_dim = data_dim
        self.data = self.get_calibration_data(image_dir)#[[...],[...]]
        iter_data = []
        for x in self.data:
            item_dict = {}
            for i,in_name in enumerate(self.input_name):
                item_dict[in_name] = x[i]
            iter_data.append(item_dict)
        self.enum_data_dicts = iter(iter_data)

    def get_next(self):
        return next(self.enum_data_dicts, None)

    def get_calibration_data(self, image_dir):
        final_results = []
        bbox_dir = image_dir + '/../groundtruth.txt'
        fp = open(bbox_dir,encoding='gb18030')
        lines = fp.readlines()  # 读取全部内容 ，并以列表方式返回
        results = []
        for line in lines:
            line = line.strip('\n').split(',')
            new_line = list(map(int, line))
            results.append(new_line)
        blobs = []
        supported = ["jpg", "png"]  # supported file suffix
        for i,image_name in enumerate(os.listdir(image_dir)):
            image_name_suffix = image_name.split('.')[-1].lower()
            if image_name_suffix not in supported:
                continue
            img = cv.imread(os.path.join(image_dir, image_name))
            img_backbone = img.copy()
            feature_template, pos_template = get_template_iunformation(img=img_backbone,bbox=results[i],net_path=self.net_path)
            img = to_numpy(self.transforms(img,results[i]).cuda()).astype(np.float32)
            feature_template = to_numpy(feature_template.cuda()).astype(np.float32)
            pos_template = to_numpy(pos_template.cuda()).astype(np.float32)
            # if img is None:
            #     continue
            # blob = cv.dnn.blobFromImage(img)
            # if self.data_dim == 'hwc':
            #     blob = cv.transposeND(blob, [0, 2, 3, 1])
            blobs.append(img)
            blobs.append(feature_template)
            blobs.append(pos_template)
            final_results.append(blobs)
        return final_results

class Quantize:
    def __init__(self, model_path, calibration_image_dir, transforms=Tracking(size=128,convert_type="backbone"), convert_type="backbone",net_path="..",per_channel=False, act_type='int8', wt_type='int8', data_dim='chw',quan_nodes = ['conv']):
        self.type_dict = {"uint8" : QuantType.QUInt8, "int8" : QuantType.QInt8}
        self.quan_nodes = quan_nodes
        self.model_path = model_path
        self.calibration_image_dir = calibration_image_dir
        self.transforms = transforms
        self.per_channel = per_channel
        self.act_type = act_type
        self.wt_type = wt_type

        # data reader
        if convert_type == "backbone":
            self.dr = DataReader_backbone(self.model_path, self.calibration_image_dir, self.transforms, data_dim)
        # elif convert_type == "complete":
        #     self.dr = DataReader_com(self.model_path, self.calibration_image_dir, self.transforms, data_dim,net_path=net_path)
    def check_opset(self):
        model = onnx.load(self.model_path)
        if model.opset_import[0].version != 13:
            print('\tmodel opset version: {}. Converting to opset 13'.format(model.opset_import[0].version))
            # convert opset version to 13
            model_opset13 = version_converter.convert_version(model, 13)
            # save converted model
            output_name = '{}-opset13.onnx'.format(self.model_path[:-5])
            onnx.save_model(model_opset13, output_name)
            # update model_path for quantization
            return output_name
        return self.model_path

    # nodes_to_exclude = ['Gemm_10', 'Gemm_12', 'Gemm_34', 'Gemm_36', 'Gemm_49', 'Gemm_51', 'Gemm_65', 'Gemm_67',
    #                     'Gemm_81', 'Gemm_83', 'Gemm_96', 'Gemm_98']
    # nodes_to_quantize = ['Conv_2', 'Conv_5', 'Conv_15', 'Conv_16', 'Conv_18', 'Conv_20', 'Conv_21', 'Conv_23',
    #                      'Conv_25', 'Conv_27', 'Conv_30', 'Conv_41', 'Conv_42', 'Conv_45', 'Conv_56', 'Conv_58',
    #                      'Conv_61', 'Conv_72', 'Conv_74', 'Conv_77', 'Conv_88', 'Conv_89', 'Conv_92', 'Conv_103'])

    def run(self,quan_nodes):
        print('Quantizing {}: act_type {}, wt_type {}'.format(self.model_path, self.act_type, self.wt_type))
        new_model_path = self.check_opset()
        output_name = '{}_{}_{}.onnx'.format(self.model_path[:-5], self.wt_type,quan_nodes[0])
        quantize_static(new_model_path, output_name, self.dr,
                        quant_format=QuantFormat.QOperator, # start from onnxruntime==1.11.0, quant_format is set to QuantFormat.QDQ by default, which performs fake quantization
                        per_channel=self.per_channel,
                        weight_type=self.type_dict[self.wt_type],
                        activation_type=self.type_dict[self.act_type],
                        nodes_to_quantize=quan_nodes)
        # if new_model_path != self.model_path:
        #     os.remove(new_model_path)
        print('\tQuantized model saved to {}'.format(output_name))
        ort_session_back = onnxruntime.InferenceSession(
            "/media/kb/2T5/hcat/HCAT/pysot_toolkit/models/mobiletrack/back_cov.onnx",
            providers=['CPUExecutionProvider'], sess_options=sess_options)
        ort_session_back_int8 = onnxruntime.InferenceSession(
            os.path.join(new_model_path,output_name),
            providers=['CPUExecutionProvider'], sess_options=sess_options)
        with torch.no_grad():
            x, feature, pos, pos_fast, z = getdata()
            ort_back_input = {'zf': to_numpy(z)}
            ort_com_input = {'x': to_numpy(x),
                             'feature_template': to_numpy(feature),
                             'pos_template': to_numpy(pos)}
            ort_com_fast_input = {'x': to_numpy(x),
                                  'feature_template': to_numpy(feature),
                                  'pos_template': to_numpy(pos_fast)}
            # test accuracy
            ort_back_outs = ort_session_back.run(None, ort_back_input)
            ort_back_int8_outs = ort_session_back_int8.run(None, ort_back_input)
            print("quan_nodes: {} ,The deviation between back_outs, the first output: {} \n".format(
                quan_nodes,np.average(np.abs(ort_back_outs[0] - ort_back_int8_outs[0]))))

models_com=dict(
    mobiletrack_com=Quantize(model_path='/media/kb/2T5/hcat/HCAT/pysot_toolkit/models/mobiletrack/com_cov.onnx',
                                  calibration_image_dir='/media/kb/08AC1CC7272DCD64/trackingdata/test/lasot_extension_subset/cosplay/cosplay-2/img',
                                  transforms=Tracking(size=256,convert_type='complete'),
                             convert_type='complete',
                             net_path="/media/kb/2T5/hcat/HCAT/pysot_toolkit/models/mobiletrack/back_cov.onnx"),
    mobiletrackfast_com=Quantize(model_path='/media/kb/2T5/hcat/HCAT/pysot_toolkit/models/mobiletrack_fast/com_cov.onnx',
                             calibration_image_dir='/media/kb/08AC1CC7272DCD64/trackingdata/test/lasot_extension_subset/cosplay/cosplay-2/img',
                             transforms=Tracking(size=256,convert_type='complete'),
                                 convert_type='complete',
                                 net_path='/media/kb/2T5/hcat/HCAT/pysot_toolkit/models/mobiletrack_fast/back_cov.onnx'),
)

if __name__ == '__main__':
    nodes_to_quantize = ['Conv_2', 'Conv_5', 'Conv_15', 'Conv_16', 'Conv_18', 'Conv_20', 'Conv_21', 'Conv_23',
                         'Conv_25', 'Conv_27', 'Conv_30', 'Conv_41', 'Conv_42', 'Conv_45', 'Conv_56', 'Conv_58',
                         'Conv_61', 'Conv_72', 'Conv_74', 'Conv_77', 'Conv_88', 'Conv_89', 'Conv_92', 'Conv_103']
    for i in range(len(nodes_to_quantize)):
        quan_nodes = nodes_to_quantize[i]
        models_backbone = dict(
            mobiletrack_backbone=Quantize(
                model_path='/media/kb/2T5/hcat/HCAT/pysot_toolkit/models/mobiletrack/back_cov.onnx',
                calibration_image_dir='/media/kb/08AC1CC7272DCD64/trackingdata/test/lasot_extension_subset/cosplay/cosplay-2/img',
                transforms=Tracking(size=128, convert_type='backbone'),
                convert_type='backbone',
                quan_nodes=['conv']),
            # mobiletrackfast_backbone=Quantize(model_path='/media/kb/2T5/hcat/HCAT/pysot_toolkit/models/mobiletrack_fast/back_cov.onnx',
            #                               calibration_image_dir='/media/kb/08AC1CC7272DCD64/trackingdata/test/lasot_extension_subset/cosplay/cosplay-2/img',
            #                               transforms=Tracking(size=128,convert_type='backbone'),
            #                               convert_type='backbone'),
        )
        selected_models_backbone = list(models_backbone.keys())
        selected_models_com = list(models_com.keys())
        print('Backbone Models to be quantized: {}'.format(str(selected_models_backbone)))
        print('Com Models to be quantized: {}'.format(str(selected_models_com)))
        for selected_model_name_backbone in selected_models_backbone:
            q = models_backbone[selected_model_name_backbone]
            q.run([quan_nodes])
    # for selected_model_name_com in selected_models_com:
    #     q = models_com[selected_model_name_com]
    #     q.run()