import copy

import torch.ao.quantization
from torch.quantization.quantize_fx import prepare_qat_fx,convert_fx
import torch.quantization.observer as observer
#将模型转换为qat模型
def qat_version_model(model):
    qconfig_dict = {
        "":torch.ao.quantization.get_default_qat_qconfig('qnnpack'),
        "module_name":[
            ('backbone.1', None),
            ('backbone.0.body.features.1.conv.3', None),
            ('backbone.0.body.features.4.conv.5', None),
            ('backbone.0.body.features.5.conv.5', None),
            ('backbone.0.body.features.6.conv.5', None),
            ('backbone.0.body.features.7.conv.5', None),
            ('backbone.0.body.features.8.conv.5', None),
            ('featurefusion_network',None),
        ],

    }
    model_to_qunatize = copy.deepcopy(model)
    model_fp32_prepared = prepare_qat_fx(model_to_qunatize,qconfig_dict)
    return model_fp32_prepared