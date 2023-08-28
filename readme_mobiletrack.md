## 安装环境

#先安装好pytorch,

`pip install -r requirements.txt`

## 速度测试

`python pysot_toolkit/speed_onnx_cpu_lighttrack.py`

## 精度测试

`python pysot_toolkit/test.py --dataset <dataset> --dataset_root <path to dataset> --net_path <path of weight> `

## 视频demo

`python pytracking/run_video.py <net_path> <video_path>`



pytorch 模型位于checkpoints/ltr/mobiletrack/mobiletrack

onnx模型位于pysot_toolkit/models中