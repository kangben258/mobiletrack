# MobileTracker

## Tracker

## Installation

This document contains detailed instructions for installing the necessary dependencied for **Mobiletrack**. The instructions 
have been tested on Ubuntu 18.04 system.

#### Install dependencies
* Create and activate a conda environment 
```bash
conda create -n mobiletrack python=3.7
conda activate mobiletrack
```
* Install PyTorch
```bash
conda install -c pytorch pytorch=1.5 torchvision=0.6.1 cudatoolkit=10.2
```

* Install other packages
```bash
conda install matplotlib pandas tqdm
pip install opencv-python tb-nightly visdom scikit-image tikzplotlib gdown timm
conda install cython scipy
sudo apt-get install libturbojpeg
pip install pycocotools jpeg4py
pip install wget yacs
pip install shapely==1.6.4.post2
```
* Install onnx and onnxruntime
* Here the version of onnxruntime-gpu needs to be compatible to the CUDA  version and CUDNN version on the machine. For more details, please refer to https://www.onnxruntime.ai/docs/reference/execution-providers/CUDA-ExecutionProvider.html . For example, on my computer, CUDA version is 10.2, CUDNN version is 8.0.3, so I choose onnxruntime-gpu==1.6.0

```
pip install onnx onnxruntime-gpu==1.6.0
```



* Setup the environment                                                                                                 
  Create the default environment setting files.

```bash
# Change directory to <PATH_of_mobiletrack>
cd mobiletrack

# Environment settings for pytracking. Saved at pytracking/evaluation/local.py
python -c "from pytracking.evaluation.environment import create_default_local_file; create_default_local_file()"

# Environment settings for ltr. Saved at ltr/admin/local.py
python -c "from ltr.admin.environment import create_default_local_file; create_default_local_file()"
```
You can modify these files to set the paths to datasets, results paths etc.
* Add the project path to environment variables  
Open ~/.bashrc, and add the following line to the end. Note to change <path_of_mobiletrack> to your real path.
```
export PYTHONPATH=<path_of_mobiletrack>:$PYTHONPATH
```
* Download the pre-trained networks  
  Download the network for [mobiletrack](https://drive.google.com/drive/folders/1kcYIb1WMDWo6_96cfN2YwpijcJZp1CIJ?usp=sharing) and put it in the directory set by "network_path" in "pytracking/evaluation/local.py". By default, it is set to pytracking/networks.

## Quick Start

#### TRAINING
* Modify [local.py](ltr/admin/local.py) to set the paths to datasets, results paths etc.
* Runing the following commands to train the mobiletrack. You can customize some parameters by modifying [mobiletrack.py](ltr/train_settings/mobiletrack/mobiletrack.py)
```bash
conda activate mobiletrack
cd ltr
python run_training.py mobiletrack mobiletrack
# for ddp
# python run_training_ddp.py mobiletrack mobiletrack --local_rank 4
```

#### Convert Model

* Convert model to onnx

```
conda activate mobiletrack
cd pysot_toolkit
python pytorch2onnx.py
```




