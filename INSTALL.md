# Installation

This document describes how to install Video-LFB and its dependencies.

**Requirements:**

- NVIDIA GPU, Linux, Python2
- Caffe2, various standard Python packages.
- We used CUDA and cuDNN in our experiments.

### Video-LFB
Clone the Video-LFB repository.
```
# VIDEO_LFB=/path/to/install/video-long-term-feature-banks
git clone https://github.com/facebookresearch/video-long-term-feature-banks $VIDEO_LFB
```

Add this repository to `$PYTHONPATH`.
```Shell
export PYTHONPATH=/path/to/video-long-term-feature-banks/lib:$PYTHONPATH
```

### Caffe2
Caffe2 is now part of PyTorch.
Please follow [PyTorch official instructions](https://github.com/pytorch/pytorch#from-source) to install from source.
**The only additional step is to add a customized operator (`affine_nd_op`) to the source code after cloning the source:**
```Shell
git clone --recursive https://github.com/pytorch/pytorch
rm -rf pytorch/caffe2/video
cp -r [path to video-long-term-feature-banks]/caffe2_customized_ops/video pytorch/caffe2/
```

#### Installation Example

In case it's still not clear,
in the following we provide the exact steps we performed to install Caffe2.
We used an Ubuntu machine with CUDA 9.0 and cuDNN 7.5,
without root permission.

```Shell
conda create -n video-lfb python=2.7
source activate video-lfb

conda install numpy pyyaml mkl mkl-include setuptools cmake cffi typing
conda install -c pytorch magma-cuda90
# install nccl by source, if not installed globally
# UPDATE: maybe we don't need to as PyTorch compiles nccl by default
#git clone https://github.com/NVIDIA/nccl.git
#cd nccl
#make src.build CUDA_HOME=<path to cuda install> -j 20
#cp -r build/include/* $CONDA_PREFIX/include
#cp -r build/lib/* $CONDA_PREFIX/lib
#cd ..

git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
git checkout v1.3.1
git submodule sync
git submodule update --init --recursive
rm -r caffe2/videos
cp -r [path to video-long-term-feature-banks]/caffe2_customized_ops/video caffe2/

export PYTHONPATH=/path/to/video-long-term-feature-banks/lib:$PYTHONPATH
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export CUDA_HOME=[path to CUDA 9.0]
#export CUDNN_LIB_DIR=[path to cnDNN]/lib64
#export CUDNN_INCLUDE_DIR=[path to cnDNN]/include
python2 setup.py install

conda install --yes protobuf
conda install --yes future
conda install --yes networkx
pip install enum34
pip install sklearn
```
