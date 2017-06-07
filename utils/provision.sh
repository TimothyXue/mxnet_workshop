#!/bin/bash
cd /home/ubuntu/src/mxnet_workshop/




export LD_LIBRARY_PATH=/home/ubuntu/src/torch/install/lib:/home/ubuntu/src/torch/install/lib:/home/ubuntu/src/cntk/bindings/python/cntk/libs:/usr/local/cuda/lib64:/usr/local/lib:/usr/lib:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/mpi/lib:/home/ubuntu/src/mxnet/mklml_lnx_2017.0.1.20161005/lib:$LD_LIBRARY_PATH

export PATH=/usr/local/cuda/bin:/home/ubuntu/src/torch/install/bin:/home/ubuntu/src/torch/install/bin:/usr/local/cuda/bin:/usr/local/bin:/opt/aws/bin:/home/ubuntu/src/cntk/bin:/usr/local/mpi/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:$PATH

jupyter nbextension enable --py widgetsnbextension
jupyter-notebook
