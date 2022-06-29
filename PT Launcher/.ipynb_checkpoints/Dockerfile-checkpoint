ARG region

# Download base PT DLC. Note that this notebook requires a SM DLC with >= PT 1.10.2
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.11.0-gpu-py38-cu113-ubuntu20.04-sagemaker

ARG WORK_DIR="ptddp_build"
RUN rm -rf "/ptddp_build";
WORKDIR $WORK_DIR
RUN pwd; pip --no-cache-dir --no-cache install git+https://github.com/vishwakaria/sagemaker-pytorch-training-toolkit/; ls; \
    cd sagemaker-pytorch-training-toolkit; ls; \
    python setup.py; \
    cd ..; pip --no-cache-dir --no-cache install git+https://github.com/vishwakaria/sagemaker-python-sdk/; \
    cd sagemaker-python-sdk; \
    python setup.py; \
    cd ../..; ls; rm -rf $WORK_DIR;
