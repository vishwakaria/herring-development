ARG region

# Download base PT DLC. Note that this notebook requires a SM DLC with >= PT 1.10.2
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.11.0-gpu-py38-cu113-ubuntu20.04-sagemaker

ARG WORK_DIR="ptddp_build"
#RUN rm -rf "/ptddp_build";
WORKDIR $WORK_DIR
RUN pwd; git clone https://github.com/vishwakaria/sagemaker-pytorch-training-toolkit/; ls; cd /ptddp_build/sagemaker-pytorch-training-toolkit; ls; python setup.py bdist_wheel; pip install --force-reinstall --no-cache-dir --no-cache sagemaker-pytorch-training==2.5.0; cd ..; ls ; \
git clone https://github.com/vishwakaria/sagemaker-python-sdk/; ls; cd /ptddp_build/sagemaker-python-sdk; python setup.py bdist_wheel; pip install --force-reinstall --no-cache-dir --no-cache sagemaker-python-sdk==2.93.0; cd ../..;