ARG region

FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:1.12.0-gpu-py38-cu113-ubuntu20.04-sagemaker
RUN pip install tensorboard && mkdir /dependencies && cd /dependencies && wget https://smdataparallel.s3.amazonaws.com/binary/pytorch/1.12.0/cu113/2022-08-04/smdistributed_dataparallel-1.5.1-cp38-cp38-linux_x86_64.whl && pip install smdistributed_dataparallel-1.5.1-cp38-cp38-linux_x86_64.whl

