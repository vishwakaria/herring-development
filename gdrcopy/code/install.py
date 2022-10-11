import os
# out = os.system("apt-get update && apt install -y check libsubunit0 libsubunit-dev && apt install -y build-essential devscripts debhelper check libsubunit-dev fakeroot pkg-config dkms && sudo apt install -y nvidia-dkms-510 && wget https://github.com/NVIDIA/gdrcopy/archive/refs/tags/v2.3.tar.gz && tar -xf v2.3.tar.gz && cd gdrcopy-2.3/packages && CUDA=/usr/local/cuda ./build-deb-packages.sh && sudo dpkg -i gdrdrv-dkms_2.3-1_amd64.Ubuntu20_04.deb && sudo dpkg -i libgdrapi_2.3-1_amd64.Ubuntu20_04.deb && sudo dpkg -i gdrcopy-tests_2.3-1_amd64.Ubuntu20_04.deb  && sudo dpkg -i gdrcopy_2.3-1_amd64.Ubuntu20_04.deb")
# print(out)

out = os.system("sanity && copybw")
print(out)