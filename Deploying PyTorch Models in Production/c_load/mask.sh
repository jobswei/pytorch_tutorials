mkdir build
cd build
. /opt/rh/devtoolset-9/enable
cmake -DCMAKE_PREFIX_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtcv/weiziyu/pytorch_tutorials/libtorch/libtorch_cu121 ..
cmake --build . --config Release