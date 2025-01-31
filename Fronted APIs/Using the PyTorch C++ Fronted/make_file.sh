NAME=$1
echo "cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project($NAME)
find_package(Torch REQUIRED)

add_executable($NAME $NAME.cpp)
target_link_libraries($NAME \"\${TORCH_LIBRARIES}\")
set_property(TARGET $NAME PROPERTY CXX_STANDARD 17)

set(CMAKE_THREAD_LIBS_INIT "-lpthread")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_WIN32_THREADS_INIT 0)
set(CMAKE_USE_PTHREADS_INIT 1)
set(THREADS_PREFER_PTHREAD_FLAG ON)
" > CMakeLists.txt

# bash make.sh
mkdir build
cd build
. /opt/rh/devtoolset-9/enable
cmake -DCMAKE_PREFIX_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtcv/weiziyu/pytorch_tutorials/libtorch/libtorch_cu121 ..
cmake --build . --config Release
