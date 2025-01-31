cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(dcgan)
find_package(Torch REQUIRED)

add_executable(dcgan dcgan.cpp)
target_link_libraries(dcgan "${TORCH_LIBRARIES}")
set_property(TARGET dcgan PROPERTY CXX_STANDARD 17)

set(CMAKE_THREAD_LIBS_INIT -lpthread)
set(CMAKE_CXX_FLAGS  -pthread)
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_WIN32_THREADS_INIT 0)
set(CMAKE_USE_PTHREADS_INIT 1)
set(THREADS_PREFER_PTHREAD_FLAG ON)

