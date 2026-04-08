cmake_minimum_required(VERSION 3.16)

project(bitserial_cnn_accelerator_sim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(bitserial_sim
    src/Accelerator.cpp
    src/ConvLayer.cpp
    src/Lane.cpp
    src/main.cpp
    src/Memory.cpp
    src/MNISTReader.cpp
    src/PEGroup.cpp
    src/ReferenceConv.cpp
    src/Scheduler.cpp
    src/Task.cpp
    src/WeightBroadcaster.cpp
    src/WorkQueue.cpp
)

target_include_directories(bitserial_sim PRIVATE include)
