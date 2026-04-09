cmake_minimum_required(VERSION 3.16)

project(bitserial_cnn_accelerator_sim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(BITSERIAL_COMMON_SOURCES
    src/Accelerator.cpp
    src/BenchmarkSuite.cpp
    src/ConvLayer.cpp
    src/Lane.cpp
    src/Memory.cpp
    src/MNISTReader.cpp
    src/PEGroup.cpp
    src/ReferenceConv.cpp
    src/Scheduler.cpp
    src/Task.cpp
    src/WeightBroadcaster.cpp
    src/WorkQueue.cpp
)

add_library(bitserial_sim_core STATIC
    ${BITSERIAL_COMMON_SOURCES}
)

target_include_directories(bitserial_sim_core PUBLIC include)

add_executable(bitserial_sim
    src/main.cpp
)

target_link_libraries(bitserial_sim PRIVATE bitserial_sim_core)

add_executable(bitserial_bench
    src/benchmark_main.cpp
)

target_link_libraries(bitserial_bench PRIVATE bitserial_sim_core)

enable_testing()

add_executable(benchmark_helper_tests
    tests/benchmark_helper_tests.cpp
)
target_link_libraries(benchmark_helper_tests PRIVATE bitserial_sim_core)

add_executable(benchmark_regression_tests
    tests/benchmark_regression_tests.cpp
)
target_link_libraries(benchmark_regression_tests PRIVATE bitserial_sim_core)

add_executable(benchmark_realworld_tests
    tests/benchmark_realworld_tests.cpp
)
target_link_libraries(benchmark_realworld_tests PRIVATE bitserial_sim_core)

add_executable(zero_skip_tests
    tests/zero_skip_tests.cpp
)
target_link_libraries(zero_skip_tests PRIVATE bitserial_sim_core)

add_test(NAME benchmark_helper_tests COMMAND benchmark_helper_tests)
add_test(NAME benchmark_regression_tests COMMAND benchmark_regression_tests)
add_test(NAME benchmark_realworld_tests COMMAND benchmark_realworld_tests)
add_test(NAME zero_skip_tests COMMAND zero_skip_tests)
set_tests_properties(
    benchmark_helper_tests
    benchmark_regression_tests
    benchmark_realworld_tests
    zero_skip_tests
    PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
