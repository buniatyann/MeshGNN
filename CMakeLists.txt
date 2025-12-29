cmake_minimum_required(VERSION 3.14)
project(MeshGNN VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-O3 -march=native)
    endif()
endif()

# Find TBB for parallel execution policies (required for std::execution)
find_package(TBB QUIET)

# Library sources
set(MESHGNN_SOURCES
    src/vector.cpp
    src/matrix.cpp
    src/mesh.cpp
    src/graph.cpp
    src/random.cpp
    src/feature_extraction.cpp
    src/mesh_processor.cpp
    src/gnn/layer.cpp
    src/gnn/pipeline.cpp
    src/gnn/training.cpp
)

# Create library
add_library(meshgnn ${MESHGNN_SOURCES})
target_include_directories(meshgnn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(TBB_FOUND)
    target_link_libraries(meshgnn PUBLIC TBB::tbb)
endif()

# Main executable
add_executable(meshgnn_demo main.cpp)
target_link_libraries(meshgnn_demo PRIVATE meshgnn)

# Testing
option(MESHGNN_BUILD_TESTS "Build unit tests" ON)

if(MESHGNN_BUILD_TESTS)
    enable_testing()

    # Try to find GTest
    find_package(GTest QUIET)

    if(GTest_FOUND)
        add_executable(meshgnn_tests
            tests/test_vector.cpp
            tests/test_matrix.cpp
            tests/test_mesh.cpp
            tests/test_gnn.cpp
        )
        target_link_libraries(meshgnn_tests PRIVATE meshgnn GTest::gtest GTest::gtest_main)

        include(GoogleTest)
        gtest_discover_tests(meshgnn_tests)
    else()
        message(STATUS "GTest not found. Tests will not be built.")
    endif()
endif()

# Install targets
install(TARGETS meshgnn
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/gnnmath
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)
