cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
set(CMAKE_PREFIX_PATH "/home/harry/.local/lib/python3.10/site-packages" ${CMAKE_PREFIX_PATH})

project(HashEncoderExtension LANGUAGES CXX CUDA)

# Find PyTorch
find_package(Torch REQUIRED)

# Add the C++ source files
add_library(hash_encoder SHARED
    hashencoder.cpp
)

# Set the necessary include directories for the extension
target_include_directories(hash_encoder PRIVATE
    ${TORCH_INCLUDE_DIRS}
)

# Link the extension library against PyTorch
target_link_libraries(hash_encoder PRIVATE
    ${TORCH_LIBRARIES}
)

# Specify the C++ standard to be used
set_property(TARGET hash_encoder PROPERTY CXX_STANDARD 14)

# Set the necessary flags for CUDA compilation
set_target_properties(hash_encoder PROPERTIES CUDA_STANDARD 14)

# Set the output name and location for the extension
set_target_properties(hash_encoder PROPERTIES
    OUTPUT_NAME "_backend"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build"
)

# Copy the compiled extension to the desired location (Python package folder)
add_custom_command(TARGET hash_encoder POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    $<TARGET_FILE:hash_encoder>
    "${CMAKE_CURRENT_SOURCE_DIR}/_backend/"
)
