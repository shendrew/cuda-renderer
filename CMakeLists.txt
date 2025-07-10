cmake_minimum_required(VERSION 3.5.0)
project(cuda_renderer VERSION 0.0.1 LANGUAGES C CXX)

add_executable(cuda_renderer
    src/render.cpp
    src/matrix.cpp
    src/camera.cpp
    src/object.cpp
    src/pipeline/cpu_object_manager.cpp

)

# Set C++ standard for C++ files
set_property(TARGET cuda_renderer PROPERTY CXX_STANDARD 17)
set_property(TARGET cuda_renderer PROPERTY CXX_STANDARD_REQUIRED YES) # Enforce C++17


# Find SDL2
find_package(SDL2 REQUIRED)
# Find openGL and GLEW for dynamic loading
find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)

# Find CUDA (This will be crucial for CUDA development)
# This module sets up necessary variables and imported targets for CUDA.
option(ENABLE_CUDA "Enable CUDA support" ON)

if(ENABLE_CUDA)
    enable_language(CUDA)
endif()

find_package(CUDAToolkit)
if(ENABLE_CUDA AND CUDAToolkit_FOUND)
    message(STATUS "CUDA toolkit found! Enabling CUDA support.")

    set_property(TARGET cuda_renderer PROPERTY CUDA_STANDARD 17)
    set_property(TARGET cuda_renderer PROPERTY CUDA_STANDARD_REQUIRED YES)

    message(STATUS "Adding CUDA pipeline sources")
    # target_sources(cuda_renderer PRIVATE
    #     src/pipeline/cuda_object_manager.cpp
    #     src/pipeline/cuda/vertex_shader.cu
    # )
    target_compile_definitions(cuda_renderer PRIVATE USE_CUDA)
    target_link_libraries(cuda_renderer PRIVATE CUDA::cudart)
else()
    message(STATUS "Skipping CUDA support, toolkit not found or disabled.")
endif()

# --- Link Libraries and Include Directories ---

# Always prefer imported targets for linking and include directories as they are more robust.
# They carry transitive includes and link information.

target_link_libraries(cuda_renderer PRIVATE
    SDL2::SDL2
    GLEW::GLEW
    OpenGL::GL
    ${CUDA_LIBS}  # Link to the CUDA Runtime library if found
)

# General include directories for your project's headers.
# No need to manually add includes for SDL2, GLEW, OpenGL, CUDA if you use their imported targets correctly,
# as those targets usually handle their own include paths.
target_include_directories(cuda_renderer PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include" # For your own 'include' folder
)