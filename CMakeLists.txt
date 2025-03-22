cmake_minimum_required(VERSION 3.5.0)
project(cuda_renderer VERSION 0.0.1 LANGUAGES C CXX)

add_executable(cuda_renderer
    src/render.cpp
    src/matrix.cpp
    src/camera.cpp

)
set_property(TARGET cuda_renderer PROPERTY CXX_STANDARD 17)

find_package(SDL2 REQUIRED)
if(SDL2_FOUND)
    include_directories(include ${SDL2_INCLUDE_DIRS})
    target_link_libraries(cuda_renderer ${SDL2_LIBRARIES})
else()
    message(FATAL_ERROR "SDL2 library not found!")
endif()

target_include_directories(cuda_renderer PRIVATE "include")