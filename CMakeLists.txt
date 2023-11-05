cmake_minimum_required(VERSION 3.12)
project(FluidSim)

# Set C/C++ standard
set(CMAKE_CXX_STANDARD 17)

# Find required packages
# find_package(MPI REQUIRED)
# find_package(OpenMP REQUIRED)
find_package(OpenGL REQUIRED)
# find_package(CUDA REQUIRED)

# Add GLFW project
add_subdirectory(external/glfw)

# Add source files
set(SOURCES
	# Fluid sim source files
	src/main.cpp
	src/render.cpp
	src/shapes.cpp

	# Glad loader source
	external/glad/src/glad.c
)

# Add CUDA source files
set(CUDA_SOURCES
	# src/kernel.cu
)

# Add sources to executable
add_executable(fluidsim ${SOURCES} ${CUDA_SOURCES})

# Add include directories
target_include_directories(fluidsim
	PRIVATE
	external/glad/include/
	external/glm/
)

# Add libraries
target_link_libraries(fluidsim
	OpenGL::GL
	glfw
)

# Add compiler flags for C/C++
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffile-prefix-map=${CMAKE_SOURCE_DIR}/=/")
endif()

# Add compiler flags for CUDA
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3;--use_fast_math)

