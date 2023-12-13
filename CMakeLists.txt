cmake_minimum_required(VERSION 3.12)
project(FluidSim LANGUAGES C CXX CUDA)

# Set C/C++ standard
set(CMAKE_CXX_STANDARD 20)

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
	src/simulation.cpp

	# Fluid sim header files
	src/config.h
	src/common.h
	src/render.h
	src/shapes.h
	src/simulation.h
	src/radix_sort.h

	external/gpu-radix-sort/radix_sort/scan.h
	external/gpu-radix-sort/radix_sort/sort.h

	# Glad loader source file
	external/glad/src/glad.c

	# ImGui headers
	external/imgui/imconfig.h
	external/imgui/imgui.h
	external/imgui/imgui_internal.h
	external/imgui/imstb_rectpack.h
	external/imgui/imstb_textedit.h
	external/imgui/imstb_truetype.h
	external/imgui/backends/imgui_impl_glfw.h
	external/imgui/backends/imgui_impl_opengl3.h
	# ImGui source files
	external/imgui/imgui.cpp
	external/imgui/imgui_demo.cpp
	external/imgui/imgui_draw.cpp
	external/imgui/imgui_tables.cpp
	external/imgui/imgui_widgets.cpp
	external/imgui/backends/imgui_impl_glfw.cpp
	external/imgui/backends/imgui_impl_opengl3.cpp
)

# Add CUDA source files
set(CUDA_SOURCES
	src/simulation.cu
	src/radix_sort.cu

	external/gpu-radix-sort/radix_sort/scan.cu
	external/gpu-radix-sort/radix_sort/sort.cu
)

# Add sources to executable
add_executable(fluidsim ${SOURCES} ${CUDA_SOURCES})
add_executable(test_radix_sort
	tests/test_radix_sort.cpp
	src/radix_sort.cu
	external/gpu-radix-sort/radix_sort/sort.cu
	external/gpu-radix-sort/radix_sort/scan.cu
)

# Set CUDA arch to compute 8.6 (for RTX 3060)
set(CMAKE_CUDA_ARCHITECTURES "sm_86") 

set_target_properties(fluidsim PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(test_radix_sort PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Add include directories
target_include_directories(fluidsim
	PUBLIC
	src/

	PRIVATE
	external/glad/include/
	external/glm/
	external/imgui/
	external/imgui/backends/

	external/gpu-radix-sort/
)
target_include_directories(test_radix_sort
	PUBLIC
	src/

	PRIVATE
	external/glm/

	external/gpu-radix-sort/
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
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3;-use_fast_math)

