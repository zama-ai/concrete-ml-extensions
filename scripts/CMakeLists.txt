cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(tfhe_cuda_backend LANGUAGES CXX)

# See if the minimum CUDA version is available. If not, only enable documentation building.
set(MINIMUM_SUPPORTED_CUDA_VERSION 10.0)
include(CheckLanguage)
# See if CUDA is available
check_language(CUDA)
# If so, enable CUDA to check the version.
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
endif()
# If CUDA is not available, or the minimum version is too low do not build
if(NOT CMAKE_CUDA_COMPILER)
  message(FATAL_ERROR "Cuda compiler not found.")
endif()
