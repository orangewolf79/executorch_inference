cmake_minimum_required(VERSION 3.15)
project(main LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# executorch subdir
add_subdirectory("third_party/executorch")

set(Torch_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/libtorch/share/cmake/Torch") # for libtorch
find_package(Torch REQUIRED) # also for libtorch

add_executable(main source/main.cpp)  

# executorch + libtorch core libraries
target_link_libraries(main
    PRIVATE
      executorch                  
      ${TORCH_LIBRARIES}
)

# directories containing header files
target_include_directories(main
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/executorch/include
    ${TORCH_INCLUDE_DIRS}
)