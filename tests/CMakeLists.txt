cmake_minimum_required(VERSION 3.10.2)

project(SingularValueDecomposition)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)

file(GLOB SOURCES "src/*.cpp")

add_executable(svd ${SOURCES})
target_link_libraries(svd ${OpenCV_LIBS})
