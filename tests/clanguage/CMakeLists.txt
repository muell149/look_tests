cmake_minimum_required(VERSION 3.10.2)

project(example-app)

#set(Torch_DIR "/app/libtorch/share/cmake/Torch")

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

include_directories(${Torch_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)

file(GLOB SOURCES "src/*.cpp")


#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

file(GLOB SOURCES "src/*.cpp")

add_executable(example-app ${SOURCES})
target_link_libraries(example-app ${OpenCV_LIBS} ${TORCH_LIBRARIES})