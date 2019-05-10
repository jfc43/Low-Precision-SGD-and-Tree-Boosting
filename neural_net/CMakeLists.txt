set (CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-O3")

add_executable(neural-net neural-net.cpp)
add_executable(neural-net-low-precision neural-net-low-precision.cpp)
