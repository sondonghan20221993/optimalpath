cmake_minimum_required(VERSION 3.0.2)
project(drone_runner)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  pb_core
  utils_msgs
)

catkin_package()

include_directories(${catkin_INCLUDE_DIRS})

add_executable(drone_runner_node src/drone_runner_node.cpp)
target_link_libraries(drone_runner_node ${catkin_LIBRARIES})
