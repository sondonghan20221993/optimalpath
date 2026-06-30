cmake_minimum_required(VERSION 3.0.2)
project(drone_runner)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  pb_core
  utils_msgs
)

find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(octomap REQUIRED)
find_package(jsoncpp REQUIRED)
find_package(OpenMP REQUIRED)

# Find Glog
find_package(PkgConfig REQUIRED)
pkg_check_modules(GLOG libglog)
if(NOT GLOG_FOUND)
  find_library(GLOG_LIBRARIES glog)
  find_path(GLOG_INCLUDE_DIRS glog/logging.h)
endif()

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OCTOMAP_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/../pb_core/include
)

add_executable(drone_runner_node
  src/drone_runner_node.cpp
)

target_link_libraries(drone_runner_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${EIGEN3_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${OCTOMAP_LIBRARIES}
  jsoncpp_lib
  ${GLOG_LIBRARIES}
  OpenMP::OpenMP_CXX
)

if(GLOG_INCLUDE_DIRS)
  target_include_directories(drone_runner_node PRIVATE ${GLOG_INCLUDE_DIRS})
endif()
