cmake_minimum_required(VERSION 3.5)
project(final_project)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
foreach(PACKAGE
  ament_cmake
  ament_cmake_python
  rclcpp
  rclpy
  geometry_msgs
  nav_msgs
  sensor_msgs
  std_msgs
  cv_bridge
  visualization_msgs
  tf2_ros)

  find_package(${PACKAGE} REQUIRED)

endforeach()

# Include Cpp "include" directory
include_directories(include)

# Create Cpp executable (add your cpp files here when ready)
# add_executable(final_project_node src/final_project_node.cpp)
# ament_target_dependencies(final_project_node
#   rclcpp geometry_msgs sensor_msgs std_msgs
# )
# install(TARGETS
#   final_project_node
#   DESTINATION lib/${PROJECT_NAME})

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# Install Python executables
install(PROGRAMS
  scripts/detect3d.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

# Install config files
install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()