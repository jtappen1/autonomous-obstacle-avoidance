cmake_minimum_required(VERSION 3.5)
project(final_project)

# ── C++ standard ──────────────────────────────────────────────────────────────
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ── Release optimisation flags ────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-O3 -march=native -ffast-math -funroll-loops -DNDEBUG)

# ── Find dependencies ─────────────────────────────────────────────────────────
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED)

# ── Include directories ───────────────────────────────────────────────────────
include_directories(include)

# ── Tracker library: UKF + track manager + ROS node logic ────────────────────
add_library(tracker_lib STATIC
  src/UKF.cpp
  src/track_manager.cpp
  src/tracker_node.cpp
)

target_include_directories(tracker_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(tracker_lib
  Eigen3::Eigen
  ${OpenCV_LIBS}
)

ament_target_dependencies(tracker_lib
  rclcpp
  sensor_msgs
  vision_msgs
  visualization_msgs
  geometry_msgs
  cv_bridge
)

# ── Tracker node executable (only main.cpp) ───────────────────────────────────
add_executable(tracker_node src/main.cpp)

target_include_directories(tracker_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(tracker_node tracker_lib)

ament_target_dependencies(tracker_node
  rclcpp
)

# ── Install C++ targets ───────────────────────────────────────────────────────
install(TARGETS
  tracker_lib
  tracker_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

# ── Python modules ────────────────────────────────────────────────────────────
ament_python_install_package(${PROJECT_NAME})

# ── Python executables ────────────────────────────────────────────────────────
install(PROGRAMS
  scripts/vision/detect3d.py
  scripts/vision/detections.py
  scripts/trackers/tracker.py
  DESTINATION lib/${PROJECT_NAME}
)

# ── Launch / config files ─────────────────────────────────────────────────────
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

# ── Tests ─────────────────────────────────────────────────────────────────────
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()