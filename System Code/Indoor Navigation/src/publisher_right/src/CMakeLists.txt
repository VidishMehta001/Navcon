set(EXECUTABLE_TARGETS "")

add_executable(video_right_publisher video_right_publisher.cc)
list(APPEND EXECUTABLE_TARGETS video_right_publisher)
set_target_properties(video_right_publisher PROPERTIES OUTPUT_NAME video_right PREFIX "")

foreach(EXECUTABLE_TARGET IN LISTS EXECUTABLE_TARGETS)
    add_dependencies(${EXECUTABLE_TARGET} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
endforeach()
