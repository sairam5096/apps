foreach(example IN ITEMS
    detr-object-detector
    single-stream-object-detector
    yolo26-object-detector
    yolov8n-detection-pipeline
)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${example}/src/cpp/CMakeLists.txt")
    add_subdirectory(${example}/src/cpp ${example})
  endif()
endforeach()
add_subdirectory(multi-stream-object-detector/src/cpp multi-stream-object-detector)

# yolov11s-1280-detection-pipeline uses the legacy cpp/ layout (not src/cpp/)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/yolov11s-1280-detection-pipeline/cpp/CMakeLists.txt")
  add_subdirectory(yolov11s-1280-detection-pipeline/cpp yolov11s-1280-detection-pipeline)
endif()
