# Detect NEAT API version: legacy Session API (neat/session.h) vs new Graph API.
# example_utils.cpp uses simaai::neat::Session which was removed in the graph-based API.
if(EXISTS "${CMAKE_SYSROOT}/usr/include/neat/session.h"
   OR EXISTS "${CMAKE_FIND_ROOT_PATH}/usr/include/neat/session.h"
   OR EXISTS "/usr/include/neat/session.h")
  set(SIMA_NEAT_LEGACY_API TRUE)
else()
  set(SIMA_NEAT_LEGACY_API FALSE)
  message(STATUS "SimaNeatApps: new Graph API detected — skipping example_utils.cpp (Session API removed)")
endif()

if(SIMA_NEAT_LEGACY_API)
  add_library(sima_neat_apps_support_runtime STATIC
    runtime/asset_utils.cpp
    runtime/config_utils.cpp
    runtime/example_utils.cpp
    object_detection/obj_detection_utils.cpp
  )
else()
  add_library(sima_neat_apps_support_runtime STATIC
    runtime/asset_utils.cpp
    runtime/config_utils.cpp
    object_detection/obj_detection_utils.cpp
  )
  target_compile_definitions(sima_neat_apps_support_runtime PUBLIC SIMA_NEAT_GRAPH_API=1)
endif()
add_library(SimaNeatApps::support_runtime ALIAS sima_neat_apps_support_runtime)

target_link_libraries(sima_neat_apps_support_runtime
  PUBLIC
    SimaNeat::sima_neat
)

target_link_libraries(sima_neat_apps_support_runtime PUBLIC nlohmann_json::nlohmann_json)

target_include_directories(sima_neat_apps_support_runtime
  PUBLIC
    ${PROJECT_SOURCE_DIR}
)

if (SIMANEAT_APPS_BUILD_OPTIVIEW_SUPPORT)
  add_library(sima_neat_apps_support_optiview STATIC
    optiview/graphpipes_optiview_helpers.cpp
  )
  add_library(SimaNeatApps::support_optiview ALIAS sima_neat_apps_support_optiview)
  target_link_libraries(sima_neat_apps_support_optiview
    PUBLIC
      SimaNeatApps::support_runtime
      SimaNeat::sima_neat
  )
  target_include_directories(sima_neat_apps_support_optiview
    PUBLIC
      ${PROJECT_SOURCE_DIR}
  )
endif()

if (BUILD_TESTING)
  add_library(sima_neat_apps_support_testing STATIC
    runtime/config_utils.cpp
    testing/metadata_json_listener.cpp
    testing/test_config.cpp
    testing/test_process.cpp
  )
  add_library(SimaNeatApps::support_testing ALIAS sima_neat_apps_support_testing)
  target_link_libraries(sima_neat_apps_support_testing
    PUBLIC
      nlohmann_json::nlohmann_json
  )
  target_include_directories(sima_neat_apps_support_testing
    PUBLIC
      ${PROJECT_SOURCE_DIR}
  )
endif()
