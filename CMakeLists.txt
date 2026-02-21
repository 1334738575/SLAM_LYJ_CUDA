cmake_minimum_required(VERSION 3.20)
project(CUDA_LYJ)
# set(CMAKE_CUDA_ARCHITECTURES 89)
# set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/bin/nvcc.exe")
# project(TEST_LYJ LANGUAGES CXX CUDA)

#################################setting##################################
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Zi")
# set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /Ob2")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Od /Ob0")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /DEBUG")
add_link_options(/DEBUG) #生成调试信息（测试在visual studio中有效）

set(${PROJECT_NAME}_LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/Output)
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install)
option(TEST_TEXTURE "test texture" OFF)

#################################package##################################
enable_language(CUDA)
find_package(CUDA)
# message(FATAL_ERROR ${CUDA_FOUND})
include_directories(${CUDA_INCLUDE_DIRS})
set(CMAKE_CUDA_ARCHITECTURES 89)
if(TEST_TEXTURE)
    set(${PROJECT_NAME}_INCLUDE_DIRS)
    set(${PROJECT_NAME}_LIBRARIES)
    set(${PROJECT_NAME}_BINARY_DIRS)
    set(OpenCV_DIR "D:/SLAM_LYJ/dependencies/Opencv/lib")
    find_package(OpenCV)
    if(NOT OpenCV_FOUND)
        message(FATAL_ERROR "OpenCV not found.")
    endif()
    list(APPEND ${PROJECT_NAME}_INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS})
    list(APPEND ${PROJECT_NAME}_LIBRARIES ${OpenCV_LIBS})
    list(APPEND ${PROJECT_NAME}_BINARY_DIRS "${OpenCV_INSTALL_PATH}/bin;")
    set(Eigen3_DIR "D:/SLAM_LYJ/dependencies/Eigen/share/eigen3/cmake")
    find_package(Eigen3)
    list(APPEND ${PROJECT_NAME}_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIRS})
endif()

#################################library##################################
file(GLOB ALL_SRC *.h *.cu *.cpp)
add_library(${PROJECT_NAME} SHARED ${ALL_SRC})
target_include_directories(${PROJECT_NAME} PUBLIC
$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
$<INSTALL_INTERFACE:include>
)
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})

# ################################output##################################
if(DEFINED ${PROJECT_NAME}_LIBRARY_OUTPUT_PATH)
    set_target_properties(${PROJECT_NAME} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${${PROJECT_NAME}_LIBRARY_OUTPUT_PATH}
        ARCHIVE_OUTPUT_DIRECTORY ${${PROJECT_NAME}_LIBRARY_OUTPUT_PATH}
        RUNTIME_OUTPUT_DIRECTORY ${${PROJECT_NAME}_LIBRARY_OUTPUT_PATH}
    )
endif()

#################################test##################################
add_executable(${PROJECT_NAME}Bin example/main.cpp)
target_include_directories(${PROJECT_NAME}Bin PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${PROJECT_NAME}Bin PRIVATE ${PROJECT_NAME})
if(TEST_TEXTURE)
    target_include_directories(${PROJECT_NAME}Bin PRIVATE ${${PROJECT_NAME}_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME}Bin PRIVATE ${${PROJECT_NAME}_LIBRARIES})
    if(MSVC)
        set_target_properties(${PROJECT_NAME}Bin PROPERTIES VS_DEBUGGER_ENVIRONMENT "PATH=${${PROJECT_NAME}_BINARY_DIRS}")
    endif()
endif()

#################################install##################################

# 安装库和头文件
install(TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}Targets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib     # DLL 在 Windows 下会被安装到 bin/
    RUNTIME DESTINATION bin     # 适用于 DLL 的运行时组件
    INCLUDES DESTINATION include
)

install(FILES CUDAInclude.h CUDADefines.h CUDACommon.h
DESTINATION include
PERMISSIONS OWNER_READ GROUP_READ WORLD_READ)

# set(TARGETNAME ${PROJECT_NAME}Targets)
# 导出目标配置
install(EXPORT ${PROJECT_NAME}Targets
    FILE ${PROJECT_NAME}Targets.cmake
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION lib/cmake
)

# 生成包配置文件
include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    INSTALL_DESTINATION lib/cmake
)
write_basic_package_version_file(
    ${PROJECT_NAME}ConfigVersion.cmake
    VERSION 1.0.0
    COMPATIBILITY SameMajorVersion
)
install(FILES 
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    DESTINATION lib/cmake
)