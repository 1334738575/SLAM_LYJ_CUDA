cmake_minimum_required(VERSION 3.20)
project(TEST_LYJ)

#################################setting##################################
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Zi")
# set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /Ob2")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Od /Ob0")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /DEBUG")
add_link_options(/DEBUG) #生成调试信息（测试在visual studio中有效）


#################################package##################################
enable_language(CUDA)
find_package(CUDA)
include_directories(${CUDA_INCLUDE_DIRS})
set(CMAKE_CUDA_ARCHITECTURES 89)

#################################library##################################
file(GLOB ALL_SRC *.h *.cu *.cpp)
add_library(${PROJECT_NAME} SHARED ${ALL_SRC})
target_include_directories(${PROJECT_NAME} PUBLIC
$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
$<INSTALL_INTERFACE:include>
)
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})

#################################test##################################
add_executable(test example/main.cpp)
target_include_directories(test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test ${PROJECT_NAME})

#################################install##################################
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/install)

# 安装库和头文件
install(TARGETS ${PROJECT_NAME}
    EXPORT TEST_LYJTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib     # DLL 在 Windows 下会被安装到 bin/
    RUNTIME DESTINATION bin     # 适用于 DLL 的运行时组件
    INCLUDES DESTINATION include
)

install(FILES test.h
DESTINATION include
PERMISSIONS OWNER_READ GROUP_READ WORLD_READ)

# set(TARGETNAME ${PROJECT_NAME}Targets)
# 导出目标配置
install(EXPORT TEST_LYJTargets
    FILE TEST_LYJTargets.cmake
    NAMESPACE TEST_LYJ::
    DESTINATION lib/cmake
)

# 生成包配置文件
include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/TEST_LYJConfig.cmake
    INSTALL_DESTINATION lib/cmake
)
write_basic_package_version_file(
    TEST_LYJConfigVersion.cmake
    VERSION 1.0.0
    COMPATIBILITY SameMajorVersion
)
install(FILES 
    ${CMAKE_CURRENT_BINARY_DIR}/TEST_LYJConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/TEST_LYJConfigVersion.cmake
    DESTINATION lib/cmake
)