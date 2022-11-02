cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

include(local_conf.cmake OPTIONAL)

project(Fisher_Stats VERSION 0.1 DESCRIPTION "Calculates the fisher Matrix of ML models" LANGUAGES CXX)


set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
	set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
		STRING "Choose the type of build." FORCE)
	# Set the possible values of build type for cmake-gui
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
		"Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(Fisher_Matrix src/Fisher_Matrix.cpp include/Fisher_Matrix.hpp)


target_compile_definitions(Fisher_Matrix PUBLIC $<$<CONFIG:Debug>:_DEBUG=1>)

find_package(Torch REQUIRED)
find_package(TBB CONFIG REQUIRED)
find_package(oneDPL CONFIG REQUIRED)




target_include_directories(Fisher_Matrix PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	"$<INSTALL_INTERFACE:include>")

target_link_libraries(Fisher_Matrix PUBLIC  "${TORCH_LIBRARIES}"  )
target_link_libraries(Fisher_Matrix PUBLIC  ${TBB_IMPORTED_TARGETS} oneDPL)


install(TARGETS Fisher_Matrix EXPORT Fisher_MatrixTargets DESTINATION lib)
install(DIRECTORY include/ DESTINATION include/)

install(EXPORT Fisher_MatrixTargets
	FILE Fisher_MatrixTargets.cmake
	DESTINATION lib/cmake/Fisher_Matrix
	)
include(CMakePackageConfigHelpers)
# generate the config file that is includes the exports
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
	"${CMAKE_CURRENT_BINARY_DIR}/Fisher_MatrixConfig.cmake"
	INSTALL_DESTINATION "lib/cmake/Fisher_Matrix"
	NO_SET_AND_CHECK_MACRO
	NO_CHECK_REQUIRED_COMPONENTS_MACRO
	)
write_basic_package_version_file(
	"${CMAKE_CURRENT_BINARY_DIR}/Fisher_MatrixConfigVersion.cmake"
	VERSION "0.1.1.0"
	COMPATIBILITY AnyNewerVersion
	)
install(FILES
	${CMAKE_CURRENT_BINARY_DIR}/Fisher_MatrixConfig.cmake
	${CMAKE_CURRENT_BINARY_DIR}/Fisher_MatrixConfigVersion.cmake
	DESTINATION lib/cmake/Fisher_Matrix
	)
export(EXPORT Fisher_MatrixTargets
	FILE "${CMAKE_CURRENT_BINARY_DIR}/Fisher_MatrixTargets.cmake"
	)
