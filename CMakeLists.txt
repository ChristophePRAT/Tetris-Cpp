cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 23)
project(tetris-ai VERSION 1.0.0)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add all source files
file(GLOB_RECURSE SOURCES "Gameplay/*.cpp" "AI/*.cpp" "Helpers/*.cpp")
add_executable(tetris-ai ${SOURCES})

# Enable AddressSanitizer
target_compile_options(tetris-ai PRIVATE -fsanitize=address)
target_link_options(tetris-ai PRIVATE -fsanitize=address)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")

set(CMAKE_BUILD_TYPE Debug)

# Variables storing SDL framework locations
set(SDL2 /Library/Frameworks/SDL2.framework)
set(SDL2_image /Library/Frameworks/SDL2_image.framework)
set(SDL2_ttf /Library/Frameworks/SDL2_ttf.framework)

# Set MLX paths
set(MLX_INCLUDE_DIR "/usr/local/include")
set(MLX_LIBRARY_DIR "/usr/local/lib")
set(MLX_METAL_PATH "${MLX_LIBRARY_DIR}/mlx.metallib")

# Find MLX library
find_library(MLX_LIBRARY mlx PATHS ${MLX_LIBRARY_DIR} REQUIRED)

if(NOT MLX_LIBRARY)
    message(FATAL_ERROR "MLX library not found. Please make sure MLX is installed correctly.")
endif()

target_link_libraries(
  tetris-ai PRIVATE
  ${SDL2}/Versions/A/SDL2
  ${SDL2_image}/Versions/A/SDL2_image
  ${SDL2_ttf}/Versions/A/SDL2_ttf
  ${MLX_LIBRARY}
)

target_include_directories(
  tetris-ai PRIVATE
  ${SDL2}/Versions/A/Headers
  ${SDL2_image}/Versions/A/Headers
  ${SDL2_ttf}/Versions/A/Headers
  ${MLX_INCLUDE_DIR}
  ./Gameplay
  ./AI
  ./Helpers
)

add_definitions(-DMETAL_PATH="${MLX_METAL_PATH}")

add_custom_command(
  TARGET tetris-ai POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  ${CMAKE_SOURCE_DIR}/resources/lmroman17-regular.otf
  ${CMAKE_BINARY_DIR}/resources/lmroman17-regular.otf
)
