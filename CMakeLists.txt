cmake_minimum_required(VERSION 3.30)
set(CMAKE_CXX_STANDARD 23)
project(tetris-ai VERSION 1.0.1)
# set(CMAKE_C_COMPILER "/usr/bin/cc")
# set(CMAKE_CXX_COMPILER "/usr/bin/c++")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add all source files
file(GLOB_RECURSE SOURCES "Gameplay/*.cpp" "AI/*.cpp" "Helpers/*.cpp")
add_executable(tetris-ai ${SOURCES})

# Enable AddressSanitizer
target_compile_options(tetris-ai PRIVATE -fsanitize=address -g)
target_link_options(tetris-ai PRIVATE -fsanitize=address -g)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")

set(CMAKE_BUILD_TYPE Debug)

# Variables storing SDL framework locations
set(SDL2 /Library/Frameworks/SDL2.framework)
set(SDL2_image /Library/Frameworks/SDL2_image.framework)
set(SDL2_ttf /Library/Frameworks/SDL2_ttf.framework)

# Set MLX paths
# set(MLX_INCLUDE_DIR "/opt/homebrew/Cellar/mlx/0.21.1/include")
# set(MLX_LIBRARY_DIR "/opt/homebrew/Cellar/mlx/0.21.1/lib")
# set(MLX_METAL_PATH "${MLX_LIBRARY_DIR}/mlx.metallib")

# Find MLX library
# find_library(MLX_LIBRARY mlx PATHS ${MLX_LIBRARY_DIR} REQUIRED)

# if(NOT MLX_LIBRARY)
#     message(FATAL_ERROR "MLX library not found. Please make sure MLX is installed correctly.")
# endif()

find_package(MLX CONFIG REQUIRED)

target_link_libraries(
  tetris-ai PRIVATE
  ${SDL2}/Versions/A/SDL2
  ${SDL2_image}/Versions/A/SDL2_image
  ${SDL2_ttf}/Versions/A/SDL2_ttf
  mlx
  # ${MLX_LIBRARY}
  # ./AI PRIVATE mlx
  # ./Gameplay PRIVATE mlx
  # ./Helpers PRIVATE mlx
)

target_include_directories(
  tetris-ai PRIVATE
  ${SDL2}/Versions/A/Headers
  ${SDL2_image}/Versions/A/Headers
  ${SDL2_ttf}/Versions/A/Headers
  # ${MLX_INCLUDE_DIR}
  ./Gameplay
  ./AI
  ./Helpers
)

# add_definitions(-DMETAL_PATH="${MLX_METAL_PATH}")

add_custom_command(
  TARGET tetris-ai POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  ${CMAKE_SOURCE_DIR}/resources/lmroman17-regular.otf
  ${CMAKE_BINARY_DIR}/resources/lmroman17-regular.otf
)
