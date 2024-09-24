cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 23)

project(tetris-ai VERSION 1.0.0)

# Add all source files
file(GLOB_RECURSE SOURCES "Gameplay/*.cpp" "AI/*.cpp" "Helpers/*.cpp")

add_executable(tetris-ai ${SOURCES})

# Enable AddressSanitizer
target_compile_options(tetris-ai PRIVATE -fsanitize=address)
target_link_options(tetris-ai PRIVATE -fsanitize=address)
# Variables storing SDL framework locations
set(SDL2 /Library/Frameworks/SDL2.framework)
set(SDL2_image /Library/Frameworks/SDL2_image.framework)
set(SDL2_ttf /Library/Frameworks/SDL2_ttf.framework)

target_link_libraries(
  tetris-ai PRIVATE
  ${SDL2}/Versions/A/SDL2
  ${SDL2_image}/Versions/A/SDL2_image
  ${SDL2_ttf}/Versions/A/SDL2_ttf
  mlx
)

target_include_directories(
  tetris-ai PRIVATE
  ${SDL2}/Versions/A/Headers
  ${SDL2_image}/Versions/A/Headers
  ${SDL2_ttf}/Versions/A/Headers
  ./Gameplay
  ./AI
  ./Helpers
)
add_definitions(-DMETAL_PATH="/Users/christopheprat/Code/TIPE/Tetris Cpp/mlx/mlx/backend/metal/kernels/mlx.metallib")


add_custom_command(
  TARGET tetris-ai POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  ${CMAKE_SOURCE_DIR}/resources/lmroman17-regular.otf
  ${CMAKE_BINARY_DIR}/resources/lmroman17-regular.otf
)
