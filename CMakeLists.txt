cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 23)

project(Sandbox VERSION 1.0.0)

# Add all source files
file(GLOB_RECURSE SOURCES "Gameplay/*.cpp" "AI/*.cpp" "Helpers/*.cpp")

add_executable(Sandbox ${SOURCES})

# Enable AddressSanitizer
target_compile_options(Sandbox PRIVATE -fsanitize=address)
target_link_options(Sandbox PRIVATE -fsanitize=address)
# Variables storing SDL framework locations
set(SDL2 /Library/Frameworks/SDL2.framework)
set(SDL2_image /Library/Frameworks/SDL2_image.framework)
set(SDL2_ttf /Library/Frameworks/SDL2_ttf.framework)

target_link_libraries(
  Sandbox PRIVATE
  ${SDL2}/Versions/A/SDL2
  ${SDL2_image}/Versions/A/SDL2_image
  ${SDL2_ttf}/Versions/A/SDL2_ttf
)

target_include_directories(
  Sandbox PRIVATE
  ${SDL2}/Versions/A/Headers
  ${SDL2_image}/Versions/A/Headers
  ${SDL2_ttf}/Versions/A/Headers
  ./Gameplay
  ./AI
  ./Helpers
)
add_custom_command(
  TARGET Sandbox POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  ${CMAKE_SOURCE_DIR}/resources/lmroman17-regular.otf
  ${CMAKE_BINARY_DIR}/resources/lmroman17-regular.otf
)