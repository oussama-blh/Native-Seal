cmake_minimum_required(VERSION 3.10)
project(NativeSEALProject)

# Add this line to suppress C4267 warnings
if(MSVC)
    add_compile_options(/wd4267)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Path to LibTorch (Updated to match your new directory structure)
set(Torch_DIR "C:/Khbich/PFE/Implementations/NativeSEAL/lib/libtorch/share/cmake/Torch")

# Find LibTorch
find_package(Torch REQUIRED)

# Add executable with all source files
add_executable(NativeSealApp
    main.cpp
    src/he/he.cpp
    src/convolution/convolution.cpp
    src/pooling/avgPooling.cpp
    src/flatten/flatten.cpp
    src/linear/linear.cpp
    src/functions/square.cpp
    src/pooling/adaptiveAvgPooling.cpp
)

# Find and link OpenMP **AFTER** defining the executable
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(NativeSealApp PRIVATE OpenMP::OpenMP_CXX)
endif()

# Include directories for project and dependencies
target_include_directories(NativeSealApp
    PRIVATE
        "${CMAKE_SOURCE_DIR}/src"  # Include custom headers
        "${CMAKE_SOURCE_DIR}/lib/SEAL/install/include/SEAL-4.1"  # SEAL headers
        "${TORCH_INCLUDE_DIRS}"  # Include LibTorch headers
)

# Link directories for SEAL
target_link_directories(NativeSealApp
    PRIVATE
        "${CMAKE_SOURCE_DIR}/lib/SEAL/install/lib"
)

# Link to SEAL and LibTorch
target_link_libraries(NativeSealApp
    PRIVATE
        seal-4.1
        "${TORCH_LIBRARIES}"  # Link LibTorch
)

# Required for linking LibTorch on Windows
set_property(TARGET NativeSealApp PROPERTY CXX_STANDARD 17)
set_property(TARGET NativeSealApp PROPERTY RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# Ensure LibTorch is linked dynamically
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Add after other configurations
if(MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET NativeSealApp POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${TORCH_DLLS}
        $<TARGET_FILE_DIR:NativeSealApp>
    )
endif()
