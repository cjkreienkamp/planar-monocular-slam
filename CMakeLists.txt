cmake_minimum_required(VERSION 3.10)
project(BundleAdjustmentG2O)

# Define G2O install path from environment
set(G2O_INSTALL_DIR "$ENV{HOME}/g2o-install")

# Set path to g2o CMake config
set(g2o_DIR "${G2O_INSTALL_DIR}/lib/cmake/g2o")

find_package(Eigen3 REQUIRED)
find_package(g2o REQUIRED)

# Include headers
include_directories(${G2O_INSTALL_DIR}/include)
include_directories(${G2O_INCLUDE_DIRS})

# Link directories BEFORE linking libraries
link_directories(${G2O_INSTALL_DIR}/lib)

# Define executable
add_executable(bundle_adjustment src/bundle_adjustment.cpp)

# Link libraries
target_link_libraries(bundle_adjustment
    g2o_core
    g2o_stuff
    g2o_types_sba
    g2o_types_slam3d
    g2o_solver_csparse
    g2o_csparse_extension
    g2o_solver_dense
    g2o_solver_eigen
    Eigen3::Eigen
    fmt::fmt
)