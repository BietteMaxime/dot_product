if(EIGEN3_INCLUDE_DIR)
  set(EIGEN3_FOUND TRUE)
else(EIGEN3_INCLUDE_DIR)
  find_path(EIGEN3_INCLUDE_DIR NAMES Eigen/Core
    PATH_SUFFIXES eigen3/
    HINTS
    /usr/local/include)
    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(Eigen3 DEFAULT_MSG EIGEN3_INCLUDE_DIR)
    mark_as_advanced(EIGEN3_INCLUDE_DIR)
endif()
