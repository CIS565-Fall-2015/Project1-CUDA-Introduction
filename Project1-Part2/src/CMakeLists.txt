set(SOURCE_FILES
    "matrix_math.hpp"
    "matrix_math.cu"
    )

cuda_add_library(src
    ${SOURCE_FILES}
    OPTIONS -arch=sm_20
    )
