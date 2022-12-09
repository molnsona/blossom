add_library(
        glad
        src/glad.c
        include/glad/glad.h
)

target_include_directories(glad PUBLIC include)
