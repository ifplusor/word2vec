cmake_minimum_required(VERSION 2.8)
project(word2vec)

#set(CMAKE_C_STAND 99)
#set(CMAKE_CXX_STAND 11)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

add_compile_options(-O2 -Wall -funroll-loops -Wno-unused-result)
link_libraries(m pthread)

add_executable(word2vec
        src/word2vec.c)

add_executable(word2phrase
        src/word2phrase.c)

add_executable(distance
        src/distance.c)

add_executable(word-analogy
        src/word-analogy.c)

add_executable(compute-accuracy
        src/compute-accuracy.c)
