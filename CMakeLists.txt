cmake_minimum_required(VERSION 3.15)

project(DistanciaEuclidianaOtimizado LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCL REQUIRED)

set(SRCS
  src/main.cpp
)

add_executable(distancia_euclidiana ${SRCS})
target_link_libraries(distancia_euclidiana PUBLIC OpenCL::OpenCL)
