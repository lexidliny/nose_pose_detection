
find_package(OpenCV REQUIRED)

set(LIB_NAME noseDetect)

find_package(PCL 1.8 REQUIRED)

add_library(noseDetect STATIC ${LIB_NAME}.cpp)

set_target_properties(noseDetect PROPERTIES 
                    ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib
                     )
add_definitions(${PCL_DEFINITIONS})


target_include_directories(noseDetect 
                    PUBLIC 
                           ${PROJECT_SOURCE_DIR}/include
                           ${PCL_INCLUDE_DIRS}
                           ${PROJECT_SOURCE_DIR}/src/facemesh
                           ${PROJECT_SOURCE_DIR}/tflite-dist/include/
                           )

target_link_directories(${LIB_NAME} 
                    PUBLIC
                        ${PROJECT_SOURCE_DIR}/lib
                        ${PCL_LIBRARY_DIRS}
                        )

message(${PROJECT_SOURCE_DIR}/include)

target_link_libraries(noseDetect 
                PUBLIC 
                        ${OpenCV_LIBS}
                        ${PCL_LIBRARIES}
                PRIVATE 
                        rs2pcd
                        )

