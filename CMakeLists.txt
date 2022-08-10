cmake_minimum_required(VERSION 3.1.0)

project(nose_pose)

add_subdirectory(${PROJECT_SOURCE_DIR}/src/noseDetec)
add_subdirectory(${PROJECT_SOURCE_DIR}/src/rs2pcd)
add_subdirectory(${PROJECT_SOURCE_DIR}/src/facemesh)

add_executable(${PROJECT_NAME} ${PROJECT_NAME}.cpp)

target_link_libraries(${PROJECT_NAME} 
                        noseDetect
                        facemesh
                     )

message(${PROJECT_SOURCE_DIR})


