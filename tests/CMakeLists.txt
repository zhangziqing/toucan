add_executable(check-design-graph check_design_graph.cpp)
llvm_update_compile_flags(check-design-graph)
target_link_libraries(check-design-graph PRIVATE Toucan ToucanAnalysis ToucanUtils MLIRParser)

add_executable(check-local-value-allocator check_local_value_allocator.cpp)
llvm_update_compile_flags(check-local-value-allocator)
target_link_libraries(check-local-value-allocator PRIVATE Toucan ToucanAnalysis ToucanUtils MLIRParser)

add_executable(check-micro-part-merging check_micro_part_merging.cpp)
llvm_update_compile_flags(check-micro-part-merging)
target_link_libraries(check-micro-part-merging PRIVATE Toucan ToucanAnalysis ToucanUtils)

add_executable(check-wide-shift check_wide_shift.cpp)
llvm_update_compile_flags(check-wide-shift)
target_link_libraries(check-wide-shift PRIVATE Toucan ToucanPasses ToucanUtils MLIRParser)

add_executable(check-array-index check_array_index.cpp)
llvm_update_compile_flags(check-array-index)
target_link_libraries(check-array-index PRIVATE Toucan ToucanPasses ToucanUtils MLIRParser)

if(BUILD_TESTING)
  foreach(check IN ITEMS design-graph local-value-allocator micro-part-merging wide-shift array-index)
    add_test(NAME ${check} COMMAND check-${check})
  endforeach()
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  add_test(NAME strict-two-round COMMAND ${Python3_EXECUTABLE}
           ${CMAKE_CURRENT_SOURCE_DIR}/check_two_round.py $<TARGET_FILE:toucan>)
endif()
