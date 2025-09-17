# CMake generated Testfile for 
# Source directory: /home/jmhu/columnar_memtable/mimalloc
# Build directory: /home/jmhu/columnar_memtable/build/mimalloc
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(test-api "/home/jmhu/columnar_memtable/build/mimalloc/mimalloc-test-api")
set_tests_properties(test-api PROPERTIES  _BACKTRACE_TRIPLES "/home/jmhu/columnar_memtable/mimalloc/CMakeLists.txt;723;add_test;/home/jmhu/columnar_memtable/mimalloc/CMakeLists.txt;0;")
add_test(test-api-fill "/home/jmhu/columnar_memtable/build/mimalloc/mimalloc-test-api-fill")
set_tests_properties(test-api-fill PROPERTIES  _BACKTRACE_TRIPLES "/home/jmhu/columnar_memtable/mimalloc/CMakeLists.txt;723;add_test;/home/jmhu/columnar_memtable/mimalloc/CMakeLists.txt;0;")
add_test(test-stress "/home/jmhu/columnar_memtable/build/mimalloc/mimalloc-test-stress")
set_tests_properties(test-stress PROPERTIES  _BACKTRACE_TRIPLES "/home/jmhu/columnar_memtable/mimalloc/CMakeLists.txt;723;add_test;/home/jmhu/columnar_memtable/mimalloc/CMakeLists.txt;0;")
