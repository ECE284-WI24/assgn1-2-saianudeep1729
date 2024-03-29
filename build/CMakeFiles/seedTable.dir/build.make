# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/snayini/assgn1-2-saianudeep1729

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/snayini/assgn1-2-saianudeep1729/build

# Include any dependencies generated for this target.
include CMakeFiles/seedTable.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/seedTable.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/seedTable.dir/flags.make

CMakeFiles/seedTable.dir/src/seedTable.cu.o: CMakeFiles/seedTable.dir/flags.make
CMakeFiles/seedTable.dir/src/seedTable.cu.o: ../src/seedTable.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/snayini/assgn1-2-saianudeep1729/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/seedTable.dir/src/seedTable.cu.o"
	/usr/local/cuda/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/snayini/assgn1-2-saianudeep1729/src/seedTable.cu -o CMakeFiles/seedTable.dir/src/seedTable.cu.o

CMakeFiles/seedTable.dir/src/seedTable.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/seedTable.dir/src/seedTable.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/seedTable.dir/src/seedTable.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/seedTable.dir/src/seedTable.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/seedTable.dir/src/twoBitCompressor.cpp.o: CMakeFiles/seedTable.dir/flags.make
CMakeFiles/seedTable.dir/src/twoBitCompressor.cpp.o: ../src/twoBitCompressor.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/snayini/assgn1-2-saianudeep1729/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/seedTable.dir/src/twoBitCompressor.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/seedTable.dir/src/twoBitCompressor.cpp.o -c /home/snayini/assgn1-2-saianudeep1729/src/twoBitCompressor.cpp

CMakeFiles/seedTable.dir/src/twoBitCompressor.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/seedTable.dir/src/twoBitCompressor.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/snayini/assgn1-2-saianudeep1729/src/twoBitCompressor.cpp > CMakeFiles/seedTable.dir/src/twoBitCompressor.cpp.i

CMakeFiles/seedTable.dir/src/twoBitCompressor.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/seedTable.dir/src/twoBitCompressor.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/snayini/assgn1-2-saianudeep1729/src/twoBitCompressor.cpp -o CMakeFiles/seedTable.dir/src/twoBitCompressor.cpp.s

CMakeFiles/seedTable.dir/src/main.cpp.o: CMakeFiles/seedTable.dir/flags.make
CMakeFiles/seedTable.dir/src/main.cpp.o: ../src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/snayini/assgn1-2-saianudeep1729/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/seedTable.dir/src/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/seedTable.dir/src/main.cpp.o -c /home/snayini/assgn1-2-saianudeep1729/src/main.cpp

CMakeFiles/seedTable.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/seedTable.dir/src/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/snayini/assgn1-2-saianudeep1729/src/main.cpp > CMakeFiles/seedTable.dir/src/main.cpp.i

CMakeFiles/seedTable.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/seedTable.dir/src/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/snayini/assgn1-2-saianudeep1729/src/main.cpp -o CMakeFiles/seedTable.dir/src/main.cpp.s

# Object files for target seedTable
seedTable_OBJECTS = \
"CMakeFiles/seedTable.dir/src/seedTable.cu.o" \
"CMakeFiles/seedTable.dir/src/twoBitCompressor.cpp.o" \
"CMakeFiles/seedTable.dir/src/main.cpp.o"

# External object files for target seedTable
seedTable_EXTERNAL_OBJECTS =

seedTable: CMakeFiles/seedTable.dir/src/seedTable.cu.o
seedTable: CMakeFiles/seedTable.dir/src/twoBitCompressor.cpp.o
seedTable: CMakeFiles/seedTable.dir/src/main.cpp.o
seedTable: CMakeFiles/seedTable.dir/build.make
seedTable: /usr/lib/x86_64-linux-gnu/libboost_program_options.so.1.71.0
seedTable: /usr/lib/x86_64-linux-gnu/libz.so
seedTable: tbb_cmake_build/tbb_cmake_build_subdir_release/libtbbmalloc_proxy.so.2
seedTable: tbb_cmake_build/tbb_cmake_build_subdir_release/libtbb_preview.so.2
seedTable: tbb_cmake_build/tbb_cmake_build_subdir_release/libtbbmalloc.so.2
seedTable: CMakeFiles/seedTable.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/snayini/assgn1-2-saianudeep1729/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable seedTable"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/seedTable.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/seedTable.dir/build: seedTable

.PHONY : CMakeFiles/seedTable.dir/build

CMakeFiles/seedTable.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/seedTable.dir/cmake_clean.cmake
.PHONY : CMakeFiles/seedTable.dir/clean

CMakeFiles/seedTable.dir/depend:
	cd /home/snayini/assgn1-2-saianudeep1729/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/snayini/assgn1-2-saianudeep1729 /home/snayini/assgn1-2-saianudeep1729 /home/snayini/assgn1-2-saianudeep1729/build /home/snayini/assgn1-2-saianudeep1729/build /home/snayini/assgn1-2-saianudeep1729/build/CMakeFiles/seedTable.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/seedTable.dir/depend

