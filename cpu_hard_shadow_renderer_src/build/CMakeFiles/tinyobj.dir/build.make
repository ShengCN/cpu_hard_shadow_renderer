# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.18

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/ysheng/Downloads/cmake-3.18.3-Linux-x86_64/bin/cmake

# The command to remove a file.
RM = /home/ysheng/Downloads/cmake-3.18.3-Linux-x86_64/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ysheng/Documents/paper_project/adobe/cpu_hard_shadow_renderer/cpu_hard_shadow_renderer_src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ysheng/Documents/paper_project/adobe/cpu_hard_shadow_renderer/cpu_hard_shadow_renderer_src/build

# Include any dependencies generated for this target.
include CMakeFiles/tinyobj.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/tinyobj.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/tinyobj.dir/flags.make

CMakeFiles/tinyobj.dir/tinyobjloader/tiny_obj_loader.cc.o: CMakeFiles/tinyobj.dir/flags.make
CMakeFiles/tinyobj.dir/tinyobjloader/tiny_obj_loader.cc.o: ../tinyobjloader/tiny_obj_loader.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ysheng/Documents/paper_project/adobe/cpu_hard_shadow_renderer/cpu_hard_shadow_renderer_src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/tinyobj.dir/tinyobjloader/tiny_obj_loader.cc.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tinyobj.dir/tinyobjloader/tiny_obj_loader.cc.o -c /home/ysheng/Documents/paper_project/adobe/cpu_hard_shadow_renderer/cpu_hard_shadow_renderer_src/tinyobjloader/tiny_obj_loader.cc

CMakeFiles/tinyobj.dir/tinyobjloader/tiny_obj_loader.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tinyobj.dir/tinyobjloader/tiny_obj_loader.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ysheng/Documents/paper_project/adobe/cpu_hard_shadow_renderer/cpu_hard_shadow_renderer_src/tinyobjloader/tiny_obj_loader.cc > CMakeFiles/tinyobj.dir/tinyobjloader/tiny_obj_loader.cc.i

CMakeFiles/tinyobj.dir/tinyobjloader/tiny_obj_loader.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tinyobj.dir/tinyobjloader/tiny_obj_loader.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ysheng/Documents/paper_project/adobe/cpu_hard_shadow_renderer/cpu_hard_shadow_renderer_src/tinyobjloader/tiny_obj_loader.cc -o CMakeFiles/tinyobj.dir/tinyobjloader/tiny_obj_loader.cc.s

# Object files for target tinyobj
tinyobj_OBJECTS = \
"CMakeFiles/tinyobj.dir/tinyobjloader/tiny_obj_loader.cc.o"

# External object files for target tinyobj
tinyobj_EXTERNAL_OBJECTS =

libtinyobj.a: CMakeFiles/tinyobj.dir/tinyobjloader/tiny_obj_loader.cc.o
libtinyobj.a: CMakeFiles/tinyobj.dir/build.make
libtinyobj.a: CMakeFiles/tinyobj.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ysheng/Documents/paper_project/adobe/cpu_hard_shadow_renderer/cpu_hard_shadow_renderer_src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libtinyobj.a"
	$(CMAKE_COMMAND) -P CMakeFiles/tinyobj.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/tinyobj.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/tinyobj.dir/build: libtinyobj.a

.PHONY : CMakeFiles/tinyobj.dir/build

CMakeFiles/tinyobj.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/tinyobj.dir/cmake_clean.cmake
.PHONY : CMakeFiles/tinyobj.dir/clean

CMakeFiles/tinyobj.dir/depend:
	cd /home/ysheng/Documents/paper_project/adobe/cpu_hard_shadow_renderer/cpu_hard_shadow_renderer_src/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ysheng/Documents/paper_project/adobe/cpu_hard_shadow_renderer/cpu_hard_shadow_renderer_src /home/ysheng/Documents/paper_project/adobe/cpu_hard_shadow_renderer/cpu_hard_shadow_renderer_src /home/ysheng/Documents/paper_project/adobe/cpu_hard_shadow_renderer/cpu_hard_shadow_renderer_src/build /home/ysheng/Documents/paper_project/adobe/cpu_hard_shadow_renderer/cpu_hard_shadow_renderer_src/build /home/ysheng/Documents/paper_project/adobe/cpu_hard_shadow_renderer/cpu_hard_shadow_renderer_src/build/CMakeFiles/tinyobj.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/tinyobj.dir/depend

