#include <iostream>
#include "common.h"

void main() {
	std::string test_str = "abc.txt";
	std::cerr << purdue::get_file_ext(test_str) << std::endl;

	system("pause");
}