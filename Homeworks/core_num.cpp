#include <omp.h>
#include <iostream>

int main() {
    int numCores = omp_get_num_procs();
    std::cout << "Number of cores: " << numCores << "\n";

    return 0;
}
