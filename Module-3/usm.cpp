// ================================================================
//            SYCL USM EXAMPLE: ALLOCATION AND ACCESS 
// ================================================================

#include <sycl/sycl.hpp>
#include <iostream>

using namespace sycl;
using namespace std;

int main() {

    const int N = 1000;

    // Create a SYCL queue
    queue q{ cpu_selector_v };

    cout << "SYCL Device: "
         << q.get_device().get_info<info::device::name>() << "\n\n";

    // -------------------- USM SHARED ALLOCATION --------------------
    int* data = malloc_shared<int>(N, q);

    // Initialize data 
    for (int i = 0; i < N; i++) {
        data[i] = i * 10;
    }

    // Print data 
    cout << "USM Shared Data:\n";
    for (int i = 0; i < N; i++) {
        cout << "data[" << i << "] = " << data[i] << "\n";
    }

    // Free USM memory
    free(data, q);

    return 0;
}
