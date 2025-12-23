#include <sycl/sycl.hpp>
#include <iostream>

using namespace sycl;
using namespace std;

int main() {

    // Create a queue using CPU selector
    queue q{ cpu_selector_v };

    // Query and print device information
    device dev = q.get_device();

    cout << "Selected SYCL Device Information:\n";
    cout << "--------------------------------\n";
    cout << "Device Name      : "
         << dev.get_info<info::device::name>() << "\n";
    cout << "Vendor           : "
         << dev.get_info<info::device::vendor>() << "\n";
	
    return 0;
}

