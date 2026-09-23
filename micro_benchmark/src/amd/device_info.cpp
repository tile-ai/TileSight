#include "common.hpp"

int main(int argc, char** argv) {
    try {
        const auto properties = select_device(argc, argv);
        int device = 0, runtime = 0;
        size_t free_bytes = 0, total_bytes = 0;
        char pci[32]{};
        HIP_CHECK(hipGetDevice(&device));
        HIP_CHECK(hipRuntimeGetVersion(&runtime));
        HIP_CHECK(hipMemGetInfo(&free_bytes, &total_bytes));
        HIP_CHECK(hipDeviceGetPCIBusId(pci, sizeof(pci), device));
        std::cout << "{\"name\":" << json_string(properties.name)
                  << ",\"gcn_arch_name\":" << json_string(properties.gcnArchName)
                  << ",\"visible_device\":" << device << ",\"pci_bus_id\":" << json_string(pci)
                  << ",\"compute_units\":" << properties.multiProcessorCount
                  << ",\"warp_size\":" << properties.warpSize
                  << ",\"max_threads_per_block\":" << properties.maxThreadsPerBlock
                  << ",\"l2_bytes\":" << properties.l2CacheSize
                  << ",\"total_memory_bytes\":" << total_bytes
                  << ",\"free_memory_bytes\":" << free_bytes
                  << ",\"reported_max_clock_khz\":" << properties.clockRate
                  << ",\"hip_runtime_version\":" << runtime << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
