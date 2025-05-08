#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <filesystem>

extern "C" {
    // Function to open IPC handle and return a pointer to the device memory
    void* open_ipc_handle(const char* handle_file) {
        cudaIpcMemHandle_t handle;
        std::ifstream file(handle_file, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Failed to open handle file: " << handle_file << std::endl;
            return nullptr;
        }
        file.read(reinterpret_cast<char*>(&handle), sizeof(cudaIpcMemHandle_t));
        file.close();

        float* device_ptr;
        cudaError_t err = cudaIpcOpenMemHandle((void**)&device_ptr, handle, cudaIpcMemLazyEnablePeerAccess);
        if (err != cudaSuccess) {
            std::cerr << "Failed to open IPC handle: " << cudaGetErrorString(err) << std::endl;
            return nullptr;
        }

        return device_ptr;
    }
}