#include <cuda_runtime.h>
#include <iostream>
#include <vector>

extern "C" {
    // Function to open IPC handle and return a pointer to the device memory
    void* open_ipc_handle(const void* handle_bytes) {
        cudaIpcMemHandle_t handle;
        // Copy handle_bytes to handle
        memcpy(&handle, handle_bytes, sizeof(cudaIpcMemHandle_t));

        void* device_ptr;
        cudaError_t err = cudaIpcOpenMemHandle(&device_ptr, handle, cudaIpcMemLazyEnablePeerAccess);
        if (err != cudaSuccess) {
            std::cerr << "Failed to open IPC handle: " << cudaGetErrorString(err) << std::endl;
            return nullptr;
        }

        return device_ptr;
    }
}