#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>

extern "C" {

// handle_bytes must point to a 64-byte cudaIpcMemHandle_t
// device is the GPU index to set as context
void* open_ipc_tensor(const void* handle_bytes, int device) {
    if (handle_bytes == nullptr) {
        fprintf(stderr, "open_ipc_tensor: null handle input\n");
        return nullptr;
    }

    // Set CUDA device
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice(%d) failed: %s\n", device, cudaGetErrorString(err));
        return nullptr;
    }

    // Copy handle from bytes
    cudaIpcMemHandle_t handle;
    memcpy(&handle, handle_bytes, sizeof(cudaIpcMemHandle_t));

    void* dev_ptr = nullptr;
    err = cudaIpcOpenMemHandle(&dev_ptr, handle, cudaIpcMemLazyEnablePeerAccess);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaIpcOpenMemHandle failed: %s\n", cudaGetErrorString(err));
        return nullptr;
    }

    return dev_ptr;
}

// Optionally: close it later
int close_ipc_tensor(void* dev_ptr) {
    if (dev_ptr == nullptr) return 0;
    cudaError_t err = cudaIpcCloseMemHandle(dev_ptr);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaIpcCloseMemHandle failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

// Export IPC handle from a given device pointer
// out_handle must point to 64-byte buffer (cudaIpcMemHandle_t)
int export_ipc_handle(void* dev_ptr, void* out_handle) {
    if (dev_ptr == nullptr || out_handle == nullptr) {
        fprintf(stderr, "Invalid null pointer input.\n");
        return -1;
    }

    cudaIpcMemHandle_t handle;
    cudaError_t err = cudaIpcGetMemHandle(&handle, dev_ptr);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaIpcGetMemHandle failed: %s\n", cudaGetErrorString(err));
        return -2;
    }

    // memcpy(out_handle, &handle, sizeof(cudaIpcMemHandle_t));
    memcpy(out_handle,reinterpret_cast<const char*>(&handle) , sizeof(cudaIpcMemHandle_t));
    return 0;
}

}
