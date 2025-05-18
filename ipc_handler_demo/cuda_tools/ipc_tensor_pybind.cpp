#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <string>
#include <vector>

namespace py = pybind11;

// Function to open an IPC handle and return a PyTorch tensor
torch::Tensor open_ipc_tensor_pybind(py::bytes handle_bytes, 
                                     int device,
                                     std::vector<int64_t> shape,
                                     torch::ScalarType dtype) {
    // Get the size of the bytes object using PyBytes_Size
    Py_ssize_t size = PyBytes_Size(handle_bytes.ptr());
    if (size != sizeof(cudaIpcMemHandle_t)) {
        throw std::runtime_error("Invalid IPC handle size");
    }

    // Set CUDA device
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaSetDevice failed: ") + cudaGetErrorString(err));
    }

    // Copy handle from bytes
    cudaIpcMemHandle_t handle;
    std::memcpy(&handle, PyBytes_AsString(handle_bytes.ptr()), sizeof(cudaIpcMemHandle_t));

    void* dev_ptr = nullptr;
    err = cudaIpcOpenMemHandle(&dev_ptr, handle, cudaIpcMemLazyEnablePeerAccess);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaIpcOpenMemHandle failed: ") + cudaGetErrorString(err));
    }

    // Calculate total number of elements
    int64_t numel = 1;
    for (auto dim : shape) {
        numel *= dim;
    }

    // Create PyTorch tensor from device pointer
    auto options = torch::TensorOptions()
        .dtype(dtype)
        .device(torch::kCUDA, device);
    
    // Create tensor without copying data
    auto tensor = torch::from_blob(dev_ptr, shape, 
        // Deleter function to close the IPC handle when tensor is deleted
        [dev_ptr](void*) {
            cudaIpcCloseMemHandle(dev_ptr);
        }, 
        options);
    
    // Return a clone to ensure proper memory management
    return tensor.clone();
}

// Function to export an IPC handle from a PyTorch tensor
py::bytes export_ipc_handle_pybind(torch::Tensor tensor) {
    if (!tensor.is_cuda()) {
        throw std::runtime_error("Tensor must be on a CUDA device");
    }

    void* dev_ptr = tensor.data_ptr();
    cudaIpcMemHandle_t handle;
    
    cudaError_t err = cudaIpcGetMemHandle(&handle, dev_ptr);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaIpcGetMemHandle failed: ") + cudaGetErrorString(err));
    }

    // Convert handle to bytes
    return py::bytes(reinterpret_cast<const char*>(&handle), sizeof(cudaIpcMemHandle_t));
}

PYBIND11_MODULE(ipc_tensor_pybind, m) {
    m.doc() = "PyBind11 module for CUDA IPC tensor handling";
    
    m.def("open_ipc_tensor", &open_ipc_tensor_pybind, 
          "Open an IPC handle and return a PyTorch tensor",
          py::arg("handle_bytes"), py::arg("device"), 
          py::arg("shape"), py::arg("dtype"));
    
    m.def("export_ipc_handle", &export_ipc_handle_pybind,
          "Export an IPC handle from a PyTorch tensor",
          py::arg("tensor"));
}