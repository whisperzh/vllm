#include <torch/extension.h>
#include <cuda_runtime.h>
#include <iostream>

torch::Tensor open_ipc_tensor(torch::Tensor shape_tensor, const std::string& dtype_str, py::bytes handle_bytes, int device) {
    // 1. Validate dtype
    c10::ScalarType dtype;
    size_t element_size;

    if (dtype_str == "float32") {
        dtype = c10::ScalarType::Float;
        element_size = 4;
    } else if (dtype_str == "int64") {
        dtype = c10::ScalarType::Long;
        element_size = 8;
    } else if (dtype_str == "float64") {
        dtype = c10::ScalarType::Double;
        element_size = 8;
    } else if (dtype_str == "int32") {
        dtype = c10::ScalarType::Int;
        element_size = 4;
    } else if (dtype_str == "uint8") {
        dtype = c10::ScalarType::Byte;
        element_size = 1;
    } else {
        throw std::invalid_argument("Unsupported dtype: " + dtype_str);
    }

    // 2. Convert handle to cudaIpcMemHandle_t
    if (py::len(handle_bytes) != sizeof(cudaIpcMemHandle_t)) {
        throw std::runtime_error("Invalid IPC handle size");
    }

    cudaIpcMemHandle_t handle;
    std::memcpy(&handle, std::string(handle_bytes).data(), sizeof(handle));

    // 3. Set device
    cudaSetDevice(device);

    // 4. Open IPC memory
    void* dev_ptr = nullptr;
    cudaError_t err = cudaIpcOpenMemHandle(&dev_ptr, handle, cudaIpcMemLazyEnablePeerAccess);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaIpcOpenMemHandle failed: " + std::string(cudaGetErrorString(err)));
    }

    // 5. Shape
    std::vector<int64_t> shape;
    shape.reserve(shape_tensor.numel());
    for (int64_t i = 0; i < shape_tensor.numel(); ++i) {
        shape.push_back(shape_tensor[i].item<int64_t>());
    }

    size_t numel = 1;
    for (auto s : shape) numel *= s;

    // 6. Wrap into tensor (no memory ownership)
    auto options = torch::TensorOptions()
        .device(torch::kCUDA, device)
        .dtype(dtype);

    auto tensor = torch::from_blob(dev_ptr, shape, options);
    return tensor;
}

PYBIND11_MODULE(ipc_tensor_read, m) {
    m.def("open_ipc_tensor", &open_ipc_tensor, "Open IPC Tensor",
          py::arg("shape_tensor"), py::arg("dtype_str"),
          py::arg("handle_bytes"), py::arg("device") = 0);
}
